"""ToolQA tool registry and environment dispatcher."""

import re
import time
from pathlib import Path


def parse_action(action_str: str) -> tuple[str, str] | tuple[None, None]:
    """Parse ActionType[argument] into action type and argument."""
    if action_str is None:
        return None, None

    action_str = action_str.strip()

    if action_str.startswith("PythonInterpreter[") and action_str.endswith("]"):
        return "PythonInterpreter", action_str[18:-1]

    pattern = r"^(\w+)\[(.+)\]$"
    match = re.match(pattern, action_str, re.DOTALL)
    if match:
        return match.group(1), match.group(2)
    return None, None


class ToolEnvironment:
    """Manages tool state and dispatches actions for ToolQA."""

    def __init__(self, corpus_dir: Path):
        from skillrag_vendor.toolqa.tools.graph import GraphToolkit
        from skillrag_vendor.toolqa.tools.table import TableToolkit

        self._corpus_dir = Path(corpus_dir)

        self.table = TableToolkit(self._corpus_dir)
        self.graph = GraphToolkit(self._corpus_dir)

        self._agenda_retriever = None
        self._scirex_retriever = None

        self._sql_conn = None
        self._sql_loaded_tables: set[str] = set()

    def _ensure_retrievers_ready(self) -> None:
        """Preload retrievers to avoid rebuilding embeddings for each case."""
        self._get_agenda_retriever()
        self._get_scirex_retriever()

    def reset(self) -> None:
        """Reset per-instance mutable state."""
        self.table.data = None
        self.graph.paper_net = None
        self.graph.author_net = None
        self.graph.id2title_dict = None
        self.graph.title2id_dict = None
        self.graph.id2author_dict = None
        self.graph.author2id_dict = None

    @property
    def sql_conn(self):
        if self._sql_conn is None:
            import sqlite3

            self._sql_conn = sqlite3.connect(":memory:")
        return self._sql_conn

    def _get_agenda_retriever(self):
        if self._agenda_retriever is None:
            from skillrag_vendor.toolqa.tools.text import get_shared_retriever

            self._agenda_retriever = get_shared_retriever(
                corpus_path=self._corpus_dir / "agenda" / "agenda_descriptions_merged.jsonl",
                text_field="event",
            )
        return self._agenda_retriever

    def _get_scirex_retriever(self):
        if self._scirex_retriever is None:
            from skillrag_vendor.toolqa.tools.text import get_shared_retriever

            self._scirex_retriever = get_shared_retriever(
                corpus_path=self._corpus_dir / "scirex" / "Preprocessed_Scirex.jsonl",
                text_field="content",
            )
        return self._scirex_retriever

    def execute(self, action_str: str) -> str:
        """Parse action string, dispatch to tool, return observation."""
        if action_str is None or action_str.strip() == "":
            return (
                "You action is filtered due to content. "
                "Please assume all the actions are permitted in this "
                "environment and take the action again."
            )

        if "], " in action_str and not action_str.startswith("PythonInterpreter["):
            return "You are sending multiple actions at once. Please send one action at a time."

        action_type, argument = parse_action(action_str)

        if action_type is None:
            return (
                "Invalid Action. Valid Actions are "
                "Calculate[<Formula>] RetrieveAgenda[<Content>] "
                "RetrieveScirex[<Content>] LoadDB[<DBName>] "
                "FilterDB[<Condition>, <Condition>, ...] GetValue[<Column>] "
                "LoadGraph[<GraphName>] NeighbourCheck[<GraphName>, <Node>] "
                "NodeCheck[<GraphName>, <Node>] "
                "EdgeCheck[<GraphName>, <Node1>, <Node2>] "
                "SQLInterpreter[<SQLCommand>] "
                "PythonInterpreter[<PythonCode>] and Finish[<answer>]."
            )

        try:
            return self._dispatch(action_type, argument)
        except Exception as exc:
            return f"Error executing {action_type}: {exc}"

    def _dispatch(self, action_type: str, argument: str) -> str:
        from skillrag_vendor.toolqa.tools.calculator import calculate
        from skillrag_vendor.toolqa.tools.code import python_interpret, sql_interpret

        dispatch_start = time.time()

        if action_type == "Finish":
            return argument

        if action_type == "Calculate":
            try:
                result = calculate(argument)
                print(f"[ToolEnv] Calculate took {time.time() - dispatch_start:.3f}s")
                return result
            except Exception:
                return "Illegal Mathematical Expression. Please try again."

        if action_type == "RetrieveAgenda":
            try:
                result = self._get_agenda_retriever().query(argument)
                print(f"[ToolEnv] RetrieveAgenda took {time.time() - dispatch_start:.3f}s, query='{argument[:50]}...'")
                return result
            except Exception:
                return "There is no information that can be matched in the database. Please try another query."

        if action_type == "RetrieveScirex":
            try:
                result = self._get_scirex_retriever().query(argument)
                print(f"[ToolEnv] RetrieveScirex took {time.time() - dispatch_start:.3f}s, query='{argument[:50]}...'")
                return result
            except Exception:
                return "There is no information that can be matched in the database. Please try another query."

        if action_type == "LoadDB":
            try:
                load_start = time.time()
                result = self.table.load_db(argument)
                print(f"[ToolEnv] LoadDB '{argument}' took {time.time() - load_start:.3f}s")
                if self.table.data is not None and argument not in self._sql_loaded_tables:
                    sql_start = time.time()
                    self.table.data.to_sql(
                        f"{argument}_data",
                        self.sql_conn,
                        if_exists="replace",
                        index=False,
                    )
                    self._sql_loaded_tables.add(argument)
                    print(f"[ToolEnv] to_sql took {time.time() - sql_start:.3f}s")
                return result
            except Exception:
                return "The database you want to query is not in the list. Please change another database for query."

        if action_type == "FilterDB":
            try:
                result = self.table.filter_db(argument)
                print(f"[ToolEnv] FilterDB took {time.time() - dispatch_start:.3f}s")
                return result
            except Exception:
                return "There is something wrong with the arguments you send for filtering. Please modify it."

        if action_type == "GetValue":
            try:
                result = self.table.get_value(argument)
                print(f"[ToolEnv] GetValue took {time.time() - dispatch_start:.3f}s")
                return result
            except Exception:
                return "The value you are querying does not exist. Please modify it."

        if action_type == "LoadGraph":
            try:
                load_start = time.time()
                result = self.graph.load_graph(argument)
                print(f"[ToolEnv] LoadGraph '{argument}' took {time.time() - load_start:.3f}s")
                return result
            except Exception:
                return "The graph you want to query is not in the list. Please change another graph for query."

        if action_type == "NeighbourCheck":
            try:
                result = self.graph.check_neighbours(argument)
                print(f"[ToolEnv] NeighbourCheck took {time.time() - dispatch_start:.3f}s")
                return result
            except Exception:
                return "There is something wrong with the arguments you send for neighbour checking. Please modify it."

        if action_type == "NodeCheck":
            try:
                result = self.graph.check_nodes(argument)
                print(f"[ToolEnv] NodeCheck took {time.time() - dispatch_start:.3f}s")
                return result
            except KeyError:
                return "The node does not exist in the graph. Please modify it."
            except Exception:
                return "There is something wrong with the arguments you send for node checking. Please modify it."

        if action_type == "EdgeCheck":
            try:
                result = self.graph.check_edges(argument)
                print(f"[ToolEnv] EdgeCheck took {time.time() - dispatch_start:.3f}s")
                return result
            except KeyError:
                return "There is no edge between the two nodes. Please modify it."
            except Exception:
                return "There is something wrong with the arguments you send for edge checking. Please modify it."

        if action_type == "SQLInterpreter":
            try:
                result = sql_interpret(argument, self.sql_conn)
                print(f"[ToolEnv] SQLInterpreter took {time.time() - dispatch_start:.3f}s")
                return result
            except Exception:
                return "There is something wrong with the SQL command you send. Please modify it."

        if action_type == "PythonInterpreter":
            try:
                result = python_interpret(argument)
                print(f"[ToolEnv] PythonInterpreter took {time.time() - dispatch_start:.3f}s")
                return result
            except Exception as exc:
                return f"An error occurred: {exc}"

        return (
            "Invalid Action. Valid Actions are "
            "Calculate[<Formula>] RetrieveAgenda[<Content>] "
            "RetrieveScirex[<Content>] LoadDB[<DBName>] "
            "FilterDB[<Condition>, <Condition>, ...] GetValue[<Column>] "
            "LoadGraph[<GraphName>] NeighbourCheck[<GraphName>, <Node>] "
            "NodeCheck[<GraphName>, <Node>] "
            "EdgeCheck[<GraphName>, <Node1>, <Node2>] "
            "SQLInterpreter[<SQLCommand>] "
            "PythonInterpreter[<PythonCode>] and Finish[<answer>]."
        )
