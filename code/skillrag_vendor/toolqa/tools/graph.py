"""Graph tools for ToolQA."""

import pickle
import threading
from pathlib import Path

_graph_cache: dict[tuple[str, str], dict] = {}
_graph_cache_lock = threading.Lock()


def _read_graph_from_disk(corpus_dir: Path, graph_name: str) -> dict:
    if graph_name != "dblp":
        raise ValueError(f"Unknown graph: {graph_name}")

    dblp_dir = corpus_dir / "dblp"

    with open(dblp_dir / "paper_net.pkl", "rb") as file_obj:
        paper_net = pickle.load(file_obj)  # noqa: S301

    with open(dblp_dir / "author_net.pkl", "rb") as file_obj:
        author_net = pickle.load(file_obj)  # noqa: S301

    with open(dblp_dir / "title2id_dict.pkl", "rb") as file_obj:
        title2id_dict = pickle.load(file_obj)  # noqa: S301

    with open(dblp_dir / "author2id_dict.pkl", "rb") as file_obj:
        author2id_dict = pickle.load(file_obj)  # noqa: S301

    with open(dblp_dir / "id2title_dict.pkl", "rb") as file_obj:
        id2title_dict = pickle.load(file_obj)  # noqa: S301

    with open(dblp_dir / "id2author_dict.pkl", "rb") as file_obj:
        id2author_dict = pickle.load(file_obj)  # noqa: S301

    return {
        "paper_net": paper_net,
        "author_net": author_net,
        "title2id_dict": title2id_dict,
        "author2id_dict": author2id_dict,
        "id2title_dict": id2title_dict,
        "id2author_dict": id2author_dict,
    }


class GraphToolkit:
    """Manages DBLP graph state."""

    def __init__(self, corpus_dir: Path):
        self.corpus_dir = Path(corpus_dir)
        self.paper_net = None
        self.author_net = None
        self.id2title_dict = None
        self.title2id_dict = None
        self.id2author_dict = None
        self.author2id_dict = None

    def load_graph(self, graph_name: str) -> str:
        key = (str(self.corpus_dir), graph_name)
        if key not in _graph_cache:
            with _graph_cache_lock:
                if key not in _graph_cache:
                    _graph_cache[key] = _read_graph_from_disk(self.corpus_dir, graph_name)

        cached = _graph_cache[key]
        self.paper_net = cached["paper_net"]
        self.author_net = cached["author_net"]
        self.id2title_dict = cached["id2title_dict"]
        self.title2id_dict = cached["title2id_dict"]
        self.id2author_dict = cached["id2author_dict"]
        self.author2id_dict = cached["author2id_dict"]

        return "DBLP data is loaded, including two graphs: AuthorNet and PaperNet."

    def _resolve_graph(self, graph_name: str):
        if graph_name == "PaperNet":
            return self.paper_net, self.title2id_dict, self.id2title_dict
        if graph_name == "AuthorNet":
            return self.author_net, self.author2id_dict, self.id2author_dict
        raise ValueError(f"Unknown graph name: {graph_name}")

    def check_neighbours(self, argument: str) -> str:
        graph_name, node = argument.split(", ", 1)
        graph, name2id, id2name = self._resolve_graph(graph_name)
        neighbours = [id2name[nid] for nid in graph.neighbors(name2id[node])]
        return str(neighbours)

    def check_nodes(self, argument: str) -> str:
        graph_name, node = argument.split(", ", 1)
        graph, name2id, _ = self._resolve_graph(graph_name)
        return str(graph.nodes[name2id[node]])

    def check_edges(self, argument: str) -> str:
        graph_name, node1, node2 = argument.split(", ", 2)

        if graph_name == "PaperNet":
            graph = self.paper_net
            dictionary = self.title2id_dict
            edge = graph.edges[dictionary[node1], dictionary[node2]]
            return str(edge)

        if graph_name == "AuthorNet":
            graph = self.author_net
            dictionary = self.author2id_dict
            edge = dict(graph.edges[dictionary[node1], dictionary[node2]])
            if "papers" in edge:
                edge["papers"] = [self.id2title_dict[paper_id] for paper_id in edge["papers"]]
            return str(edge)

        raise ValueError(f"Unknown graph name: {graph_name}")
