"""Table tools for ToolQA."""

import json
import threading
from pathlib import Path

import pandas as pd

_df_cache: dict[tuple[str, str], pd.DataFrame] = {}
_df_cache_lock = threading.Lock()


def _read_db_from_disk(corpus_dir: Path, target_db: str) -> pd.DataFrame:
    if target_db == "flights":
        path = corpus_dir / "flights" / "Combined_Flights_2022.csv"
        frame = pd.read_csv(path, low_memory=False)
    elif target_db == "coffee":
        path = corpus_dir / "coffee" / "coffee_price.csv"
        frame = pd.read_csv(path, low_memory=False)
    elif target_db == "airbnb":
        path = corpus_dir / "airbnb" / "Airbnb_Open_Data.csv"
        frame = pd.read_csv(path, low_memory=False)
    elif target_db == "yelp":
        path = corpus_dir / "yelp" / "yelp_academic_dataset_business.json"
        rows = []
        with open(path, "r", encoding="utf-8") as file_obj:
            for line in file_obj:
                rows.append(json.loads(line))
        frame = pd.DataFrame(rows)
    else:
        raise ValueError(f"Unknown database: {target_db}")

    return frame.astype(str)


class TableToolkit:
    """Manages tabular database state for flights, coffee, airbnb, yelp."""

    def __init__(self, corpus_dir: Path):
        self.corpus_dir = Path(corpus_dir)
        self.data: pd.DataFrame | None = None

    def load_db(self, target_db: str) -> str:
        key = (str(self.corpus_dir), target_db)
        if key not in _df_cache:
            with _df_cache_lock:
                if key not in _df_cache:
                    _df_cache[key] = _read_db_from_disk(self.corpus_dir, target_db)

        self.data = _df_cache[key]
        column_names = ", ".join(self.data.columns.tolist())
        return (
            f"We have successfully loaded the {target_db} database, "
            f"including the following columns: {column_names}."
        )

    @staticmethod
    def _strip_quotes(value: str) -> str:
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            return value[1:-1]
        return value

    def filter_db(self, argument: str) -> str:
        backup_data = self.data
        commands = argument.split(", ")

        for cmd in commands:
            try:
                if ">=" in cmd:
                    col, val = cmd.split(">=", 1)
                    val = self._strip_quotes(val)
                    self.data = self.data[self.data[col] >= val]
                elif "<=" in cmd:
                    col, val = cmd.split("<=", 1)
                    val = self._strip_quotes(val)
                    self.data = self.data[self.data[col] <= val]
                elif ">" in cmd:
                    col, val = cmd.split(">", 1)
                    val = self._strip_quotes(val)
                    self.data = self.data[self.data[col] > val]
                elif "<" in cmd:
                    col, val = cmd.split("<", 1)
                    val = self._strip_quotes(val)
                    self.data = self.data[self.data[col] < val]
                elif "=" in cmd:
                    col, val = cmd.split("=", 1)
                    val = self._strip_quotes(val)
                    self.data = self.data[self.data[col] == val]

                if len(self.data) == 0:
                    self.data = backup_data
                    return (
                        f"The filtering query {cmd} is incorrect. "
                        "Please modify the condition."
                    )
            except Exception:
                return (
                    f"We have failed when conducting the {cmd} command. "
                    "Please make changes."
                )

        current_length = len(self.data)
        if current_length > 0:
            return f"We have successfully filtered the data ({current_length} rows)."

        rows = []
        for index in range(len(self.data)):
            outputs = []
            for attr in self.data.columns:
                outputs.append(f"{attr}: {self.data.iloc[index][attr]}")
            rows.append(", ".join(outputs))
        return "\n".join(rows)

    def get_value(self, column: str) -> str:
        if len(self.data) == 1:
            return str(self.data.iloc[0][column])
        return ", ".join(self.data[column].tolist())
