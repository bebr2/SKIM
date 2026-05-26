"""Code execution tools for ToolQA: PythonInterpreter and SQLInterpreter."""

import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile


def python_interpret(code: str) -> str:
    """Execute Python code in a subprocess and return ans."""
    tmp_dir = tempfile.mkdtemp()
    try:
        script = os.path.join(tmp_dir, "script.py")
        with open(script, "w", encoding="utf-8") as file_obj:
            file_obj.write("ans = 0\n")
            file_obj.write(code)
            file_obj.write("\nprint(ans)\n")

        result = subprocess.run(
            [sys.executable, script],
            cwd=tmp_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())
        return result.stdout.strip()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def sql_interpret(sql_cmd: str, conn: sqlite3.Connection) -> str:
    """Execute SQL against sqlite connection and return formatted rows."""
    translated = re.sub(r"(\w+)\.(\w+_data)\b", r"\1_data", sql_cmd)

    cursor = conn.cursor()
    cursor.execute(translated)

    if cursor.description is None:
        return "Query executed successfully."

    column_names = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()

    rows_string = []
    for row in rows:
        current_row = [f"{column_names[i]}: {row[i]}" for i in range(len(row))]
        rows_string.append(", ".join(current_row))
    return "\n".join(rows_string)
