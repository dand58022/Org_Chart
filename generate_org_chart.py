"""
Generate a PDF organizational chart from MQ.xlsx using pandas, NetworkX, and Graphviz.

Requirements:
  python -m pip install pandas openpyxl networkx graphviz

Graphviz's `dot` executable must also be installed separately and available on PATH.
On Windows, install it from https://graphviz.org/download/ or with:
  winget install --id Graphviz.Graphviz --exact

The default layout uses Graphviz's left-to-right rank direction so broad peer
groups stack vertically instead of producing an extremely wide top-down chart.

Outputs:
  org_chart.dot
  org_chart.pdf
  org_chart.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

from graphviz import Digraph
import networkx as nx
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "MQ.xlsx"
DEFAULT_DOT = SCRIPT_DIR / "org_chart.dot"
DEFAULT_PDF = SCRIPT_DIR / "org_chart.pdf"
DEFAULT_PNG = SCRIPT_DIR / "org_chart.png"

REQUIRED_COLUMNS = [
    "Employee ID",
    "Job Title",
    "Position Title",
    "Name",
    "Supervisor ID",
    "Supervisor Name",
    "Email",
    "Division Name",
    "District Name",
    "Unit Name",
]

COLUMN_ALIASES = {
    "Employee ID": ["Employee ID"],
    "Job Title": ["Job Title"],
    "Position Title": ["Position Title"],
    "Name": ["Name"],
    "Supervisor ID": ["Supervisor ID", "Supervisor ID."],
    "Supervisor Name": ["Supervisor Name"],
    "Email": ["Email", "Email Address"],
    "Division Name": ["Division Name", "Division"],
    "District Name": ["District Name", "District"],
    "Unit Name": ["Unit Name", "Unit"],
}

MAX_LABEL_LINE = 34
MAX_EMAIL_LINE = 42
UNASSIGNED = "Unassigned"


def clean_value(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "nat", "<na>"}:
        return ""
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return " ".join(text.split())


def normalized_text(value: str) -> str:
    return " ".join(clean_value(value).casefold().split())


def resolve_columns(columns: list[str]) -> dict[str, str]:
    normalized = {clean_value(column).casefold(): column for column in columns}
    resolved = {}

    for target, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            source = normalized.get(alias.casefold())
            if source:
                resolved[target] = source
                break
        if target not in resolved:
            raise ValueError(f"Missing required column: {target}")

    return resolved


def load_employee_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input workbook not found: {path}")

    raw = pd.read_excel(path, dtype=str)
    resolved = resolve_columns(list(raw.columns))
    data = raw[[resolved[column] for column in REQUIRED_COLUMNS]].copy()
    data.columns = REQUIRED_COLUMNS

    for column in REQUIRED_COLUMNS:
        data[column] = data[column].map(clean_value)

    data = data[data["Employee ID"] != ""].copy()
    data = data.drop_duplicates(subset=["Employee ID"], keep="first")
    return data.reset_index(drop=True)


def meaningfully_different(first: str, second: str) -> bool:
    left = normalized_text(first)
    right = normalized_text(second)
    return bool(left and right and left != right)


def wrapped_lines(value: str, width: int) -> list[str]:
    value = clean_value(value)
    if not value:
        return []
    return textwrap.wrap(
        value,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    ) or [value]


def build_employee_label(employee: dict[str, str]) -> str:
    position_title = employee["Position Title"]
    job_title = employee["Job Title"]
    preferred_role = position_title or job_title
    alternate_role = job_title if preferred_role == position_title else position_title

    lines = []
    lines.extend(wrapped_lines(employee["Name"], MAX_LABEL_LINE))
    lines.extend(wrapped_lines(preferred_role, MAX_LABEL_LINE))
    if meaningfully_different(preferred_role, alternate_role):
        lines.extend(wrapped_lines(alternate_role, MAX_LABEL_LINE))
    lines.extend(wrapped_lines(employee["Email"], MAX_EMAIL_LINE))
    return "\n".join(line for line in lines if line)


def employee_sort_key(graph: nx.DiGraph, employee_id: str) -> tuple[str, str]:
    employee = graph.nodes[employee_id]
    return (
        normalized_text(employee.get("Name", "")),
        normalized_text(employee.get("Employee ID", "")),
    )


def build_graph(data: pd.DataFrame) -> nx.DiGraph:
    graph = nx.DiGraph()
    employees = {
        row["Employee ID"]: {column: row[column] for column in REQUIRED_COLUMNS}
        for _, row in data.iterrows()
    }

    for employee_id, employee in employees.items():
        graph.add_node(employee_id, **employee)

    employee_ids = set(employees)
    candidate_edges = set()
    for employee_id, employee in employees.items():
        supervisor_id = employee["Supervisor ID"]
        if supervisor_id and supervisor_id in employee_ids and supervisor_id != employee_id:
            candidate_edges.add((supervisor_id, employee_id))

    graph.add_edges_from(sorted(candidate_edges))
    remove_cycles(graph)
    return graph


def remove_cycles(graph: nx.DiGraph) -> None:
    while True:
        try:
            cycle = nx.find_cycle(graph, orientation="original")
        except nx.NetworkXNoCycle:
            return

        supervisor_id, employee_id, _ = cycle[-1]
        if graph.has_edge(supervisor_id, employee_id):
            graph.remove_edge(supervisor_id, employee_id)
        else:
            return


def roots_for_graph(graph: nx.DiGraph) -> list[str]:
    roots = [node for node, degree in graph.in_degree() if degree == 0]
    return sorted(
        roots,
        key=lambda employee_id: (
            normalized_text(graph.nodes[employee_id].get("Division Name", "")),
            normalized_text(graph.nodes[employee_id].get("District Name", "")),
            normalized_text(graph.nodes[employee_id].get("Unit Name", "")),
            normalized_text(graph.nodes[employee_id].get("Name", "")),
            employee_id,
        ),
    )


def filtered_graph(graph: nx.DiGraph, root_employee_id: str | None) -> nx.DiGraph:
    if not root_employee_id:
        return graph
    if root_employee_id not in graph:
        raise ValueError(f"Root Employee ID not found: {root_employee_id}")
    descendants = nx.descendants(graph, root_employee_id)
    descendants.add(root_employee_id)
    return graph.subgraph(descendants).copy()


def find_graphviz_tool(tool_name: str) -> str | None:
    tool = shutil.which(tool_name)
    if tool:
        return tool

    local_tools = SCRIPT_DIR / ".tools" / "graphviz"
    if local_tools.exists():
        for tool_path in local_tools.rglob(f"{tool_name}.exe"):
            os.environ["PATH"] = f"{tool_path.parent}{os.pathsep}{os.environ.get('PATH', '')}"
            return str(tool_path)

    common_paths = [
        Path(os.environ.get("ProgramFiles", "")) / "Graphviz" / "bin",
        Path(os.environ.get("ProgramFiles", "")) / "Graphviz" / "bin" / f"{tool_name}.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Graphviz" / "bin",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Graphviz" / "bin" / f"{tool_name}.exe",
    ]
    for path in common_paths:
        if path.name.lower() == f"{tool_name}.exe" and path.exists():
            os.environ["PATH"] = f"{path.parent}{os.pathsep}{os.environ.get('PATH', '')}"
            return str(path)
        if path.is_dir() and (path / f"{tool_name}.exe").exists():
            os.environ["PATH"] = f"{path}{os.pathsep}{os.environ.get('PATH', '')}"
            return str(path / f"{tool_name}.exe")

    return None


def add_graphviz_nodes(dot: Digraph, graph: nx.DiGraph, node_ids: list[str]) -> None:
    for employee_id in node_ids:
        dot.node(employee_id, label=build_employee_label(dict(graph.nodes[employee_id])))


def grouping_value(graph: nx.DiGraph, employee_id: str, field: str) -> str:
    return clean_value(graph.nodes[employee_id].get(field, "")) or UNASSIGNED


def cluster_id(prefix: str, parts: tuple[str, ...]) -> str:
    safe_parts = [
        "".join(char if char.isalnum() else "_" for char in part.lower()).strip("_") or "unassigned"
        for part in parts
    ]
    return f"cluster_{prefix}_{'_'.join(safe_parts)}"


def grouped_node_ids(graph: nx.DiGraph) -> dict[str, dict[str, dict[str, list[str]]]]:
    grouped: dict[str, dict[str, dict[str, list[str]]]] = {}
    for employee_id in graph.nodes:
        division = grouping_value(graph, employee_id, "Division Name")
        district = grouping_value(graph, employee_id, "District Name")
        unit = grouping_value(graph, employee_id, "Unit Name")
        grouped.setdefault(division, {}).setdefault(district, {}).setdefault(unit, []).append(employee_id)
    return grouped


def add_grouped_clusters(dot: Digraph, graph: nx.DiGraph) -> None:
    grouped = grouped_node_ids(graph)
    for division in sorted(grouped, key=normalized_text):
        with dot.subgraph(name=cluster_id("division", (division,))) as division_cluster:
            division_cluster.attr(
                label=division,
                color="#B8C8D6",
                penwidth="1.3",
                fontname="Calibri",
                fontsize="22",
                style="rounded",
                margin="18",
            )
            for district in sorted(grouped[division], key=normalized_text):
                with division_cluster.subgraph(name=cluster_id("district", (division, district))) as district_cluster:
                    district_cluster.attr(
                        label=district,
                        color="#D3DEE7",
                        penwidth="1.0",
                        fontname="Calibri",
                        fontsize="17",
                        style="rounded",
                        margin="14",
                    )
                    for unit in sorted(grouped[division][district], key=normalized_text):
                        with district_cluster.subgraph(
                            name=cluster_id("unit", (division, district, unit))
                        ) as unit_cluster:
                            unit_cluster.attr(
                                label=unit,
                                color="#E6EEF3",
                                penwidth="0.8",
                                fontname="Calibri",
                                fontsize="13",
                                style="rounded",
                                margin="10",
                            )
                            add_graphviz_nodes(
                                unit_cluster,
                                graph,
                                sorted(
                                    grouped[division][district][unit],
                                    key=lambda item: employee_sort_key(graph, item),
                                ),
                            )


def build_dot(graph: nx.DiGraph, rankdir: str = "LR") -> Digraph:
    dot = Digraph("org_chart", engine="dot")
    dot.attr(
        "graph",
        rankdir=rankdir,
        bgcolor="white",
        outputorder="edgesfirst",
        ordering="out",
        ranksep="0.42",
        nodesep="0.12",
        concentrate="false",
        splines="polyline",
        newrank="true",
        pack="true",
        packmode="array_u4",
        packmargin="8",
        remincross="true",
        compound="true",
        margin="0.20",
        page="17,11",
        pagedir="BL",
    )
    dot.attr(
        "node",
        shape="box",
        style="rounded,filled",
        fillcolor="#F2F8FC",
        color="#3C5A73",
        penwidth="1.2",
        fontname="Calibri",
        fontsize="10",
        margin="0.08,0.05",
        width="2.20",
        height="0.55",
    )
    dot.attr(
        "edge",
        color="#555555",
        penwidth="0.8",
        arrowsize="0.6",
        arrowhead="none",
    )

    add_grouped_clusters(dot, graph)

    for supervisor_id in sorted(graph.nodes, key=lambda item: employee_sort_key(graph, item)):
        children = sorted(graph.successors(supervisor_id), key=lambda item: employee_sort_key(graph, item))
        for employee_id in children:
            same_unit = (
                grouping_value(graph, supervisor_id, "Division Name"),
                grouping_value(graph, supervisor_id, "District Name"),
                grouping_value(graph, supervisor_id, "Unit Name"),
            ) == (
                grouping_value(graph, employee_id, "Division Name"),
                grouping_value(graph, employee_id, "District Name"),
                grouping_value(graph, employee_id, "Unit Name"),
            )
            dot.edge(
                supervisor_id,
                employee_id,
                weight="4" if same_unit else "1",
                constraint="true" if same_unit else "false",
            )

    return dot


def render_outputs(
    dot: Digraph,
    dot_path: Path,
    pdf_path: Path,
    png_path: Path | None,
    use_unflatten: bool = False,
) -> None:
    dot.save(filename=str(dot_path))
    dot_exe = find_graphviz_tool("dot")
    if not dot_exe:
        raise RuntimeError(
            "Graphviz executable `dot` was not found. The DOT file was saved, "
            "but PDF rendering requires installing Graphviz and adding it to PATH."
        )

    render_input = dot_path
    unflattened_path = dot_path.with_name(f"{dot_path.stem}.unflattened{dot_path.suffix}")
    unflatten_exe = find_graphviz_tool("unflatten") if use_unflatten else None
    if unflatten_exe:
        with unflattened_path.open("w", encoding="utf-8") as output:
            subprocess.run(
                [unflatten_exe, "-l", "4", "-c", "6", str(dot_path)],
                stdout=output,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
        render_input = unflattened_path

    subprocess.run(
        [dot_exe, "-Tpdf", str(render_input), "-o", str(pdf_path)],
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )

    if png_path:
        subprocess.run(
            [dot_exe, "-Tpng", str(render_input), "-o", str(png_path)],
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Graphviz PDF org chart from MQ.xlsx.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input Excel workbook.")
    parser.add_argument("--dot", type=Path, default=DEFAULT_DOT, help="Output DOT file.")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF, help="Output PDF file.")
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG, help="Optional PNG preview file.")
    parser.add_argument("--no-png", action="store_true", help="Skip PNG preview rendering.")
    parser.add_argument("--root", help="Optional root Employee ID for a focused subtree chart.")
    parser.add_argument(
        "--rankdir",
        choices=["LR", "TB"],
        default="LR",
        help="Graphviz rank direction. LR is the compact default; TB is traditional top-down.",
    )
    parser.add_argument(
        "--unflatten",
        action="store_true",
        help="Run DOT through Graphviz unflatten before rendering. Useful mainly with --rankdir TB.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = load_employee_data(args.input)
    graph = filtered_graph(build_graph(data), clean_value(args.root))
    if graph.number_of_nodes() == 0:
        raise ValueError("No employees with valid Employee ID values were found.")

    dot = build_dot(graph, rankdir=args.rankdir)
    render_outputs(dot, args.dot, args.pdf, None if args.no_png else args.png, args.unflatten)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Failed to generate org chart: {exc}", file=sys.stderr)
        raise SystemExit(1)
