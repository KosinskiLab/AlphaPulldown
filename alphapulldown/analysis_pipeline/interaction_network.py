"""Utilities for plotting AlphaPulldown interaction networks."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path
import math
import re

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


_INTERFACE_PATTERN = re.compile(r"^(?P<left>[A-Z]+)_(?P<right>[A-Z]+)$")


def split_job_name(job_name: str) -> list[str]:
    """Split AlphaPulldown job names into ordered interactors."""

    if "_and_" in job_name:
        return [part for part in job_name.split("_and_") if part]
    if "+" in job_name:
        return [part for part in job_name.split("+") if part]
    return [job_name] if job_name else []


def _parse_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _chain_label_to_index(label: str) -> int:
    index = 0
    for character in label:
        index = index * 26 + (ord(character) - ord("A") + 1)
    return index - 1


def _extract_pairs_for_row(job_name: str, interface: str | None) -> list[tuple[str, str]]:
    interactors = split_job_name(job_name)
    if len(interactors) < 2:
        return []

    if isinstance(interface, str):
        match = _INTERFACE_PATTERN.fullmatch(interface.strip())
        if match is not None:
            left_index = _chain_label_to_index(match.group("left"))
            right_index = _chain_label_to_index(match.group("right"))
            if left_index < len(interactors) and right_index < len(interactors):
                return [(interactors[left_index], interactors[right_index])]

    if len(interactors) == 2:
        return [(interactors[0], interactors[1])]

    return list(combinations(interactors, 2))


def build_interaction_edge_table(
    score_table: pd.DataFrame,
    *,
    score_column: str = "iptm_ptm",
    min_score: float = 0.0,
    max_pae: float | None = None,
) -> pd.DataFrame:
    """Collapse an AlphaPulldown score table into one undirected edge table."""

    aggregated: dict[tuple[str, str], dict[str, object]] = {}
    for row in score_table.to_dict(orient="records"):
        job_name = str(row.get("jobs", "")).strip()
        if not job_name:
            continue

        score = _parse_float(row.get(score_column))
        if score is None or score < min_score:
            continue

        if max_pae is not None:
            pae = _parse_float(row.get("average_interface_pae"))
            if pae is None or pae > max_pae:
                continue

        pairs = _extract_pairs_for_row(job_name, row.get("interface"))
        for left, right in pairs:
            source, target = sorted((left, right))
            key = (source, target)
            record = aggregated.setdefault(
                key,
                {
                    "source": source,
                    "target": target,
                    "score": score,
                    "support": 0,
                    "jobs": set(),
                    "self_interaction": source == target,
                },
            )
            record["score"] = max(float(record["score"]), score)
            record["support"] = int(record["support"]) + 1
            cast_jobs = record["jobs"]
            assert isinstance(cast_jobs, set)
            cast_jobs.add(job_name)

    rows: list[dict[str, object]] = []
    for record in aggregated.values():
        jobs = sorted(record.pop("jobs"))  # type: ignore[arg-type]
        rows.append({**record, "jobs": ";".join(jobs)})

    edge_table = pd.DataFrame(rows)
    if edge_table.empty:
        return edge_table
    return edge_table.sort_values(
        by=["self_interaction", "score", "source", "target"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)


def summarise_nodes(edge_table: pd.DataFrame) -> pd.DataFrame:
    """Create a node summary table from an edge table."""

    degree_by_node: dict[str, int] = defaultdict(int)
    best_score_by_node: dict[str, float] = defaultdict(float)
    self_score_by_node: dict[str, float] = defaultdict(float)

    for row in edge_table.to_dict(orient="records"):
        source = str(row["source"])
        target = str(row["target"])
        score = float(row["score"])
        if bool(row.get("self_interaction")):
            self_score_by_node[source] = max(self_score_by_node[source], score)
            best_score_by_node[source] = max(best_score_by_node[source], score)
            continue

        degree_by_node[source] += 1
        degree_by_node[target] += 1
        best_score_by_node[source] = max(best_score_by_node[source], score)
        best_score_by_node[target] = max(best_score_by_node[target], score)

    nodes = sorted(set(degree_by_node) | set(best_score_by_node) | set(self_score_by_node))
    node_rows = [
        {
            "node": node,
            "degree": degree_by_node[node],
            "best_score": best_score_by_node[node],
            "self_interaction_score": self_score_by_node[node],
        }
        for node in nodes
    ]
    return pd.DataFrame(node_rows).sort_values(
        by=["degree", "best_score", "node"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def _connected_components(nodes: list[str], edges: pd.DataFrame) -> list[list[str]]:
    adjacency: dict[str, set[str]] = {node: set() for node in nodes}
    for row in edges.to_dict(orient="records"):
        if bool(row.get("self_interaction")):
            continue
        source = str(row["source"])
        target = str(row["target"])
        adjacency[source].add(target)
        adjacency[target].add(source)

    components: list[list[str]] = []
    seen: set[str] = set()
    for node in nodes:
        if node in seen:
            continue
        stack = [node]
        component: list[str] = []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.append(current)
            stack.extend(sorted(adjacency[current] - seen))
        components.append(sorted(component))
    return components


def _spring_layout(component_nodes: list[str], component_edges: pd.DataFrame, *, seed: int) -> dict[str, np.ndarray]:
    if len(component_nodes) == 1:
        return {component_nodes[0]: np.zeros(2, dtype=float)}

    index_by_node = {node: index for index, node in enumerate(component_nodes)}
    positions = np.random.default_rng(seed).normal(scale=0.25, size=(len(component_nodes), 2))
    weights = np.ones((len(component_nodes), len(component_nodes)), dtype=float)

    for row in component_edges.to_dict(orient="records"):
        if bool(row.get("self_interaction")):
            continue
        left = index_by_node[str(row["source"])]
        right = index_by_node[str(row["target"])]
        weights[left, right] = max(float(row["score"]), 0.05)
        weights[right, left] = weights[left, right]

    ideal_distance = math.sqrt(1.0 / max(len(component_nodes), 1))
    temperature = 0.25
    for _ in range(150):
        displacement = np.zeros_like(positions)
        delta = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        distance = np.linalg.norm(delta, axis=-1) + 1e-6

        repulsion = delta * ((ideal_distance * ideal_distance) / (distance * distance))[:, :, np.newaxis]
        displacement += np.nansum(repulsion, axis=1)

        for left in range(len(component_nodes)):
            for right in range(left + 1, len(component_nodes)):
                if weights[left, right] == 1.0 and weights[right, left] == 1.0:
                    continue
                difference = positions[left] - positions[right]
                edge_distance = np.linalg.norm(difference) + 1e-6
                attraction = difference * ((edge_distance / ideal_distance) * weights[left, right])
                displacement[left] -= attraction
                displacement[right] += attraction

        norms = np.linalg.norm(displacement, axis=1) + 1e-6
        positions += (displacement / norms[:, np.newaxis]) * np.minimum(norms, temperature)[:, np.newaxis]
        positions -= positions.mean(axis=0)
        temperature *= 0.95

    max_extent = np.max(np.linalg.norm(positions, axis=1))
    if max_extent > 0:
        positions /= max_extent
    return {node: positions[index] for node, index in index_by_node.items()}


def compute_network_layout(edge_table: pd.DataFrame, *, seed: int = 0) -> dict[str, np.ndarray]:
    """Compute a dependency-free spring layout for an interaction network."""

    if edge_table.empty:
        return {}

    nodes = sorted(set(edge_table["source"]) | set(edge_table["target"]))
    components = _connected_components(nodes, edge_table)
    component_columns = max(1, math.ceil(math.sqrt(len(components))))

    layout: dict[str, np.ndarray] = {}
    for component_index, component_nodes in enumerate(
        sorted(components, key=len, reverse=True)
    ):
        component_edges = edge_table[
            edge_table["source"].isin(component_nodes)
            & edge_table["target"].isin(component_nodes)
        ]
        component_layout = _spring_layout(
            component_nodes,
            component_edges,
            seed=seed + component_index,
        )

        row_index = component_index // component_columns
        column_index = component_index % component_columns
        offset = np.asarray([column_index * 3.0, -row_index * 3.0])
        for node, position in component_layout.items():
            layout[node] = position + offset

    return layout


def plot_interaction_network(
    edge_table: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str = "Interaction Network",
    label_top_n: int = 30,
    dpi: int = 150,
    seed: int = 0,
) -> Path:
    """Render an interaction network plot to disk."""

    if edge_table.empty:
        raise ValueError("edge_table is empty; no interactions passed the selected filters")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    layout = compute_network_layout(edge_table, seed=seed)
    node_summary = summarise_nodes(edge_table)
    component_ids = {
        node: index
        for index, component in enumerate(
            _connected_components(sorted(layout), edge_table)
        )
        for node in component
    }

    fig, ax = plt.subplots(figsize=(11, 8), dpi=dpi)
    ax.set_title(title)

    non_self_edges = edge_table[~edge_table["self_interaction"]]
    if not non_self_edges.empty:
        min_score = float(non_self_edges["score"].min())
        max_score = float(non_self_edges["score"].max())
        score_span = max(max_score - min_score, 1e-6)
        for row in non_self_edges.to_dict(orient="records"):
            source = str(row["source"])
            target = str(row["target"])
            x_values = [layout[source][0], layout[target][0]]
            y_values = [layout[source][1], layout[target][1]]
            normalized_score = (float(row["score"]) - min_score) / score_span
            ax.plot(
                x_values,
                y_values,
                color="0.55",
                alpha=0.35 + 0.45 * normalized_score,
                linewidth=1.0 + 3.0 * normalized_score,
                zorder=1,
            )

    color_map = plt.get_cmap("tab20", max(len(component_ids), 1))
    max_degree = max(int(node_summary["degree"].max()), 1)
    max_self_score = max(float(node_summary["self_interaction_score"].max()), 1.0)

    for row in node_summary.to_dict(orient="records"):
        node = str(row["node"])
        x_coord, y_coord = layout[node]
        degree = int(row["degree"])
        self_score = float(row["self_interaction_score"])
        node_size = 220 + 110 * degree + (180 if self_score > 0 else 0)
        face_color = color_map(component_ids.get(node, 0))
        ax.scatter(
            [x_coord],
            [y_coord],
            s=node_size,
            c=[face_color],
            edgecolors="black",
            linewidths=1.0,
            zorder=2,
        )
        if self_score > 0:
            ax.scatter(
                [x_coord],
                [y_coord],
                s=node_size * (1.15 + 0.35 * (self_score / max_self_score)),
                facecolors="none",
                edgecolors="black",
                linewidths=1.2,
                zorder=3,
            )

    labels_to_draw = node_summary.head(label_top_n)
    for row in labels_to_draw.to_dict(orient="records"):
        node = str(row["node"])
        x_coord, y_coord = layout[node]
        ax.text(
            x_coord,
            y_coord + 0.09,
            node,
            fontsize=9,
            ha="center",
            va="bottom",
            zorder=4,
        )

    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)
    return output_file


def write_table(table: pd.DataFrame, output_path: str | Path) -> Path:
    """Write a pandas table to CSV."""

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_file, index=False)
    return output_file
