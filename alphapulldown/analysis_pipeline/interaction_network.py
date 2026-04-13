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
from matplotlib.collections import LineCollection


_INTERFACE_PATTERN = re.compile(r"^(?P<left>[A-Z]+)_(?P<right>[A-Z]+)$")
_HOMO_OLIGOMER_PATTERN = re.compile(r"^(?P<name>.+)_homo_(?P<count>\d+)er$")


def _expand_ap_style_homo_oligomer(token: str) -> list[str]:
    match = _HOMO_OLIGOMER_PATTERN.fullmatch(token)
    if match is None:
        return [token] if token else []
    return [match.group("name")] * int(match.group("count"))


def split_job_name(job_name: str) -> list[str]:
    """Split AlphaPulldown job names into ordered interactors."""

    if "_and_" in job_name:
        parts = [part for part in job_name.split("_and_") if part]
    elif "+" in job_name:
        parts = [part for part in job_name.split("+") if part]
    else:
        parts = [job_name] if job_name else []

    expanded_parts: list[str] = []
    for part in parts:
        expanded_parts.extend(_expand_ap_style_homo_oligomer(part))
    return expanded_parts


def _parse_float(value) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(parsed) else parsed


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
    if len(component_nodes) == 2:
        return {
            component_nodes[0]: np.asarray([-0.45, 0.0], dtype=float),
            component_nodes[1]: np.asarray([0.45, 0.0], dtype=float),
        }

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

    def component_sort_key(component_nodes: list[str]) -> tuple[int, float, str]:
        component_edges = edge_table[
            edge_table["source"].isin(component_nodes)
            & edge_table["target"].isin(component_nodes)
        ]
        max_score = 0.0
        if not component_edges.empty:
            max_score = float(component_edges["score"].max())
        return (-len(component_nodes), -max_score, component_nodes[0])

    layout: dict[str, np.ndarray] = {}
    for component_index, component_nodes in enumerate(
        sorted(components, key=component_sort_key)
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
    score_label: str = "score",
) -> Path:
    """Render an interaction network plot to disk."""

    if edge_table.empty:
        raise ValueError("edge_table is empty; no interactions passed the selected filters")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    layout = compute_network_layout(edge_table, seed=seed)
    node_summary = summarise_nodes(edge_table)
    components = _connected_components(sorted(layout), edge_table)
    component_count = max(len(components), 1)

    fig_width = min(18.0, max(10.0, 6.0 + 1.8 * math.sqrt(component_count)))
    fig_height = min(13.0, max(7.0, fig_width * 0.72))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    ax.set_title(title)

    non_self_edges = edge_table[~edge_table["self_interaction"]]
    if not non_self_edges.empty:
        min_score = float(non_self_edges["score"].min())
        max_score = float(non_self_edges["score"].max())
        score_span = max(max_score - min_score, 1e-6)
        norm_min = min_score
        norm_max = max_score
        if math.isclose(norm_min, norm_max):
            norm_min = max(0.0, min_score - 0.01)
            norm_max = max_score + 0.01

        segments = []
        edge_scores = []
        linewidths = []
        for row in non_self_edges.to_dict(orient="records"):
            source = str(row["source"])
            target = str(row["target"])
            score = float(row["score"])
            normalized_score = (score - min_score) / score_span
            segments.append(
                [
                    (layout[source][0], layout[source][1]),
                    (layout[target][0], layout[target][1]),
                ]
            )
            edge_scores.append(score)
            linewidths.append(0.7 + 3.1 * normalized_score)

        edge_collection = LineCollection(
            segments,
            cmap="viridis",
            norm=plt.Normalize(norm_min, norm_max),
            linewidths=linewidths,
            alpha=0.72,
            zorder=1,
        )
        edge_collection.set_array(np.asarray(edge_scores))
        ax.add_collection(edge_collection)
        colorbar = fig.colorbar(edge_collection, ax=ax, fraction=0.035, pad=0.01)
        colorbar.set_label(score_label)

    max_self_score = max(float(node_summary["self_interaction_score"].max()), 1.0)
    node_count = len(node_summary)
    base_node_size = 120 if node_count <= 40 else 90 if node_count <= 120 else 60

    for row in node_summary.to_dict(orient="records"):
        node = str(row["node"])
        x_coord, y_coord = layout[node]
        degree = int(row["degree"])
        self_score = float(row["self_interaction_score"])
        node_size = base_node_size + 55 * math.sqrt(max(degree, 0))
        if self_score > 0:
            node_size += 55
        ax.scatter(
            [x_coord],
            [y_coord],
            s=node_size,
            c=["#f8fbff"],
            edgecolors="#263648",
            linewidths=0.9,
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

    if label_top_n > 0:
        labels_to_draw = node_summary.head(label_top_n)
        label_font_size = 8 if len(labels_to_draw) <= 60 else 7
        for row in labels_to_draw.to_dict(orient="records"):
            node = str(row["node"])
            x_coord, y_coord = layout[node]
            ax.text(
                x_coord,
                y_coord + 0.12,
                node,
                fontsize=label_font_size,
                ha="center",
                va="bottom",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                    "pad": 0.6,
                },
                zorder=4,
            )

    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.08)
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
