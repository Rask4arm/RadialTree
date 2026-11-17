"""
Interactive Hierarchical Data Explorer
--------------------------------------
Run with:
    pip install dash pandas networkx plotly
    python hierarchical_data_explorer.py

Upload a CSV, select the columns that define the hierarchy (in order),
optionally select a classification column, and generate an interactive tree.
"""

import base64
import io
from collections import Counter
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update


EDGE_COLOR = "#17658A"
NODE_BORDER = "#003746"
PALETTE = px.colors.qualitative.Plotly


# ---------------------------------------------------------------------
# CSV and tree-building helpers
# ---------------------------------------------------------------------

def read_uploaded_csv(contents):
    """Decode a Dash upload and return a DataFrame, detecting the delimiter."""
    if not contents:
        raise ValueError("Please upload a CSV file first.")

    try:
        _, content_string = contents.split(",", 1)
        decoded = base64.b64decode(content_string)
        # utf-8-sig handles CSV files exported with a BOM.
        text = decoded.decode("utf-8-sig")
        return pd.read_csv(io.StringIO(text), sep=None, engine="python")
    except UnicodeDecodeError as exc:
        raise ValueError(
            "This file could not be read as UTF-8. Please save it as a UTF-8 CSV."
        ) from exc
    except Exception as exc:
        raise ValueError(f"Could not read this CSV: {exc}") from exc


def build_tree(df, hierarchy_columns, class_column=None, root_name="Dataset"):
    """
    Build a nested tree using the selected columns in their supplied order.

    The final hierarchy column creates leaf nodes. If a class column is
    selected, its value is stored on each leaf for coloring/comparison.
    """
    if not hierarchy_columns:
        raise ValueError("Select at least one hierarchy column.")

    missing = [c for c in hierarchy_columns if c not in df.columns]
    if class_column and class_column not in df.columns:
        missing.append(class_column)
    if missing:
        raise ValueError(f"Selected columns are not in the CSV: {missing}")

    if len(set(hierarchy_columns)) != len(hierarchy_columns):
        raise ValueError("A hierarchy column was selected more than once.")

    if class_column in hierarchy_columns:
        raise ValueError(
            "The classification column cannot also be a hierarchy column."
        )

    clean = df.dropna(subset=hierarchy_columns).copy()
    if clean.empty:
        raise ValueError("No rows contain values for all selected hierarchy columns.")

    root = {"id": "root", "name": str(root_name or "Dataset"), "children": []}
    indexes = {}
    counter = 0

    for _, row in clean.iterrows():
        parent = root
        parent_id = "root"

        for level, column in enumerate(hierarchy_columns):
            raw_value = row[column]
            value = str(raw_value).strip()

            if not value or value.lower() in {"nan", "none", "nat"}:
                break

            is_leaf_level = level == len(hierarchy_columns) - 1
            class_value = None

            if is_leaf_level and class_column:
                raw_class = row[class_column]
                if pd.notna(raw_class):
                    class_value = str(raw_class).strip()
                    if not class_value:
                        class_value = None

            # Include class in the leaf key so identical leaf names with
            # different classes remain separately represented.
            key = (parent_id, value, class_value if is_leaf_level else None)

            if key not in indexes:
                counter += 1
                node_id = f"node_{counter}"
                node = {"id": node_id, "name": value}

                if is_leaf_level:
                    node["class"] = class_value
                else:
                    node["children"] = []

                parent["children"].append(node)
                indexes[key] = node

            parent = indexes[key]
            parent_id = parent["id"]

    if not root["children"]:
        raise ValueError("The selected columns did not produce any tree nodes.")

    return root


def flatten_tree(node, parent=None, nodes=None, edges=None):
    """Convert nested tree data into node metadata and graph edges."""
    nodes = {} if nodes is None else nodes
    edges = [] if edges is None else edges

    node_id = node["id"]
    nodes[node_id] = {
        "id": node_id,
        "name": node.get("name", node_id),
        "class": node.get("class"),
        "is_original_leaf": not bool(node.get("children")),
    }

    if parent is not None:
        edges.append((parent, node_id))

    for child in node.get("children", []):
        flatten_tree(child, node_id, nodes, edges)

    return nodes, edges


def make_tree_context(tree):
    nodes, edges = flatten_tree(tree)

    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    # Preserve original parent relationships
    original_parent = {
        child: parent
        for parent, child in edges
    }

    class_order = list(dict.fromkeys(
        meta["class"]
        for meta in nodes.values()
        if meta["class"] is not None
    ))

    colors = {
        label: PALETTE[i % len(PALETTE)]
        for i, label in enumerate(class_order)
    }
    colors["other"] = "#A0A0A0"

    return nodes, edges, graph, class_order, colors, original_parent


def orient_from_root(graph, root_id, original_parent):
    """Orient tree away from root, excluding its former parent branch."""

    if root_id not in graph:
        root_id = next(iter(graph.nodes))

    children = {node_id: [] for node_id in graph.nodes}

    # Get the root's original parent
    excluded = original_parent.get(root_id)

    visited = {root_id}
    queue = [root_id]

    for current in queue:
        for neighbor in graph.neighbors(current):
            if neighbor == excluded or neighbor in visited:
                continue

            visited.add(neighbor)
            children[current].append(neighbor)
            queue.append(neighbor)

    return children


def descendant_leaf_classes(node_id, children, nodes):
    """Return class labels for classified leaves beneath a node."""
    labels = []
    stack = [node_id]

    while stack:
        current = stack.pop()
        kids = children[current]

        if not kids:
            label = nodes[current]["class"]
            if label is not None:
                labels.append(label)
        else:
            stack.extend(kids)

    return labels


def class_counts(children, nodes):
    return {
        node_id: Counter(descendant_leaf_classes(node_id, children, nodes))
        for node_id in nodes
    }


def tree_positions(root_id, children):
    """Place the root at the center and descendants on concentric rings."""
    import math

    leaves = []

    def collect_leaves(node_id):
        if not children[node_id]:
            leaves.append(node_id)
        else:
            for child in children[node_id]:
                collect_leaves(child)

    collect_leaves(root_id)

    angle = {
        leaf: -math.pi / 2 + 2 * math.pi * i / max(1, len(leaves))
        for i, leaf in enumerate(leaves)
    }

    def place_internal(node_id):
        if children[node_id]:
            for child in children[node_id]:
                place_internal(child)
            child_angles = [angle[child] for child in children[node_id]]
            sx = sum(math.cos(a) for a in child_angles)
            sy = sum(math.sin(a) for a in child_angles)
            angle[node_id] = math.atan2(sy, sx)
        return angle[node_id]

    place_internal(root_id)

    depth = {root_id: 0}
    queue = [root_id]
    for current in queue:
        for child in children[current]:
            depth[child] = depth[current] + 1
            queue.append(child)

    radius_step = 3.2
    positions = {
        node_id: (
            0.0 if depth[node_id] == 0
            else depth[node_id] * radius_step * math.cos(angle[node_id]),
            0.0 if depth[node_id] == 0
            else depth[node_id] * radius_step * math.sin(angle[node_id]),
        )
        for node_id in depth
    }
    return positions, depth


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------

def make_tree_figure(tree, root_id, selected_ids):
    import math

    nodes, _, graph, class_order, colors, original_parent = make_tree_context(tree)
    if root_id not in nodes:
        root_id = tree["id"]

    children = orient_from_root(graph, root_id, original_parent)
    counts = class_counts(children, nodes)
    positions, _ = tree_positions(root_id, children)

    fig = go.Figure()

    edge_x, edge_y = [], []
    for parent, kids in children.items():
        for child in kids:
            x0, y0 = positions[parent]
            x1, y1 = positions[child]
            r0, r1 = math.hypot(x0, y0), math.hypot(x1, y1)
            a0 = math.atan2(y0, x0) if r0 else math.atan2(y1, x1)
            a1 = math.atan2(y1, x1)
            delta = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi

            for i in range(19):
                t = i / 18
                radius = r0 + (r1 - r0) * t
                theta = a0 + delta * t
                edge_x.append(radius * math.cos(theta))
                edge_y.append(radius * math.sin(theta))
            edge_x.append(None)
            edge_y.append(None)

    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color=EDGE_COLOR, width=2.2),
        hoverinfo="skip", showlegend=False,
    ))

    click_x, click_y, click_custom, click_text = [], [], [], []
    label_x, label_y, label_text = [], [], []

    for node_id, meta in nodes.items():
        if node_id not in positions:
            continue

        x, y = positions[node_id]
        counter = counts[node_id]
        total = sum(counter.values())
        is_leaf = not children[node_id]
        width = 1.45 if is_leaf else 1.8
        height = 0.58
        left, bottom = x - width / 2, y - height / 2

        active_classes = [c for c in class_order if counter.get(c, 0) > 0]
        active_classes += [
            c for c in counter if c not in colors and counter[c] > 0
        ]

        if total:
            cursor = left
            for class_name in active_classes:
                value = counter[class_name]
                segment_width = width * value / total
                fig.add_shape(
                    type="rect",
                    x0=cursor, x1=cursor + segment_width,
                    y0=bottom, y1=bottom + height,
                    fillcolor=colors.get(class_name, colors["other"]),
                    line=dict(color=NODE_BORDER, width=1.2),
                    layer="above",
                )
                cursor += segment_width
        else:
            fig.add_shape(
                type="rect",
                x0=left, x1=left + width,
                y0=bottom, y1=bottom + height,
                fillcolor="#D9E0E4",
                line=dict(color=NODE_BORDER, width=1.2),
                layer="above",
            )

        if node_id in selected_ids or node_id == root_id:
            fig.add_shape(
                type="rect",
                x0=left - 0.06, x1=left + width + 0.06,
                y0=bottom - 0.06, y1=bottom + height + 0.06,
                fillcolor="rgba(0,0,0,0)",
                line=dict(
                    color="#111111" if node_id in selected_ids else "#17658A",
                    width=3 if node_id in selected_ids else 2,
                ),
                layer="above",
            )

        label_x.append(x)
        label_y.append(y)
        label_text.append(meta["name"])
        click_x.append(x)
        click_y.append(y)
        click_custom.append(node_id)

        class_summary = ", ".join(
            f"{label}: {counter[label]}"
            for label in class_order if counter.get(label, 0)
        ) or "no class labels"

        click_text.append(f"{meta['name']}<br>Class counts: {class_summary}")

    fig.add_trace(go.Scatter(
        x=click_x, y=click_y, mode="markers",
        marker=dict(size=42, color="rgba(0,0,0,0.01)", line=dict(width=0)),
        customdata=click_custom, text=click_text,
        hovertemplate="%{text}<extra></extra>", showlegend=False,
    ))

    fig.add_trace(go.Scatter(
        x=label_x, y=label_y, mode="text",
        text=label_text, textposition="middle center",
        textfont=dict(size=10, color="#111111"),
        hoverinfo="skip", showlegend=False,
    ))

    max_radius = max(
        (math.hypot(x, y) for x, y in positions.values()), default=1
    )
    extent = max_radius + 2.5

    fig.update_layout(
        margin=dict(l=25, r=25, t=55, b=25),
        paper_bgcolor="white", plot_bgcolor="white",
        title=f"Radial tree — current root: {nodes[root_id]['name']}",
        clickmode="event+select",
        xaxis=dict(visible=False, fixedrange=False, range=[-extent, extent]),
        yaxis=dict(
            visible=False, fixedrange=False, range=[-extent, extent],
            scaleanchor="x", scaleratio=1,
        ),
        showlegend=False, height=820, dragmode="pan",
    )
    return fig


def make_comparison_figure(tree, selected_ids, root_id):
    nodes, _, graph, class_order, colors, original_parent = make_tree_context(tree)
    children = orient_from_root(graph, root_id, original_parent)
    counts = class_counts(children, nodes)
    fig = go.Figure()

    if not selected_ids:
        fig.add_annotation(
            text="Click two nodes in the tree to compare their class bars.",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(size=14),
        )
    else:
        for row, node_id in enumerate(selected_ids[:2]):
            counter = counts[node_id]
            total = sum(counter.values())

            if total == 0:
                fig.add_annotation(
                    text=f"{nodes[node_id]['name']}: no class-labelled leaves",
                    x=0.5, y=1 - row * 0.5,
                    xref="paper", yref="paper", showarrow=False,
                )
                continue

            for class_name in class_order:
                value = counter.get(class_name, 0)
                if value <= 0:
                    continue

                fig.add_trace(go.Bar(
                    x=[value], y=[nodes[node_id]["name"]],
                    orientation="h", name=class_name,
                    marker_color=colors.get(class_name, colors["other"]),
                    text=[str(value)], textposition="inside",
                    hovertemplate=(
                        f"{class_name}: %{{x}} (%{{customdata:.1%}})"
                        "<extra></extra>"
                    ),
                    customdata=[value / total],
                    showlegend=(row == 0),
                ))

    fig.update_layout(
        barmode="stack", title="Selected node comparison",
        margin=dict(l=20, r=15, t=75, b=35), height=260,
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis_title="Number of descendant leaves",
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


# ---------------------------------------------------------------------
# Dash application
# ---------------------------------------------------------------------

app = Dash(__name__)
app.title = "Interactive Hierarchical Data Explorer"

app.layout = html.Div(
    style={
        "fontFamily": "Arial, sans-serif",
        "padding": "16px",
        "background": "#f7f9fa",
    },
    children=[
        html.H2("Interactive Hierarchical Data Explorer"),
        html.P(
            "Upload a CSV, choose the columns that define its hierarchy, "
            "and optionally choose a column to classify the leaf nodes."
        ),

        dcc.Upload(
            id="upload-csv",
            children=html.Div([
                "Drag and drop a CSV here, or ",
                html.A("select a file"),
            ]),
            style={
                "width": "100%", "height": "65px", "lineHeight": "65px",
                "borderWidth": "1px", "borderStyle": "dashed",
                "borderRadius": "8px", "textAlign": "center",
                "background": "white", "marginBottom": "12px",
            },
            multiple=False,
        ),
        html.Div(id="upload-status", style={"marginBottom": "12px"}),

        dcc.Store(id="data-store"),
        dcc.Store(id="tree-store"),
        dcc.Store(id="current-root", data="root"),
        dcc.Store(id="selected-nodes", data=[]),

        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "minmax(260px, 1fr) minmax(600px, 2fr)",
                "gap": "16px",
                "alignItems": "start",
            },
            children=[
                html.Div(
                    style={
                        "background": "white", "padding": "16px",
                        "borderRadius": "8px",
                    },
                    children=[
                        html.H3("Dataset configuration"),
                        html.Label("Hierarchy columns (select in top-to-bottom order)"),
                        dcc.Dropdown(
                            id="hierarchy-columns",
                            options=[],
                            value=[],
                            multi=True,
                            placeholder="Select hierarchy columns in order",
                        ),
                        html.Br(),
                        html.Label("Classification column (optional)"),
                        dcc.Dropdown(
                            id="class-column",
                            options=[],
                            value=None,
                            clearable=True,
                            placeholder="Choose a column for leaf colors",
                        ),
                        html.Br(),
                        html.Label("Root node name"),
                        dcc.Input(
                            id="root-name", type="text", value="Dataset",
                            style={"width": "100%", "padding": "8px"},
                        ),
                        html.Br(), html.Br(),
                        html.Button(
                            "Generate tree", id="generate-tree", n_clicks=0,
                            style={"padding": "10px 14px"},
                        ),
                        html.Div(
                            id="generation-status",
                            style={"marginTop": "12px", "whiteSpace": "pre-wrap"},
                        ),
                        html.Hr(),
                        html.H3("Tree controls"),
                        html.Button(
                            "Make selected node root", id="make-root",
                            n_clicks=0, style={"padding": "9px", "marginRight": "6px"},
                        ),
                        html.Button(
                            "Original root", id="reset-root", n_clicks=0,
                            style={"padding": "9px"},
                        ),
                        html.Br(), html.Br(),
                        html.Button(
                            "Clear selection", id="clear-selection", n_clicks=0,
                            style={"padding": "9px"},
                        ),
                        html.Div(
                            id="selection-status",
                            style={"marginTop": "12px", "fontWeight": "bold"},
                        ),
                        dcc.Graph(
                            id="comparison-graph",
                            config={"displaylogo": False},
                        ),
                        html.Div(
                            id="class-legend",
                            style={"fontWeight": "bold", "marginTop": "10px"},
                        ),
                    ],
                ),
                html.Div(
                    style={
                        "background": "white", "padding": "10px",
                        "borderRadius": "8px",
                    },
                    children=[
                        dcc.Graph(
                            id="tree-graph",
                            config={
                                "scrollZoom": True,
                                "displaylogo": False,
                                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                            },
                        ),
                    ],
                ),
            ],
        ),
        html.P(
            "Tip: scroll to zoom and drag to pan. Click up to two nodes to compare them.",
            style={"color": "#555", "fontSize": "13px", "marginTop": "12px"},
        ),
    ],
)


@app.callback(
    Output("data-store", "data"),
    Output("hierarchy-columns", "options"),
    Output("hierarchy-columns", "value"),
    Output("class-column", "options"),
    Output("upload-status", "children"),
    Input("upload-csv", "contents"),
    State("upload-csv", "filename"),
    prevent_initial_call=True,
)
def load_csv(contents, filename):
    try:
        df = read_uploaded_csv(contents)
        if len(df.columns) < 1:
            raise ValueError("The CSV does not contain any columns.")

        columns = [str(c) for c in df.columns]
        records = df.to_json(orient="records", date_format="iso")

        options = [{"label": c, "value": c} for c in columns]
        # Initially select all columns; the user can remove the class
        # column from the hierarchy and choose it separately.
        default_hierarchy = columns[:-1]

        return (
            {"records": records, "columns": columns, "filename": filename},
            options,
            default_hierarchy,
            options,
            f"Loaded {filename or 'CSV'}: {len(df):,} rows and {len(columns)} columns.",
        )
    except Exception as exc:
        return None, [], [], [], f"Upload error: {exc}"


@app.callback(
    Output("tree-store", "data"),
    Output("current-root", "data", allow_duplicate=True),
    Output("selected-nodes", "data", allow_duplicate=True),
    Output("generation-status", "children"),
    Input("generate-tree", "n_clicks"),
    State("data-store", "data"),
    State("hierarchy-columns", "value"),
    State("class-column", "value"),
    State("root-name", "value"),
    prevent_initial_call=True,
)
def generate_tree(_clicks, stored_data, hierarchy_columns, class_column, root_name):
    if not stored_data:
        return no_update, no_update, no_update, "Upload a CSV first."

    try:
        df = pd.read_json(io.StringIO(stored_data["records"]), orient="records")
        tree = build_tree(
            df,
            hierarchy_columns or [],
            class_column=class_column,
            root_name=root_name or "Dataset",
        )
        nodes, _, _, class_order, _, _ = make_tree_context(tree)

        status = (
            f"Tree generated successfully: {len(nodes):,} nodes, "
            f"{len(class_order)} classification categories."
        )
        return tree, tree["id"], [], status
    except Exception as exc:
        return no_update, no_update, no_update, f"Could not generate tree: {exc}"


@app.callback(
    Output("selected-nodes", "data"),
    Input("tree-graph", "clickData"),
    Input("clear-selection", "n_clicks"),
    State("selected-nodes", "data"),
    State("tree-store", "data"),
    prevent_initial_call=True,
)
def update_selection(click_data, _clear_clicks, selected, tree):
    if ctx.triggered_id == "clear-selection":
        return []

    if ctx.triggered_id == "tree-graph" and click_data and tree:
        node_id = click_data["points"][0].get("customdata")
        nodes, *_ = make_tree_context(tree)

        if node_id not in nodes:
            return selected or []

        selected = list(selected or [])
        if node_id in selected:
            selected.remove(node_id)
        else:
            selected.append(node_id)

        return selected[-2:]

    return selected or []


@app.callback(
    Output("current-root", "data"),
    Input("make-root", "n_clicks"),
    Input("reset-root", "n_clicks"),
    State("selected-nodes", "data"),
    State("current-root", "data"),
    State("tree-store", "data"),
    prevent_initial_call=True,
)
def update_root(_make_clicks, _reset_clicks, selected, current_root, tree):
    if not tree:
        return no_update

    if ctx.triggered_id == "reset-root":
        return tree["id"]

    if ctx.triggered_id == "make-root" and selected:
        return selected[-1]

    return current_root


@app.callback(
    Output("tree-graph", "figure"),
    Input("tree-store", "data"),
    Input("current-root", "data"),
    Input("selected-nodes", "data"),
)
def update_tree_figure(tree, root_id, selected):
    if not tree:
        fig = go.Figure()
        fig.add_annotation(
            text="Upload a CSV and generate a tree to begin.",
            x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
        )
        fig.update_layout(height=820, paper_bgcolor="white")
        return fig

    return make_tree_figure(tree, root_id or tree["id"], selected or [])


@app.callback(
    Output("comparison-graph", "figure"),
    Input("tree-store", "data"),
    Input("selected-nodes", "data"),
    Input("current-root", "data"),
)
def update_comparison(tree, selected, root_id):
    if not tree:
        return go.Figure()
    return make_comparison_figure(tree, selected or [], root_id or tree["id"])


@app.callback(
    Output("selection-status", "children"),
    Input("selected-nodes", "data"),
    Input("tree-store", "data"),
)
def update_status(selected, tree):
    if not tree or not selected:
        return "Selected: none — click up to two nodes to compare."

    nodes, *_ = make_tree_context(tree)
    names = [nodes[node_id]["name"] for node_id in selected if node_id in nodes]

    if len(names) == 1:
        return f"Selected: {names[0]} — select one more node."
    if len(names) >= 2:
        return f"Comparing: {names[0]} and {names[1]}"
    return "Selected: none."


@app.callback(
    Output("class-legend", "children"),
    Input("tree-store", "data"),
)
def update_legend(tree):
    if not tree:
        return "Classification colors will appear here."

    _, _, _, class_order, colors, _ = make_tree_context(tree)
    if not class_order:
        return "No classification column selected."

    return html.Div([
        html.Span("Class colors: "),
        *[
            html.Span(
                f"■ {label}  ",
                style={"color": colors[label], "marginRight": "10px"},
            )
            for label in class_order
        ],
    ])


if __name__ == "__main__":
    app.run(debug=True)

