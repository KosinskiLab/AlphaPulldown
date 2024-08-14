#!/usr/bin/env python
import argparse
import csv
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import matplotlib.colors as mcolors
import tempfile
import os
from networkx.algorithms import community

def process_interactions(input_path, threshold):
    interactions = []
    with open(input_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            job = row['jobs']
            if '_' in job and float(row['iptm_ptm']) > threshold:
                items = job.split('_')
                if len(items) == 2:  # Ensure there are exactly two items to use as source and target
                    source, target = items
                    score = float(row['iptm_ptm'])
                    interactions.append([source, target, score])
    return interactions

def save_interactions(interactions, output_path):
    with open(output_path, 'w') as f:
        f.write('source,target,interaction,score\n')
        for inter in interactions:
            f.write(f'{inter[0]},{inter[1]},pp,{inter[2]}\n')

def create_plot(input_path, save_cytoscope=None):
    # Load the data
    df = pd.read_csv(input_path)

    # Create a graph
    G = nx.Graph()

    # Add edges to the graph with weights (scores)
    for index, row in df.iterrows():
        G.add_edge(row['source'], row['target'], weight=row['score'])

    # Increase repulsion by increasing the `k` parameter
    pos = nx.spring_layout(G, k=0.1, iterations=200)

    # Detect communities for better visualization of clusters
    communities = list(community.greedy_modularity_communities(G))
    community_map = {}
    for i, community_nodes in enumerate(communities):
        for node in community_nodes:
            community_map[node] = i

    # Assign colors based on communities
    community_colors = [community_map[node] for node in G.nodes()]

    edge_traces = []
    for edge in G.edges(data=True):
        x_edge = [pos[edge[0]][0], pos[edge[1]][0], None]
        y_edge = [pos[edge[0]][1], pos[edge[1]][1], None]
        color = 'gray'

        edge_trace = go.Scatter(
            x=x_edge, y=y_edge,
            line=dict(width=1, color=color),
            mode='lines',
            hoverinfo='text',
            text=[f"Source: {edge[0]}<br>Target: {edge[1]}<br>Score: {edge[2]['weight']}"],
        )
        edge_traces.append(edge_trace)

    # Extract node positions
    x_nodes = [pos[node][0] for node in G.nodes()]
    y_nodes = [pos[node][1] for node in G.nodes()]

    # Create the node trace with colors based on community
    node_trace = go.Scatter(
        x=x_nodes, y=y_nodes,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='Rainbow',  # Use a distinct colormap for communities
            size=10,  # Adjust node size as needed
            color=community_colors,  # Color nodes based on their community
            line=dict(width=1, color='black'),  # Add black edges around the nodes
            colorbar=dict(
                thickness=15,
                title='Community',
                xanchor='left',
                titleside='right'
            ),
        ),
    )

    # Add custom hover text
    node_trace.text = [f"UniProt ID: {node}<br>Connections: {nx.degree(G, node)}" for node in G.nodes()]

    # Combine all edge and node traces into one figure with larger canvas size
    fig = go.Figure(data=edge_traces + [node_trace],
                 layout=go.Layout(
                    width=1500,  # Keep canvas large for better visibility
                    height=800,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=0, l=0, r=0, t=0),
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=False, zeroline=False))
                 )

    # Show the plot
    fig.show()

    # Optionally save the plot data for Cytoscape
    if save_cytoscope:
        df.to_csv(save_cytoscope, index=False)

def main():
    parser = argparse.ArgumentParser(description="Process interaction data and create a plot.")
    parser.add_argument("input_path", help="Path to the input CSV file.")
    parser.add_argument("--save_cytoscope", help="Optional path to save Cytoscape CSV file.", default=None)
    parser.add_argument("--threshold", type=float, help="Threshold to filter out non-interacting proteins.", default=0.0)

    args = parser.parse_args()

    # Process the interactions
    interactions = process_interactions(args.input_path, args.threshold)

    # Optionally save the processed interactions to a file for Cytoscape
    if args.save_cytoscope:
        save_interactions(interactions, args.save_cytoscope)

    # Create a DataFrame from the interactions for plotting
    df = pd.DataFrame(interactions, columns=["source", "target", "score"])
    df['interaction'] = 'pp'

    # Use a temporary file for plotting
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_csv:
        df.to_csv(temp_csv.name, index=False)
        temp_csv_path = temp_csv.name

    # Create the plot and optionally save for Cytoscape
    create_plot(temp_csv_path, save_cytoscope=args.save_cytoscope)

    # Clean up the temporary file
    os.remove(temp_csv_path)

if __name__ == "__main__":
    main()
