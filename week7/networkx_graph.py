import json
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations



def load_entities(filepath="cleaned_entities.json"):
    """
    Load cleaned & deduplicated entities from Week 6 output.
    Expected format:
    [
      {
        "doc_id": "arxiv_001",
        "entities": [
          {"text": "BERT", "label": "METHOD"},
          {"text": "NLP", "label": "TASK"},
          ...
        ]
      },
      ...
    ]
    """
    with open(filepath, "r") as f:
        return json.load(f)

def build_knowledge_graph(entity_data):
    """
    Construct a NetworkX graph where:
      - Nodes  = unique entities (text + label)
      - Edges  = co-occurrence within the same document
      - Weight = number of documents the pair co-occurs in
    """
    G = nx.Graph()

    cooccurrence = Counter()

    for doc in entity_data:
        doc_id = doc["doc_id"]
        entities = doc["entities"]

        for ent in entities:
            node_id = ent["text"].lower()          
            if not G.has_node(node_id):
                G.add_node(node_id,
                           label=ent["label"],
                           display=ent["text"],
                           frequency=0)
            G.nodes[node_id]["frequency"] += 1    
        
        unique_texts = list({e["text"].lower() for e in entities})
        for a, b in combinations(unique_texts, 2):
            cooccurrence[(a, b)] += 1

    for (a, b), weight in cooccurrence.items():
        if G.has_node(a) and G.has_node(b):
            G.add_edge(a, b, weight=weight, relation="co-occurrence")

    return G








def compute_metrics(G):
    """
    Compute structural metrics aligned with the evaluation strategy (§5.2).
    Returns a dict of metric values.
    """
    metrics = {}

   
    metrics["num_nodes"] = G.number_of_nodes()
    metrics["num_edges"] = G.number_of_edges()

    
    metrics["density"] = round(nx.density(G), 6)

    
    degrees = [d for _, d in G.degree()]
    metrics["avg_degree"]    = round(sum(degrees) / len(degrees), 4) if degrees else 0
    metrics["max_degree"]    = max(degrees) if degrees else 0
    metrics["min_degree"]    = min(degrees) if degrees else 0
    metrics["avg_clustering_coefficient"] = round(nx.average_clustering(G), 6)


    components = list(nx.connected_components(G))
    metrics["num_connected_components"] = len(components)
    metrics["largest_component_size"]   = max(len(c) for c in components)


    largest_cc = G.subgraph(max(components, key=len)).copy()
    metrics["giant_component_density"]  = round(nx.density(largest_cc), 6)
    metrics["giant_component_diameter"] = nx.diameter(largest_cc) if nx.is_connected(largest_cc) else None

    degree_centrality = nx.degree_centrality(G)
    top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    metrics["top_nodes_by_degree_centrality"] = top_nodes

    betweenness = nx.betweenness_centrality(G, normalized=True)
    top_between = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
    metrics["top_nodes_by_betweenness"] = top_between

    return metrics


def save_graph(G, path="knowledge_graph.pkl"):
    """Persist graph object for downstream use (Week 8+)."""
    with open(path, "wb") as f:
        pickle.dump(G, f)
    print(f"Graph saved → {path}")


def save_metrics_report(metrics, path="graph_metrics_report.json"):
    """Save metrics as JSON for the week's deliverable."""
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics report saved → {path}")


def print_metrics_report(metrics):
    """Pretty-print metrics to console."""
    print("\n" + "="*50)
    print("       KNOWLEDGE GRAPH — METRICS REPORT")
    print("="*50)
    print(f"  Nodes                     : {metrics['num_nodes']}")
    print(f"  Edges                     : {metrics['num_edges']}")
    print(f"  Graph Density             : {metrics['density']}")
    print(f"  Average Degree            : {metrics['avg_degree']}")
    print(f"  Max Degree                : {metrics['max_degree']}")
    print(f"  Avg Clustering Coefficient: {metrics['avg_clustering_coefficient']}")
    print(f"  Connected Components      : {metrics['num_connected_components']}")
    print(f"  Largest Component Size    : {metrics['largest_component_size']}")
    print(f"  Giant Component Diameter  : {metrics['giant_component_diameter']}")
    print("\n  Top 10 Nodes (Degree Centrality):")
    for node, score in metrics["top_nodes_by_degree_centrality"]:
        print(f"    {node:<30} {score:.4f}")
    print("\n  Top 10 Nodes (Betweenness Centrality):")
    for node, score in metrics["top_nodes_by_betweenness"]:
        print(f"    {node:<30} {score:.4f}")
    print("="*50 + "\n")

def visualize_graph(G, top_n=50, output_path="graph_preview.png"):
    """
    Draw the top-N nodes by degree for a quick sanity-check plot.
    Full interactive visualisation is handled in Week 10 (PyVis/D3.js).
    """
    top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:top_n]
    subgraph = G.subgraph([n for n, _ in top_nodes])

    label_colors = {
        "METHOD":      "#4C8BE2",
        "TASK":        "#E2844C",
        "DATASET":     "#6DBE6D",
        "INSTITUTION": "#C46DE2",
        "TECHNIQUE":   "#E2C44C",
    }
    node_colors = [
        label_colors.get(subgraph.nodes[n].get("label", ""), "#AAAAAA")
        for n in subgraph.nodes()
    ]

    edge_weights = [subgraph[u][v].get("weight", 1) for u, v in subgraph.edges()]

    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(subgraph, seed=42, k=0.5)
    nx.draw_networkx(
        subgraph, pos,
        node_color=node_colors,
        node_size=500,
        font_size=8,
        width=[0.5 + w * 0.5 for w in edge_weights],
        alpha=0.85,
        edge_color="#CCCCCC"
    )
    plt.title(f"Knowledge Graph Preview (top {top_n} nodes by degree)", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Graph preview saved → {output_path}")















if __name__ == "__main__":
    import os

    if os.path.exists("cleaned_entities.json"):
        print("Loading entity data from cleaned_entities.json ...")
        entity_data = load_entities("cleaned_entities.json")
    else:
        print("cleaned_entities.json not found")
    
    print("Building knowledge graph ...")
    G = build_knowledge_graph(entity_data)

    print("Computing graph metrics ...")
    metrics = compute_metrics(G)
    print_metrics_report(metrics)

    save_graph(G, "knowledge_graph.pkl")
    save_metrics_report(metrics, "graph_metrics_report.json")
    visualize_graph(G, top_n=40, output_path="graph_preview.png")

    print("Week 7 complete ✓")