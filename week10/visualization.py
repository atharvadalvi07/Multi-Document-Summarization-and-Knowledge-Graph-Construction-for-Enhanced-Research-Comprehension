"""
Week 10: Knowledge Graph Visualization Prototype
-------------------------------------------------
Project: Multi-Document Summarization and Knowledge Graph Construction
Author : Atharva Dalvi, Texas Tech University

Deliverable: Interactive HTML visualization of the NetworkX knowledge graph
             built in Week 7 and refined in Weeks 8–9.

Pipeline:
    1. Load the serialized graph from Week 7/8/9 (falls back to mock data)
    2. Compute community assignments (Louvain, from Week 9)
    3. Map communities → colors for visual grouping
    4. Render with PyVis → self-contained interactive HTML file
    5. Save output HTML + a JSON metadata summary

Dependencies:
    pip install pyvis networkx python-louvain
"""

import os
import json
import pickle
import random
import logging
from pathlib import Path
from collections import defaultdict

import networkx as nx

# ── Optional community detection (Week 9 output) ──────────────────────────────
try:
    import community as community_louvain  # python-louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False
    logging.warning("python-louvain not found. Falling back to NetworkX greedy modularity.")

# ── PyVis ──────────────────────────────────────────────────────────────────────
try:
    from pyvis.network import Network
except ImportError:
    raise ImportError("PyVis is required: pip install pyvis")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

GRAPH_INPUT_PATH   = "knowledge_graph.pkl"          # Week 7/8/9 serialized graph
HTML_OUTPUT_PATH   = "knowledge_graph.html"  # interactive visualization
META_OUTPUT_PATH   = "visualization_meta.json"

VIS_CONFIG = {
    "height": "800px",
    "width":  "100%",
    "bgcolor": "#1a1a2e",          # dark background
    "font_color": "#e0e0e0",
    "min_node_size":  10,
    "max_node_size":  50,
    "edge_width_min": 0.5,
    "edge_width_max": 6.0,
    "top_k_nodes":    80,          # cap nodes for readability (0 = no cap)
    "physics_solver": "forceAtlas2Based",
}

# ══════════════════════════════════════════════════════════════════════════════
# COMMUNITY PALETTE
# ══════════════════════════════════════════════════════════════════════════════

COMMUNITY_PALETTE = [
    "#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261",
    "#9b5de5", "#00b4d8", "#f72585", "#80b918", "#ff6b6b",
    "#06d6a0", "#ffd166", "#118ab2", "#ef476f", "#26547c",
]


def get_community_color(community_id: int) -> str:
    return COMMUNITY_PALETTE[community_id % len(COMMUNITY_PALETTE)]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 – LOAD GRAPH
# ══════════════════════════════════════════════════════════════════════════════

def load_graph(path: str) -> nx.Graph:
    """Load serialized graph or fall back to a realistic mock graph."""
    if os.path.exists(path):
        logger.info(f"Loading graph from {path}")
        with open(path, "rb") as f:
            G = pickle.load(f)
        if not isinstance(G, (nx.Graph, nx.DiGraph)):
            raise ValueError("Loaded object is not a NetworkX graph.")
        logger.info(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    else:
        logger.warning(f"{path} not found — generating mock graph for demonstration.")
        return _build_mock_graph()


def _build_mock_graph() -> nx.Graph:
    """
    Realistic mock knowledge graph mimicking NLP/ML research entities.
    Nodes carry 'entity_type' and 'frequency'; edges carry 'weight'.
    """
    G = nx.Graph()

    entities = {
        # (label, entity_type, frequency)
        "BERT":             ("Model",       18),
        "GPT-4":            ("Model",       15),
        "T5":               ("Model",       12),
        "Transformer":      ("Technique",   20),
        "Attention":        ("Technique",   17),
        "SciSpacy":         ("Tool",        10),
        "spaCy":            ("Tool",        14),
        "NER":              ("Task",        16),
        "Summarization":    ("Task",        13),
        "Knowledge Graph":  ("Task",        11),
        "Named Entity":     ("Concept",     14),
        "PubMed":           ("Dataset",     9),
        "arXiv":            ("Dataset",     8),
        "ROUGE":            ("Metric",      12),
        "BLEU":             ("Metric",      10),
        "NetworkX":         ("Tool",        9),
        "PyVis":            ("Tool",        7),
        "Louvain":          ("Technique",   8),
        "Community Detection": ("Task",     7),
        "Co-occurrence":    ("Technique",   9),
        "Dependency Parse": ("Technique",   8),
        "Entity Linking":   ("Task",        6),
        "Coreference":      ("Technique",   7),
        "BioNLP":           ("Domain",      6),
        "Deep Learning":    ("Technique",   15),
        "Fine-tuning":      ("Technique",   11),
        "Zero-shot":        ("Technique",   9),
        "Embeddings":       ("Concept",     14),
        "Graph Density":    ("Metric",      5),
        "Clustering Coeff": ("Metric",      5),
    }

    for node, (etype, freq) in entities.items():
        G.add_node(node, entity_type=etype, frequency=freq)

    edges = [
        ("BERT",          "Transformer",       9),
        ("GPT-4",         "Transformer",       8),
        ("T5",            "Transformer",       7),
        ("Transformer",   "Attention",         10),
        ("BERT",          "Fine-tuning",       8),
        ("BERT",          "Embeddings",        7),
        ("GPT-4",         "Zero-shot",         6),
        ("SciSpacy",      "NER",               9),
        ("spaCy",         "NER",               10),
        ("NER",           "Named Entity",      8),
        ("Named Entity",  "Entity Linking",    6),
        ("Named Entity",  "Coreference",       5),
        ("Summarization", "ROUGE",             7),
        ("Summarization", "BLEU",              6),
        ("Summarization", "BERT",              5),
        ("Knowledge Graph","NetworkX",         8),
        ("Knowledge Graph","Co-occurrence",    7),
        ("Knowledge Graph","Dependency Parse", 6),
        ("NetworkX",      "PyVis",             5),
        ("NetworkX",      "Louvain",           6),
        ("Louvain",       "Community Detection",8),
        ("Community Detection","Graph Density",4),
        ("Community Detection","Clustering Coeff",4),
        ("PubMed",        "BioNLP",            7),
        ("PubMed",        "SciSpacy",          6),
        ("arXiv",         "Summarization",     5),
        ("arXiv",         "Knowledge Graph",   4),
        ("Deep Learning", "BERT",              8),
        ("Deep Learning", "Transformer",       9),
        ("Deep Learning", "Embeddings",        7),
        ("Co-occurrence", "Dependency Parse",  5),
        ("spaCy",         "SciSpacy",          6),
    ]

    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    logger.info(f"Mock graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 – PRUNE TO TOP-K NODES (OPTIONAL)
# ══════════════════════════════════════════════════════════════════════════════

def prune_graph(G: nx.Graph, top_k: int) -> nx.Graph:
    """Keep only the top-k nodes by degree to keep the visualization readable."""
    if top_k <= 0 or G.number_of_nodes() <= top_k:
        return G
    degrees = dict(G.degree())
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:top_k]
    sub = G.subgraph(top_nodes).copy()
    logger.info(f"Pruned to top-{top_k} nodes: {sub.number_of_nodes()} nodes, {sub.number_of_edges()} edges")
    return sub


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 – COMMUNITY DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_communities(G: nx.Graph) -> dict:
    """Return {node: community_id} mapping."""
    # Work on undirected copy
    UG = G.to_undirected() if G.is_directed() else G

    if LOUVAIN_AVAILABLE:
        partition = community_louvain.best_partition(UG)
        logger.info(f"Louvain communities: {len(set(partition.values()))}")
    else:
        # NetworkX greedy modularity fallback
        communities_gen = nx.algorithms.community.greedy_modularity_communities(UG)
        partition = {}
        for cid, community_set in enumerate(communities_gen):
            for node in community_set:
                partition[node] = cid
        logger.info(f"Greedy modularity communities: {len(set(partition.values()))}")

    return partition


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 – BUILD PYVIS NETWORK
# ══════════════════════════════════════════════════════════════════════════════

def _scale(value: float, min_val: float, max_val: float,
           out_min: float, out_max: float) -> float:
    if max_val == min_val:
        return (out_min + out_max) / 2
    return out_min + (value - min_val) / (max_val - min_val) * (out_max - out_min)


def build_pyvis_network(G: nx.Graph, partition: dict, cfg: dict) -> Network:
    net = Network(
        height=cfg["height"],
        width=cfg["width"],
        bgcolor=cfg["bgcolor"],
        font_color=cfg["font_color"],
        notebook=False,
    )

    # ── node size scaling ──
    frequencies = [G.nodes[n].get("frequency", 1) for n in G.nodes()]
    freq_min, freq_max = min(frequencies), max(frequencies)
    degrees = dict(G.degree())

    entity_type_shapes = {
        "Model":     "diamond",
        "Technique": "ellipse",
        "Tool":      "box",
        "Task":      "triangle",
        "Dataset":   "star",
        "Metric":    "dot",
        "Concept":   "ellipse",
        "Domain":    "hexagon",
    }

    for node in G.nodes():
        freq  = G.nodes[node].get("frequency", 1)
        etype = G.nodes[node].get("entity_type", "Concept")
        cid   = partition.get(node, 0)
        color = get_community_color(cid)
        size  = _scale(freq, freq_min, freq_max,
                       cfg["min_node_size"], cfg["max_node_size"])
        shape = entity_type_shapes.get(etype, "dot")

        net.add_node(
            node,
            label=node,
            color=color,
            size=size,
            shape=shape,
            title=(
                f"<b>{node}</b><br>"
                f"Type: {etype}<br>"
                f"Frequency: {freq}<br>"
                f"Degree: {degrees[node]}<br>"
                f"Community: {cid}"
            ),
        )

    # ── edge width scaling ──
    weights = [d.get("weight", 1) for _, _, d in G.edges(data=True)]
    w_min, w_max = (min(weights), max(weights)) if weights else (1, 1)

    for u, v, data in G.edges(data=True):
        w     = data.get("weight", 1)
        width = _scale(w, w_min, w_max, cfg["edge_width_min"], cfg["edge_width_max"])
        # same community → highlight edge
        same_comm = partition.get(u, -1) == partition.get(v, -2)
        color = "#888888" if not same_comm else get_community_color(partition.get(u, 0))

        net.add_edge(
            u, v,
            value=width,
            title=f"Weight: {w:.2f}",
            color={"color": color, "opacity": 0.7},
        )

    # ── physics / interaction options ──
    net.set_options(f"""
    {{
      "physics": {{
        "solver": "{cfg['physics_solver']}",
        "forceAtlas2Based": {{
          "gravitationalConstant": -60,
          "centralGravity": 0.01,
          "springLength": 120,
          "springConstant": 0.08,
          "damping": 0.4
        }},
        "minVelocity": 0.75,
        "stabilization": {{ "iterations": 200 }}
      }},
      "interaction": {{
        "hover": true,
        "tooltipDelay": 150,
        "navigationButtons": true,
        "keyboard": true
      }},
      "nodes": {{
        "font": {{ "size": 13, "face": "Inter, Arial, sans-serif" }},
        "borderWidth": 2,
        "borderWidthSelected": 4
      }},
      "edges": {{
        "smooth": {{ "type": "continuous" }}
      }}
    }}
    """)

    return net


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 – SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

def save_visualization_metadata(G: nx.Graph, partition: dict, output_path: str):
    community_map = defaultdict(list)
    for node, cid in partition.items():
        community_map[cid].append(node)

    meta = {
        "num_nodes":       G.number_of_nodes(),
        "num_edges":       G.number_of_edges(),
        "num_communities": len(community_map),
        "communities": {
            str(cid): {
                "size":  len(members),
                "color": get_community_color(cid),
                "nodes": members,
            }
            for cid, members in community_map.items()
        },
        "top_10_nodes_by_degree": sorted(
            [(n, d) for n, d in G.degree()],
            key=lambda x: x[1], reverse=True
        )[:10],
    }

    with open(output_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Metadata saved → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=== Week 10: Knowledge Graph Visualization ===")

    # 1. Load
    G = load_graph(GRAPH_INPUT_PATH)

    # 2. Prune (optional cap)
    G = prune_graph(G, VIS_CONFIG["top_k_nodes"])

    # 3. Community detection (reuse Week 9 output if available)
    partition = detect_communities(G)

    # 4. Build PyVis network
    net = build_pyvis_network(G, partition, VIS_CONFIG)

    # 5. Save HTML
    net.save_graph(HTML_OUTPUT_PATH)
    logger.info(f"Interactive visualization saved → {HTML_OUTPUT_PATH}")

    # 6. Save metadata JSON
    save_visualization_metadata(G, partition, META_OUTPUT_PATH)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Week 10 Deliverables")
    print("=" * 60)
    print(f"  HTML visualization : {HTML_OUTPUT_PATH}")
    print(f"  Metadata JSON      : {META_OUTPUT_PATH}")
    print(f"  Nodes rendered     : {G.number_of_nodes()}")
    print(f"  Edges rendered     : {G.number_of_edges()}")
    print(f"  Communities found  : {len(set(partition.values()))}")
    print("=" * 60)


if __name__ == "__main__":
    main()