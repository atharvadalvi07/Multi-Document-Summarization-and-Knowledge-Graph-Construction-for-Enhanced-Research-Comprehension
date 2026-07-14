"""
Week 11: Filtering and Layer Controls for Knowledge Graph Visualization
=======================================================================
Deliverable: Updated interactive HTML visualization with:
  - Community/topic layer filtering
  - Entity type filtering
  - Edge type filtering
  - Degree-based node filtering (min degree slider)
  - Search/highlight by node name
  - Legend with toggle controls

Inputs  (from previous weeks):
  - outputs/week9/community_assignments.json   (Week 9)
  - outputs/week8/enhanced_graph.json          (Week 8)
  - outputs/week5/entities.json                (Week 5, optional)
  - outputs/week10/knowledge_graph.html        (Week 10 base viz, optional)

Outputs:
  - outputs/week11/knowledge_graph_filtered.html
  - outputs/week11/viz_manifest.json
"""

import json
import os
import math
import re
from pathlib import Path
from collections import defaultdict

import networkx as nx

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = Path("outputs/week11")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Color palettes ────────────────────────────────────────────────────────────
COMMUNITY_COLORS = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
    "#D4A5A5", "#A8D5BA", "#FFD580", "#B5C8E2", "#F4A261",
]

ENTITY_TYPE_COLORS = {
    "method":      "#4E79A7",
    "dataset":     "#F28E2B",
    "technique":   "#E15759",
    "task":        "#59A14F",
    "institution": "#B07AA1",
    "concept":     "#76B7B2",
    "unknown":     "#BAB0AC",
}

EDGE_TYPE_COLORS = {
    "co_occurrence": "#999999",
    "dependency":    "#E15759",
    "semantic":      "#4E79A7",
    "citation":      "#F28E2B",
    "unknown":       "#CCCCCC",
}


# ═════════════════════════════════════════════════════════════════════════════
# 1.  LOAD PIPELINE OUTPUTS
# ═════════════════════════════════════════════════════════════════════════════

def load_json_safe(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"  [WARN] Not found: {path} — using default.")
        return default
    except json.JSONDecodeError as e:
        print(f"  [WARN] JSON error in {path}: {e} — using default.")
        return default


def load_graph_from_json(path):
    """
    Load a NetworkX graph from the enhanced_graph.json produced by Week 8.
    Handles both node-link format and a dict with 'nodes'/'edges' keys.
    """
    data = load_json_safe(path, default=None)
    if data is None:
        return None

    # Try NetworkX node-link format first
    try:
        G = nx.node_link_graph(data)
        print(f"  Loaded graph via node_link_graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    except Exception:
        pass

    # Fallback: manual construction from {'nodes': [...], 'edges': [...]}
    try:
        G = nx.Graph()
        for n in data.get("nodes", []):
            nid = n.get("id", n.get("label", str(n)))
            attrs = {k: v for k, v in n.items() if k != "id"}
            G.add_node(nid, **attrs)
        for e in data.get("edges", []):
            src = e.get("source", e.get("from"))
            tgt = e.get("target", e.get("to"))
            if src is not None and tgt is not None:
                attrs = {k: v for k, v in e.items() if k not in ("source", "target", "from", "to")}
                G.add_edge(src, tgt, **attrs)
        print(f"  Loaded graph manually: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    except Exception as ex:
        print(f"  [ERROR] Could not parse graph JSON: {ex}")
        return None


def build_synthetic_graph():
    """
    Fallback: build a small demo graph so the script always produces output.
    """
    print("  Building synthetic demo graph …")
    G = nx.barabasi_albert_graph(40, 3, seed=42)
    entity_types = list(ENTITY_TYPE_COLORS.keys())
    edge_types   = list(EDGE_TYPE_COLORS.keys())
    for i, node in enumerate(G.nodes()):
        G.nodes[node]["label"]       = f"entity_{node}"
        G.nodes[node]["entity_type"] = entity_types[i % len(entity_types)]
    for u, v, d in G.edges(data=True):
        d["edge_type"] = edge_types[(u + v) % len(edge_types)]
        d["weight"]    = round(0.3 + (u % 5) * 0.15, 2)
    return G


# ═════════════════════════════════════════════════════════════════════════════
# 2.  ENRICH GRAPH WITH COMMUNITY & ENTITY METADATA
# ═════════════════════════════════════════════════════════════════════════════

def enrich_graph(G, community_data, entities_data):
    """
    Attach community IDs, entity types, and edge types to the graph.
    """
    # ── Community assignments ──────────────────────────────────────────────
    community_map = {}  # node → community_id (int)
    if community_data:
        raw = community_data
        # Support {'node': community_id} or {'communities': {node: id}} or list
        if isinstance(raw, dict):
            if "communities" in raw:
                raw = raw["communities"]
            # Now raw should be {node: community_id} or {community_id: [nodes]}
            # Detect which format
            sample_val = next(iter(raw.values()), None)
            if isinstance(sample_val, list):
                # {community_id: [nodes]}
                for cid, nodes in raw.items():
                    for n in nodes:
                        community_map[str(n)] = int(cid)
            else:
                # {node: community_id}
                for n, cid in raw.items():
                    community_map[str(n)] = int(cid) if str(cid).lstrip("-").isdigit() else hash(str(cid)) % 15

    # ── Entity type map from entities.json ────────────────────────────────
    entity_type_map = {}
    if entities_data:
        # Support list of {'text': ..., 'label': ...} dicts
        if isinstance(entities_data, list):
            for ent in entities_data:
                if isinstance(ent, dict):
                    text  = ent.get("text", ent.get("entity", ""))
                    label = ent.get("label", ent.get("type", "unknown")).lower()
                    entity_type_map[text.lower()] = label

    # ── Apply to nodes ─────────────────────────────────────────────────────
    for node in G.nodes():
        ndata = G.nodes[node]
        node_str = str(node)

        # Community
        cid = community_map.get(node_str, community_map.get(node, None))
        if cid is None:
            cid = hash(node_str) % 10  # deterministic fallback
        ndata["community"] = cid
        ndata["color"]     = COMMUNITY_COLORS[cid % len(COMMUNITY_COLORS)]

        # Entity type
        label = ndata.get("label", node_str).lower()
        etype = (ndata.get("entity_type")
                 or entity_type_map.get(label)
                 or entity_type_map.get(node_str.lower())
                 or "unknown")
        ndata["entity_type"] = etype.lower()

        # Node size based on degree
        deg = G.degree(node)
        ndata["size"] = max(3, min(10, 3 + deg*0.25))

        # Display label
        if "label" not in ndata or not ndata["label"]:
            ndata["label"] = node_str

    # ── Apply to edges ─────────────────────────────────────────────────────
    for u, v, d in G.edges(data=True):
        if "edge_type" not in d or not d["edge_type"]:
            d["edge_type"] = "co_occurrence"
        etype = d["edge_type"].lower()
        d["color"] = EDGE_TYPE_COLORS.get(etype, EDGE_TYPE_COLORS["unknown"])
        if "weight" not in d:
            d["weight"] = 1.0

    return G


# ═════════════════════════════════════════════════════════════════════════════
# 3.  COMPUTE LAYOUT
# ═════════════════════════════════════════════════════════════════════════════

def compute_layout(G):
    """
    Spring layout, scaled to a ±500 viewport.
    Returns {node: (x, y)} with float coordinates.
    """
    n = G.number_of_nodes()
    k = k = 3.0/math.sqrt(n) if n > 1 else 1.0
    pos = nx.spring_layout(G, k=k, iterations=120, seed=42)
    # Scale to [-480, 480]
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_range = x_max - x_min or 1
    y_range = y_max - y_min or 1
    scaled = {}
    for node, (x, y) in pos.items():
        # scaled[node] = (
        #     round(((x - x_min) / x_range * 3200) - 6400, 2),
        #     round(((y - y_min) / y_range * 3200) - 6400, 2),
        # )
        scaled[node] = (
            round(((x - (x_min + x_max) / 2) / x_range * 1600), 2),
            round(((y - (y_min + y_max) / 2) / y_range * 1600), 2),
        )
    return scaled


# ═════════════════════════════════════════════════════════════════════════════
# 4.  SERIALISE GRAPH TO JSON FOR JS
# ═════════════════════════════════════════════════════════════════════════════

def graph_to_js_data(G, layout):
    nodes = []
    for node in G.nodes():
        d    = G.nodes[node]
        x, y = layout.get(node, (0, 0))
        nodes.append({
            "id":          str(node),
            "label":       d.get("label", str(node)),
            "community":   d.get("community", 0),
            "entity_type": d.get("entity_type", "unknown"),
            "color":       d.get("color", "#BAB0AC"),
            "size":        d.get("size", 10),
            "degree":      G.degree(node),
            "x":           x,
            "y":           y,
        })

    edges = []
    for u, v, d in G.edges(data=True):
        edges.append({
            "source":    str(u),
            "target":    str(v),
            "edge_type": d.get("edge_type", "co_occurrence"),
            "color":     d.get("color", "#999999"),
            "weight":    round(float(d.get("weight", 1.0)), 3),
        })

    communities = sorted(set(n["community"] for n in nodes))
    entity_types = sorted(set(n["entity_type"] for n in nodes))
    edge_types   = sorted(set(e["edge_type"]  for e in edges))

    return {
        "nodes":        nodes,
        "edges":        edges,
        "communities":  communities,
        "entity_types": entity_types,
        "edge_types":   edge_types,
        "stats": {
            "num_nodes":  len(nodes),
            "num_edges":  len(edges),
            "density":    round(nx.density(G), 4),
            "num_communities": len(communities),
        }
    }


# ═════════════════════════════════════════════════════════════════════════════
# 5.  BUILD THE HTML PAGE
# ═════════════════════════════════════════════════════════════════════════════

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Knowledge Graph — Week 11 (Filtered Visualization)</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', Arial, sans-serif; background: #1a1a2e; color: #e0e0e0; display: flex; height: 100vh; overflow: hidden; }

  /* ── Sidebar ── */
  #sidebar {
    width: 280px; min-width: 240px; background: #16213e; padding: 14px;
    display: flex; flex-direction: column; gap: 14px; overflow-y: auto;
    border-right: 1px solid #0f3460; flex-shrink: 0;
  }
  #sidebar h1 { font-size: 14px; color: #e94560; letter-spacing: 1px; text-transform: uppercase; }
  .panel { background: #0f3460; border-radius: 8px; padding: 10px; }
  .panel h3 { font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #a0aec0; margin-bottom: 8px; }

  /* Search */
  #search-input {
    width: 100%; padding: 7px 10px; background: #1a1a2e; border: 1px solid #0f3460;
    border-radius: 6px; color: #e0e0e0; font-size: 13px; outline: none;
  }
  #search-input:focus { border-color: #e94560; }

  /* Degree slider */
  .slider-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; font-size: 12px; }
  input[type=range] { width: 100%; accent-color: #e94560; }

  /* Toggle buttons */
  .toggle-group { display: flex; flex-wrap: wrap; gap: 5px; }
  .tog {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 8px; border-radius: 12px; font-size: 11px; cursor: pointer;
    border: 1px solid transparent; transition: opacity .2s, border-color .2s;
    user-select: none;
  }
  .tog .dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
  .tog.off { opacity: 0.35; border-color: #555; }
  .tog:hover { opacity: 1; }

  /* Stats */
  .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
  .stat-box { background: #1a1a2e; border-radius: 6px; padding: 6px 10px; text-align: center; }
  .stat-box .val { font-size: 18px; font-weight: 700; color: #e94560; }
  .stat-box .lbl { font-size: 10px; color: #a0aec0; text-transform: uppercase; }

  /* ── Canvas area ── */
  #canvas-wrap { flex: 1; position: relative; }
  #canvas { display: block; width: 100%; height: 100%; cursor: grab; }
  #canvas:active { cursor: grabbing; }

  /* Tooltip */
  #tooltip {
    position: absolute; background: rgba(15,52,96,0.95); border: 1px solid #e94560;
    border-radius: 8px; padding: 10px 14px; font-size: 12px; pointer-events: none;
    max-width: 220px; line-height: 1.6; display: none; z-index: 10;
  }
  #tooltip .tt-title { font-size: 14px; font-weight: 700; color: #e94560; margin-bottom: 4px; }

  /* Reset button */
  #btn-reset {
    position: absolute; top: 12px; right: 12px; z-index: 5;
    background: #e94560; color: #fff; border: none; border-radius: 6px;
    padding: 6px 14px; cursor: pointer; font-size: 12px; letter-spacing: .5px;
  }
  #btn-reset:hover { background: #c73652; }

  /* Info bar */
  #info-bar {
    position: absolute; bottom: 0; left: 0; right: 0;
    background: rgba(15,52,96,0.85); font-size: 11px; color: #a0aec0;
    padding: 5px 12px; display: flex; gap: 20px;
  }
</style>
</head>
<body>

<!-- ═══════════ SIDEBAR ═══════════ -->
<div id="sidebar">
  <h1>Knowledge Graph</h1>

  <!-- Stats -->
  <div class="panel">
    <h3>Graph Stats</h3>
    <div class="stat-grid">
      <div class="stat-box"><div class="val" id="s-nodes">—</div><div class="lbl">Visible Nodes</div></div>
      <div class="stat-box"><div class="val" id="s-edges">—</div><div class="lbl">Visible Edges</div></div>
      <div class="stat-box"><div class="val" id="s-comm">—</div><div class="lbl">Communities</div></div>
      <div class="stat-box"><div class="val" id="s-dens">—</div><div class="lbl">Density</div></div>
    </div>
  </div>

  <!-- Search -->
  <div class="panel">
    <h3>Search / Highlight</h3>
    <input id="search-input" type="text" placeholder="Type node name…"/>
  </div>

  <!-- Degree filter -->
  <div class="panel">
    <h3>Min. Degree Filter</h3>
    <div class="slider-row">
      <span>Degree ≥ <b id="deg-val">1</b></span>
      <span id="deg-count" style="color:#e94560;"></span>
    </div>
    <input id="deg-slider" type="range" min="1"  value="36"/>
  </div>

  <!-- Community filter -->
  <div class="panel">
    <h3>Communities (Topics)</h3>
    <div id="comm-toggles" class="toggle-group"></div>
  </div>

  <!-- Entity type filter -->
  <div class="panel">
    <h3>Entity Types</h3>
    <div id="etype-toggles" class="toggle-group"></div>
  </div>

  <!-- Edge type filter -->
  <div class="panel">
    <h3>Edge Types</h3>
    <div id="edgetype-toggles" class="toggle-group"></div>
  </div>

  <!-- Controls hint -->
  <div style="font-size:10px;color:#718096;padding:4px;">
    Scroll to zoom · Drag canvas to pan · Click node to inspect
  </div>
</div>

<!-- ═══════════ CANVAS ═══════════ -->
<div id="canvas-wrap">
  <canvas id="canvas"></canvas>
  <div id="tooltip"></div>
  <button id="btn-reset">⟳ Reset View</button>
  <div id="info-bar">
    <span id="ib-zoom">Zoom: 1.00×</span>
    <span id="ib-selected">Selected: —</span>
  </div>
</div>

<script>
// ══════════════════════════════════════════════════════
//  DATA (injected by Python)
// ══════════════════════════════════════════════════════
const GRAPH_DATA = __GRAPH_DATA__;

// ══════════════════════════════════════════════════════
//  STATE
// ══════════════════════════════════════════════════════
const canvas   = document.getElementById('canvas');
const ctx      = canvas.getContext('2d');
const tooltip  = document.getElementById('tooltip');

let visibleComms   = new Set(GRAPH_DATA.communities);
let visibleEtypes  = new Set(GRAPH_DATA.entity_types);
let visibleEtypes2 = new Set(GRAPH_DATA.edge_types);   // edge types
let minDegree      = 1;
let searchTerm     = '';
let selectedNode   = null;

// Camera
let camX = 0, camY = 0, camScale = 1;
let dragging = false, dragStart = {x:0, y:0}, camStart = {x:0, y:0};

// Pre-build node map
const nodeMap = {};
GRAPH_DATA.nodes.forEach(n => { nodeMap[n.id] = n; });

// ══════════════════════════════════════════════════════
//  FILTERING
// ══════════════════════════════════════════════════════
function isNodeVisible(n) {
  if (!visibleComms.has(n.community))    return false;
  if (!visibleEtypes.has(n.entity_type)) return false;
  if (n.degree < minDegree)              return false;
  return true;
}
function isEdgeVisible(e) {
  if (!visibleEtypes2.has(e.edge_type)) return false;
  const s = nodeMap[e.source], t = nodeMap[e.target];
  return s && t && isNodeVisible(s) && isNodeVisible(t);
}

// ══════════════════════════════════════════════════════
//  RESIZE
// ══════════════════════════════════════════════════════
function fitGraph() {
  const nodes = GRAPH_DATA.nodes;
  if (!nodes.length) return;
  const xs = nodes.map(n => n.x), ys = nodes.map(n => n.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const graphW = maxX - minX || 1, graphH = maxY - minY || 1;
  const padding = 60;
  const scaleX = (canvas.width  - padding * 2) / graphW;
  const scaleY = (canvas.height - padding * 2) / graphH;
  camScale = 1.7;
  const midX = (minX + maxX) / 2;
  const midY = (minY + maxY) / 2;
  camX = -500;
  camY = 1000;

}
function resize() {
  const wrap = document.getElementById('canvas-wrap');
  canvas.width  = wrap.clientWidth;
  canvas.height = wrap.clientHeight;
  fitGraph();
  draw();
}
window.addEventListener('resize', resize);
window.addEventListener('load', () => { setTimeout(() => { resize(); }, 500); });

// ══════════════════════════════════════════════════════
//  DRAW
// ══════════════════════════════════════════════════════
function worldToScreen(x, y) {
  return [ x * camScale + camX + canvas.width/2,
           y * camScale + camY + canvas.height/2 ];
}
function screenToWorld(sx, sy) {
  return [ (sx - canvas.width/2  - camX) / camScale,
           (sy - canvas.height/2 - camY) / camScale ];
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const visNodes = GRAPH_DATA.nodes.filter(isNodeVisible);
  const visEdges = GRAPH_DATA.edges.filter(isEdgeVisible);
  const visNodeIds = new Set(visNodes.map(n => n.id));

  // Update stats
  document.getElementById('s-nodes').textContent = visNodes.length;
  document.getElementById('s-edges').textContent = visEdges.length;
  document.getElementById('s-comm').textContent  = new Set(visNodes.map(n=>n.community)).size;
  document.getElementById('s-dens').textContent  = GRAPH_DATA.stats.density;

  const search = searchTerm.toLowerCase().trim();

  // ── Edges ──
  visEdges.forEach(e => {
    const s = nodeMap[e.source], t = nodeMap[e.target];
    const [sx, sy] = worldToScreen(s.x, s.y);
    const [tx, ty] = worldToScreen(t.x, t.y);
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.lineTo(tx, ty);
    ctx.strokeStyle = e.color + '88';
    ctx.lineWidth = Math.min(0.5, 0.3 + e.weight * 0.3);   
    ctx.stroke();
  });

  // ── Nodes ──
  visNodes.forEach(n => {
    const [sx, sy] = worldToScreen(n.x, n.y);
    const r = (n.size / 2) * Math.sqrt(camScale) * 0.6;
    const isSelected = selectedNode && selectedNode.id === n.id;
    const isMatch    = search && n.label.toLowerCase().includes(search);

    // Glow for selected / search match
    if (isSelected || isMatch) {
      ctx.beginPath();
      ctx.arc(sx, sy, r + 6, 0, Math.PI * 2);
      ctx.fillStyle = isSelected ? '#e9456088' : '#ffd70088';
      ctx.fill();
    }

    ctx.beginPath();
    ctx.arc(sx, sy, r, 0, Math.PI * 2);
    ctx.fillStyle = n.color;
    ctx.fill();
    ctx.strokeStyle = isSelected ? '#e94560' : '#ffffff33';
    ctx.lineWidth   = isSelected ? 2 : 0.8;
    ctx.stroke();

    // Label (only if zoomed in enough or selected/match)
    if (camScale > 0.6 || isSelected || isMatch) {
      const fs = Math.max(9, Math.min(14, 10 * camScale));
      ctx.font      = `${fs}px Segoe UI, Arial, sans-serif`;
      ctx.fillStyle = isSelected ? '#e94560' : (isMatch ? '#ffd700' : '#ffffff');
      ctx.textAlign = 'center';
      ctx.fillText(n.label, sx, sy + r + fs + 2);
    }
  });

  // Zoom info
  document.getElementById('ib-zoom').textContent = `Zoom: ${camScale.toFixed(2)}×`;
  document.getElementById('ib-selected').textContent =
    selectedNode ? `Selected: ${selectedNode.label} (deg ${selectedNode.degree})` : 'Selected: —';
}

// ══════════════════════════════════════════════════════
//  MOUSE EVENTS
// ══════════════════════════════════════════════════════
canvas.addEventListener('mousedown', e => {
  dragging  = true;
  dragStart = { x: e.clientX, y: e.clientY };
  camStart  = { x: camX, y: camY };
  selectedNode = hitTest(e.clientX, e.clientY);
  showTooltip(selectedNode, e.clientX, e.clientY);
  draw();
});
canvas.addEventListener('mousemove', e => {
  if (dragging) {
    camX = camStart.x + (e.clientX - dragStart.x);
    camY = camStart.y + (e.clientY - dragStart.y);
    draw();
  } else {
    const hit = hitTest(e.clientX, e.clientY);
    showTooltip(hit, e.clientX, e.clientY);
    canvas.style.cursor = hit ? 'pointer' : 'grab';
  }
});
canvas.addEventListener('mouseup',   () => { dragging = false; });
canvas.addEventListener('mouseleave',() => { dragging = false; tooltip.style.display = 'none'; });

canvas.addEventListener('wheel', e => {
  e.preventDefault();
  const factor = e.deltaY < 0 ? 1.12 : 1/1.12;
  camScale = Math.max(0.05, Math.min(10, camScale * factor));
  draw();
}, { passive: false });

document.getElementById('btn-reset').addEventListener('click', () => {
  fitGraph(); selectedNode = null; draw();
});

// ── Hit test ──────────────────────────────────────────
function hitTest(mx, my) {
  for (const n of GRAPH_DATA.nodes) {
    if (!isNodeVisible(n)) continue;
    const [sx, sy] = worldToScreen(n.x, n.y);
    const r = (n.size / 2) * Math.sqrt(camScale) + 4;
    if ((mx-sx)**2 + (my-sy)**2 < r*r) return n;
  }
  return null;
}

// ── Tooltip ───────────────────────────────────────────
function showTooltip(node, mx, my) {
  if (!node) { tooltip.style.display = 'none'; return; }
  const w = canvas.getBoundingClientRect();
  const rx = mx - w.left, ry = my - w.top;

  // Neighbours
  const nbrs = GRAPH_DATA.edges
    .filter(e => (e.source === node.id || e.target === node.id) && isEdgeVisible(e))
    .map(e => e.source === node.id ? e.target : e.source)
    .slice(0, 5)
    .map(id => nodeMap[id]?.label || id)
    .join(', ');

  tooltip.innerHTML = `
    <div class="tt-title">${node.label}</div>
    <div><b>Entity type:</b> ${node.entity_type}</div>
    <div><b>Community:</b> ${node.community}</div>
    <div><b>Degree:</b> ${node.degree}</div>
    ${nbrs ? `<div><b>Neighbours:</b> ${nbrs}${GRAPH_DATA.edges.filter(e=>(e.source===node.id||e.target===node.id)&&isEdgeVisible(e)).length>5?'…':''}</div>` : ''}
  `;
  tooltip.style.display = 'block';
  const tw = 230, th = 110;
  tooltip.style.left = (rx + 14 + tw < canvas.width  ? rx + 14 : rx - tw - 14) + 'px';
  tooltip.style.top  = (ry + 14 + th < canvas.height ? ry + 14 : ry - th - 14) + 'px';
}

// ══════════════════════════════════════════════════════
//  SIDEBAR CONTROLS
// ══════════════════════════════════════════════════════

// ── Community toggles ─────────────────────────────────
const commContainer = document.getElementById('comm-toggles');
GRAPH_DATA.communities.forEach(cid => {
  const col = "__COMMUNITY_COLORS__"[cid % "__COMM_LEN__"];
  const btn = document.createElement('div');
  btn.className = 'tog';
  btn.dataset.cid = cid;
  btn.style.background = col + '33';
  btn.style.borderColor = col;
  btn.innerHTML = `<span class="dot" style="background:${col}"></span>C${cid}`;
  btn.addEventListener('click', () => {
    if (visibleComms.has(cid)) visibleComms.delete(cid);
    else                        visibleComms.add(cid);
    btn.classList.toggle('off', !visibleComms.has(cid));
    draw();
  });
  commContainer.appendChild(btn);
});

// ── Entity-type toggles ───────────────────────────────
const etypeContainer = document.getElementById('etype-toggles');
const ETYPE_COLORS = __ETYPE_COLORS__;
GRAPH_DATA.entity_types.forEach(et => {
  const col = ETYPE_COLORS[et] || '#BAB0AC';
  const btn = document.createElement('div');
  btn.className = 'tog';
  btn.dataset.et = et;
  btn.style.background = col + '33';
  btn.style.borderColor = col;
  btn.innerHTML = `<span class="dot" style="background:${col}"></span>${et}`;
  btn.addEventListener('click', () => {
    if (visibleEtypes.has(et)) visibleEtypes.delete(et);
    else                        visibleEtypes.add(et);
    btn.classList.toggle('off', !visibleEtypes.has(et));
    draw();
  });
  etypeContainer.appendChild(btn);
});

// ── Edge-type toggles ─────────────────────────────────
const edgeContainer = document.getElementById('edgetype-toggles');
const EDGE_COLORS = __EDGE_COLORS__;
GRAPH_DATA.edge_types.forEach(et => {
  const col = EDGE_COLORS[et] || '#CCCCCC';
  const btn = document.createElement('div');
  btn.className = 'tog';
  btn.dataset.et = et;
  btn.style.background = col + '33';
  btn.style.borderColor = col;
  btn.innerHTML = `<span class="dot" style="background:${col}"></span>${et}`;
  btn.addEventListener('click', () => {
    if (visibleEtypes2.has(et)) visibleEtypes2.delete(et);
    else                         visibleEtypes2.add(et);
    btn.classList.toggle('off', !visibleEtypes2.has(et));
    draw();
  });
  edgeContainer.appendChild(btn);
});

// ── Degree slider ─────────────────────────────────────
const slider = document.getElementById('deg-slider');
const degVal = document.getElementById('deg-val');
const degCnt = document.getElementById('deg-count');
const maxDeg = Math.max(...GRAPH_DATA.nodes.map(n => n.degree));
slider.max   = Math.max(1, maxDeg);

function updateDegreeSlider() {
  minDegree = parseInt(slider.value);
  degVal.textContent = minDegree;
  const cnt = GRAPH_DATA.nodes.filter(n => n.degree >= minDegree).length;
  degCnt.textContent = cnt + ' nodes';
  draw();
}
slider.addEventListener('input', updateDegreeSlider);
updateDegreeSlider();

// ── Search ─────────────────────────────────────────────
document.getElementById('search-input').addEventListener('input', e => {
  searchTerm = e.target.value;
  draw();
});

// ══════════════════════════════════════════════════════
//  INIT
// ══════════════════════════════════════════════════════
setTimeout(resize, 500);
</script>
</body>
</html>
"""


def build_html(graph_data):
    """
    Inject Python data into the HTML template and return a complete HTML string.
    """
    comm_colors_js = json.dumps(COMMUNITY_COLORS)
    etype_colors_js = json.dumps(ENTITY_TYPE_COLORS)
    edge_colors_js  = json.dumps(EDGE_TYPE_COLORS)

    html = HTML_TEMPLATE
    html = html.replace('__GRAPH_DATA__',     json.dumps(graph_data, ensure_ascii=False))
    html = html.replace('"__COMMUNITY_COLORS__"', comm_colors_js)
    html = html.replace('"__COMM_LEN__"',         str(len(COMMUNITY_COLORS)))
    html = html.replace('__ETYPE_COLORS__',    etype_colors_js)
    html = html.replace('__EDGE_COLORS__',     edge_colors_js)
    return html


# ═════════════════════════════════════════════════════════════════════════════
# 6.  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Week 11 — Filtering and Layer Controls Visualization")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n[1/5] Loading pipeline outputs …")
    G = load_graph_from_json("outputs/week8_outputs/enhanced_graph.json")
    if G is None:
        G = build_synthetic_graph()

    community_data = load_json_safe("outputs/week9_outputs/community_assignments.json", default={})
    entities_data  = load_json_safe("outputs/entity_frequencies.json",              default=[])

    # ── Enrich ────────────────────────────────────────────────────────────
    print("\n[2/5] Enriching graph with metadata …")
    G = enrich_graph(G, community_data, entities_data)

    # ── Layout ────────────────────────────────────────────────────────────
    print("\n[3/5] Computing spring layout …")
    layout = compute_layout(G)

    # ── Serialise ─────────────────────────────────────────────────────────
    print("\n[4/5] Serialising graph …")
    graph_data = graph_to_js_data(G, layout)

    # Save manifest JSON
    manifest_path = OUTPUT_DIR / "viz_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "week": 11,
            "stats": graph_data["stats"],
            "communities":  graph_data["communities"],
            "entity_types": graph_data["entity_types"],
            "edge_types":   graph_data["edge_types"],
        }, f, indent=2)
    print(f"  Manifest saved → {manifest_path}")

    # ── Build HTML ────────────────────────────────────────────────────────
    print("\n[5/5] Building HTML visualization …")
    html = build_html(graph_data)

    out_path = OUTPUT_DIR / "knowledge_graph_filtered.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  HTML saved → {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    s = graph_data["stats"]
    print("\n" + "=" * 60)
    print("Week 11 Complete")
    print(f"  Nodes       : {s['num_nodes']}")
    print(f"  Edges       : {s['num_edges']}")
    print(f"  Density     : {s['density']}")
    print(f"  Communities : {s['num_communities']}")
    print(f"  Entity types: {graph_data['entity_types']}")
    print(f"  Edge types  : {graph_data['edge_types']}")
    print("\nDeliverables:")
    print(f"  {out_path}")
    print(f"  {manifest_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()