"""
Week 12: Evaluation Experiments
Evaluates summary quality (ROUGE, BLEU, compression ratio) and
graph structural metrics (density, degree, clustering coefficient,
community detection quality), then writes a JSON results report.
"""

import os
import json
import math
import glob
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = "week12_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  SUMMARY QUALITY EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def load_summaries():
    """
    Week 3  → baseline_summaries.json
              fields: original (reference), summary (hypothesis)
              + pre-computed rouge1, rouge2, rougeL, compression_ratio

    Week 4  → clustering_results.json
              fields per cluster: individual_merge_summary, joint_summary
              + pre-computed individual_rouge* / joint_rouge* scores
    """
    pairs       = {"baseline": [], "clustered_individual": [], "clustered_joint": []}
    precomputed = {"baseline": [], "clustered_individual": [], "clustered_joint": []}

    # ── Week 3 ───────────────────────────────────────────────────────────────
    w3_candidates = (glob.glob("outputs/baseline_summaries.json"))
    for path in w3_candidates:
        try:
            with open(path) as f:
                items = json.load(f)
            if not isinstance(items, list) or not items:
                continue
            if "original" not in items[0]:
                continue
            print(f"  Week 3: loaded {path} ({len(items)} papers)")
            for item in items:
                ref = item.get("original", "")
                hyp = item.get("summary", "")
                if ref and hyp:
                    pairs["baseline"].append((ref.strip(), hyp.strip()))
                    precomputed["baseline"].append({
                        "rouge1":           item.get("rouge1"),
                        "rouge2":           item.get("rouge2"),
                        "rougeL":           item.get("rougeL"),
                        "compression_ratio": item.get("compression_ratio"),
                    })
            break
        except Exception as e:
            print(f"  [warn] {path}: {e}")

    # ── Week 4 ───────────────────────────────────────────────────────────────
    w4_candidates = (glob.glob("outputs/clustering_results.json"))
    for path in w4_candidates:
        try:
            with open(path) as f:
                items = json.load(f)
            if not isinstance(items, list) or not items:
                continue
            if "individual_merge_summary" not in items[0]:
                continue
            print(f"  Week 4: loaded {path} ({len(items)} clusters)")
            for item in items:
                ind = item.get("individual_merge_summary", "")
                jnt = item.get("joint_summary", "")
                if ind:
                    pairs["clustered_individual"].append(("", ind.strip()))
                    precomputed["clustered_individual"].append({
                        "rouge1": item.get("individual_rouge1"),
                        "rouge2": item.get("individual_rouge2"),
                        "rougeL": item.get("individual_rougeL"),
                    })
                if jnt:
                    pairs["clustered_joint"].append(("", jnt.strip()))
                    precomputed["clustered_joint"].append({
                        "rouge1": item.get("joint_rouge1"),
                        "rouge2": item.get("joint_rouge2"),
                        "rougeL": item.get("joint_rougeL"),
                    })
            break
        except Exception as e:
            print(f"  [warn] {path}: {e}")

    return pairs, precomputed


# ── Lightweight ROUGE-L ───────────────────────────────────────────────────────
def _lcs_length(a_tokens, b_tokens):
    m, n = len(a_tokens), len(b_tokens)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a_tokens[i - 1] == b_tokens[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr
    return prev[n]

def rouge_l(reference: str, hypothesis: str) -> dict:
    ref_tok = reference.lower().split()
    hyp_tok = hypothesis.lower().split()
    lcs = _lcs_length(ref_tok, hyp_tok)
    p  = lcs / len(hyp_tok) if hyp_tok else 0.0
    r  = lcs / len(ref_tok)  if ref_tok  else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


# ── Lightweight BLEU-1..4 ────────────────────────────────────────────────────
def _ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def bleu(reference: str, hypothesis: str, max_n: int = 4) -> dict:
    ref_tok = reference.lower().split()
    hyp_tok = hypothesis.lower().split()
    if not hyp_tok:
        return {f"bleu_{n}": 0.0 for n in range(1, max_n + 1)}
    log_sum = 0.0
    scores = {}
    for n in range(1, max_n + 1):
        ref_ng = _ngrams(ref_tok, n)
        hyp_ng = _ngrams(hyp_tok, n)
        if not hyp_ng:
            scores[f"bleu_{n}"] = 0.0
            log_sum = -math.inf
            continue
        ref_counts = {}
        for g in ref_ng:
            ref_counts[g] = ref_counts.get(g, 0) + 1
        clip = sum(min(hyp_ng.count(g), ref_counts.get(g, 0)) for g in set(hyp_ng))
        p = clip / len(hyp_ng)
        scores[f"bleu_{n}"] = round(p, 4)
        log_sum += math.log(p) if p > 0 else -math.inf
    bp = 1.0 if len(hyp_tok) >= len(ref_tok) else math.exp(1 - len(ref_tok) / len(hyp_tok))
    geo = math.exp(log_sum / max_n) if log_sum != -math.inf else 0.0
    scores["bleu_avg"] = round(bp * geo, 4)
    return scores

def compression_ratio(reference: str, hypothesis: str) -> float:
    rw = len(reference.split())
    hw = len(hypothesis.split())
    return round(hw / rw, 4) if rw else 0.0


def evaluate_summaries(pairs: dict, precomputed: dict) -> dict:
    results = {}
    for split, pair_list in pairs.items():
        if not pair_list:
            results[split] = {"note": "No data found — skipped."}
            continue

        pre = precomputed.get(split, [])
        has_reference = any(ref for ref, _ in pair_list)

        # ── Use pre-computed ROUGE where available (Week 3 already ran rouge_score) ─
        if pre and pre[0].get("rougeL") is not None:
            valid = [p for p in pre if p.get("rougeL") is not None]
            n = len(valid)
            rouge_result = {
                "rouge1": round(sum(p["rouge1"] for p in valid) / n, 4),
                "rouge2": round(sum(p["rouge2"] for p in valid) / n, 4) if valid[0].get("rouge2") is not None else None,
                "rougeL": round(sum(p["rougeL"] for p in valid) / n, 4),
                "source": "pre-computed (rouge_score library)",
            }
        elif has_reference:
            rl_scores = [rouge_l(r, h) for r, h in pair_list if r]
            n = len(rl_scores)
            rouge_result = {
                "rougeL_f1": round(sum(s["f1"] for s in rl_scores) / n, 4),
                "source": "computed (lightweight LCS)",
            }
        else:
            rouge_result = {"note": "No reference text — ROUGE skipped for clustered summaries."}

        # ── BLEU (only if reference available) ───────────────────────────────
        if has_reference:
            bleu_scores = [bleu(r, h) for r, h in pair_list if r]
            n_b = len(bleu_scores)
            bleu_result = {k: round(sum(s[k] for s in bleu_scores) / n_b, 4) for k in bleu_scores[0]}
        else:
            bleu_result = {"note": "No reference — BLEU skipped."}

        # ── Compression ratio ─────────────────────────────────────────────────
        if pre and pre[0].get("compression_ratio") is not None:
            crs = [p["compression_ratio"] for p in pre if p.get("compression_ratio") is not None]
            comp_result = {"mean": round(sum(crs)/len(crs), 4), "min": round(min(crs), 4),
                           "max": round(max(crs), 4), "source": "pre-computed"}
        elif has_reference:
            crs = [compression_ratio(r, h) for r, h in pair_list if r]
            comp_result = {"mean": round(sum(crs)/len(crs), 4), "min": round(min(crs), 4),
                           "max": round(max(crs), 4)}
        else:
            hyp_lens = [len(h.split()) for _, h in pair_list]
            comp_result = {"avg_summary_words": round(sum(hyp_lens)/len(hyp_lens), 1),
                           "note": "No reference — showing avg word count only."}

        results[split] = {
            "num_entries": len(pair_list),
            "rouge": rouge_result,
            "bleu":  bleu_result,
            "compression": comp_result,
        }

        rl_display = rouge_result.get("rougeL") or rouge_result.get("rougeL_f1", "N/A")
        print(f"  [{split}] {len(pair_list)} entries | ROUGE-L={rl_display}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 2.  GRAPH STRUCTURAL METRICS
# ══════════════════════════════════════════════════════════════════════════════

# def load_graph():
    try:
        import networkx as nx
    except ImportError:
        print("  [error] networkx not installed.")
        return None

    candidates = (glob.glob("week9_outputs/*.json")   +
                  glob.glob("week9_outputs/*.graphml") +
                  glob.glob("week8_outputs/*.json")   +
                  glob.glob("week8_outputs/*.graphml"))

    for path in candidates:
        try:
            if path.endswith(".graphml"):
                G = nx.read_graphml(path)
                print(f"  Loaded GraphML: {path}  ({G.number_of_nodes()}N, {G.number_of_edges()}E)")
                return G
            else:
                with open(path) as f:
                    data = json.load(f)
                if "nodes" in data and "links" in data:
                    G = nx.node_link_graph(data)
                    print(f"  Loaded node-link JSON: {path}  ({G.number_of_nodes()}N, {G.number_of_edges()}E)")
                    return G
        except Exception as e:
            print(f"  [warn] {path}: {e}")

    print("  [warn] No graph found in week8_outputs/ or week9_outputs/.")
    return None

def load_graph():
    import networkx as nx
    candidates = ["week8_outputs/enhanced_graph.json"]  # canonical output only
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            if "nodes" in data and "links" in data:
                G = nx.node_link_graph(data)
                print(f"  Loaded node-link JSON: {path}  ({G.number_of_nodes()}N, {G.number_of_edges()}E)")
                return G
        except Exception as e:
            print(f"  [warn] {path}: {e}")
    print("  [warn] No graph found.")
    return None

def load_communities():
    candidates = (glob.glob("week9_outputs/*communit*.json") +
                  glob.glob("week9_outputs/*partition*.json") +
                  glob.glob("week9_outputs/*.json"))
    for path in candidates:
        try:
            with open(path) as f:
                data = json.load(f)
            if "partition" in data:
                return data["partition"], data.get("algorithm", "unknown"), data.get("modularity")
        except Exception:
            pass
    return None, None, None


def _simple_undirected(G):
    """Convert any graph type to a plain undirected Graph."""
    import networkx as nx
    if G.is_multigraph():
        return nx.Graph(G.to_undirected())
    return G.to_undirected() if G.is_directed() else G


def graph_structural_metrics(G) -> dict:
    import networkx as nx

    n = G.number_of_nodes()
    e = G.number_of_edges()
    if n == 0:
        return {"note": "Empty graph."}

    Gu = _simple_undirected(G)
    density = nx.density(Gu)
    degrees = [d for _, d in Gu.degree()]
    avg_degree = sum(degrees) / n
    avg_clustering = nx.average_clustering(Gu)

    cc = list(nx.connected_components(Gu))
    num_components = len(cc)
    largest_cc_size = max(len(c) for c in cc)

    diameter = None
    try:
        lcc = Gu.subgraph(max(cc, key=len)).copy()
        if lcc.number_of_nodes() <= 2000:
            diameter = nx.diameter(lcc)
    except Exception:
        pass

    return {
        "num_nodes": n,
        "num_edges": e,
        "density": round(density, 6),
        "avg_degree": round(avg_degree, 4),
        "max_degree": max(degrees),
        "min_degree": min(degrees),
        "avg_clustering_coefficient": round(avg_clustering, 4),
        "num_connected_components": num_components,
        "largest_component_size": largest_cc_size,
        "diameter_largest_component": diameter,
    }


def community_metrics(G, partition, algorithm, saved_modularity) -> dict:
    import networkx as nx
    from networkx.algorithms.community.quality import modularity as nx_modularity

    if partition is None:
        return {"note": "No community partition found — skipped."}

    Gu = _simple_undirected(G)
    comm_map = {}
    for node, cid in partition.items():
        comm_map.setdefault(cid, set()).add(node)

    communities = [c & set(Gu.nodes()) for c in comm_map.values()]
    communities = [c for c in communities if c]
    sizes = sorted([len(c) for c in communities], reverse=True)

    try:
        mod = nx_modularity(Gu, communities)
    except Exception:
        mod = saved_modularity

    try:
        from networkx.algorithms.community.quality import coverage, performance
        cov  = coverage(Gu, communities)
        perf = performance(Gu, communities)
    except Exception:
        cov, perf = None, None

    return {
        "algorithm": algorithm,
        "num_communities": len(communities),
        "modularity": round(mod, 4) if mod is not None else None,
        "coverage":   round(cov,  4) if cov  is not None else None,
        "performance": round(perf, 4) if perf is not None else None,
        "community_sizes": {
            "min": min(sizes), "max": max(sizes),
            "mean": round(sum(sizes)/len(sizes), 2),
            "top5": sizes[:5],
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    report = {}

    print("\n[1/2] Evaluating summaries …")
    pairs, precomputed = load_summaries()
    summary_results = evaluate_summaries(pairs, precomputed)
    report["summary_evaluation"] = summary_results

    print("\n[2/2] Computing graph metrics …")
    G = load_graph()
    if G is not None:
        struct = graph_structural_metrics(G)
        report["graph_structural_metrics"] = struct
        print(f"  density={struct.get('density')}  avg_degree={struct.get('avg_degree')}  "
              f"avg_clustering={struct.get('avg_clustering_coefficient')}")

        partition, algorithm, saved_mod = load_communities()
        comm = community_metrics(G, partition, algorithm, saved_mod)
        report["community_detection_metrics"] = comm
        if "num_communities" in comm:
            print(f"  communities={comm['num_communities']}  modularity={comm['modularity']}  "
                  f"algorithm={comm['algorithm']}")
    else:
        report["graph_structural_metrics"]    = {"note": "Graph could not be loaded."}
        report["community_detection_metrics"] = {"note": "Graph could not be loaded."}

    out_path = os.path.join(OUTPUT_DIR, "evaluation_results.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n✅  Results saved → {out_path}")

    # ── Print summary table ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print("WEEK 12 EVALUATION SUMMARY")
    print("="*60)

    for split, res in summary_results.items():
        if "note" in res:
            print(f"\n  {split.upper()}: {res['note']}")
            continue
        print(f"\n  {split.upper()} ({res['num_entries']} entries)")
        rl = res['rouge'].get('rougeL') or res['rouge'].get('rougeL_f1', 'N/A')
        r1 = res['rouge'].get('rouge1', 'N/A')
        r2 = res['rouge'].get('rouge2', 'N/A')
        print(f"    ROUGE-1 / 2 / L : {r1} / {r2} / {rl}")
        print(f"    BLEU (avg)      : {res['bleu'].get('bleu_avg', res['bleu'].get('note','N/A'))}")
        cm = res['compression']
        print(f"    Compression     : {cm.get('mean', cm.get('avg_summary_words','N/A'))}")

    sm = report.get("graph_structural_metrics", {})
    if "num_nodes" in sm:
        print(f"\n  GRAPH STRUCTURE")
        print(f"    Nodes / Edges        : {sm['num_nodes']} / {sm['num_edges']}")
        print(f"    Density              : {sm['density']}")
        print(f"    Avg degree           : {sm['avg_degree']}")
        print(f"    Avg clustering coeff : {sm['avg_clustering_coefficient']}")
        print(f"    Connected components : {sm['num_connected_components']}")
        if sm.get("diameter_largest_component") is not None:
            print(f"    Diameter (LCC)       : {sm['diameter_largest_component']}")

    cm = report.get("community_detection_metrics", {})
    if "num_communities" in cm:
        print(f"\n  COMMUNITY DETECTION ({cm['algorithm']})")
        print(f"    Communities  : {cm['num_communities']}")
        print(f"    Modularity   : {cm['modularity']}")
        if cm.get("coverage")    is not None: print(f"    Coverage     : {cm['coverage']}")
        if cm.get("performance") is not None: print(f"    Performance  : {cm['performance']}")
        print(f"    Top-5 sizes  : {cm['community_sizes']['top5']}")

    print("="*60)


if __name__ == "__main__":
    main()