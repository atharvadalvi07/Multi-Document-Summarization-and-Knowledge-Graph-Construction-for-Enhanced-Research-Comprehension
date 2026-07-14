import argparse
import sys
import os
import shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
for week in [f"week{i}" for i in range(1, 13)]:
    sys.path.insert(0, os.path.join(ROOT, week))

DATA_DIR    = os.path.join(ROOT, "data")
OUTPUTS_DIR = os.path.join(ROOT, "outputs")

# Week-specific output subdirectories
W6_DIR  = os.path.join(OUTPUTS_DIR, "week6_outputs")
W7_DIR  = os.path.join(OUTPUTS_DIR, "week7_outputs")
W8_DIR  = os.path.join(OUTPUTS_DIR, "week8_outputs")
W9_DIR  = os.path.join(OUTPUTS_DIR, "week9_outputs")
W10_DIR = os.path.join(OUTPUTS_DIR, "week10_outputs")
W11_DIR = os.path.join(OUTPUTS_DIR, "week11_outputs")
W12_DIR = os.path.join(OUTPUTS_DIR, "week12_outputs")

# Final package: only the files needed to verify/support the manuscript
FINAL_DIR = os.path.join(OUTPUTS_DIR, "final_deliverables")


def run_week1():
    print("\n" + "=" * 60)
    print("WEEK 1: Fetching papers from arXiv")
    print("=" * 60)
    import fetch_papers
    fetch_papers.main()


def run_week2():
    print("\n" + "=" * 60)
    print("WEEK 2: Preprocessing dataset")
    print("=" * 60)
    import preprocess
    raw_path    = os.path.join(DATA_DIR, "raw_papers.json")
    output_path = os.path.join(DATA_DIR, "cleaned_dataset.json")
    report_path = os.path.join(DATA_DIR, "dataset_stats.txt")
    cleaned = preprocess.preprocess(raw_path, output_path)
    preprocess.generate_stats_report(cleaned, report_path)


def run_week3():
    print("\n" + "=" * 60)
    print("WEEK 3: Baseline Transformer Summarization + ROUGE")
    print("=" * 60)
    import baseline_summarization
    cleaned_path = os.path.join(DATA_DIR, "cleaned_dataset.json")
    output_path  = os.path.join(OUTPUTS_DIR, "baseline_summaries.json")
    baseline_summarization.run_baseline_evaluation(cleaned_path, output_path, n_papers=10)


def run_week4():
    print("\n" + "=" * 60)
    print("WEEK 4: Clustering + Joint Summarization")
    print("=" * 60)
    import clustering_summarization
    cleaned_path = os.path.join(DATA_DIR, "cleaned_dataset.json")
    output_path  = os.path.join(OUTPUTS_DIR, "clustering_results.json")
    clustering_summarization.run_clustering_evaluation(cleaned_path, output_path, n_clusters=4)


def run_week5():
    print("\n" + "=" * 60)
    print("WEEK 5: SciSpacy NER Pipeline")
    print("=" * 60)
    import ner_pipeline
    cleaned_path     = os.path.join(DATA_DIR, "cleaned_dataset.json")
    output_path      = os.path.join(OUTPUTS_DIR, "ner_results.json")
    global_freq_path = os.path.join(OUTPUTS_DIR, "entity_frequencies.json")
    ner_pipeline.run_ner_pipeline(cleaned_path, output_path, global_freq_path, n_papers=15)


def run_week6():
    print("\n" + "=" * 60)
    print("WEEK 6: Entity Normalization & Deduplication")
    print("=" * 60)
    import entity_normalization
    os.makedirs(W6_DIR, exist_ok=True)
    ner_path    = os.path.join(OUTPUTS_DIR, "ner_results.json")
    output_path = os.path.join(W6_DIR, "canonical_entities.json")
    entity_normalization.run_normalization(ner_path, output_path)


def run_week7():
    print("\n" + "=" * 60)
    print("WEEK 7: Initial NetworkX Knowledge Graph")
    print("=" * 60)
    import networkx_graph
    os.makedirs(W7_DIR, exist_ok=True)
    entities_path = os.path.join(W6_DIR, "canonical_entities.json")
    graph_pkl     = os.path.join(W7_DIR, "knowledge_graph.pkl")
    metrics_path  = os.path.join(W7_DIR, "metrics_report.json")
    viz_path      = os.path.join(W7_DIR, "week7_graph.png")
    networkx_graph.run_graph_construction(entities_path, graph_pkl, metrics_path, viz_path)


def run_week8():
    print("\n" + "=" * 60)
    print("WEEK 8: Enhanced Edge Construction")
    print("=" * 60)
    import edge_construction
    os.makedirs(W8_DIR, exist_ok=True)
    entities_path = os.path.join(W6_DIR, "canonical_entities.json")
    cleaned_path  = os.path.join(DATA_DIR, "cleaned_dataset.json")
    graph_json    = os.path.join(W8_DIR, "enhanced_graph.json")
    graph_graphml = os.path.join(W8_DIR, "enhanced_graph.graphml")
    metrics_csv   = os.path.join(W8_DIR, "metrics_report.csv")
    viz_path      = os.path.join(W8_DIR, "week8_graph.png")
    edge_construction.run_week8_pipeline(
        entities_path, cleaned_path,
        graph_json, graph_graphml, metrics_csv, viz_path,
    )


def run_week9():
    print("\n" + "=" * 60)
    print("WEEK 9: Community Detection & Graph Analytics")
    print("=" * 60)
    import community_detection
    os.makedirs(W9_DIR, exist_ok=True)
    graph_json = os.path.join(W8_DIR, "enhanced_graph.json")
    community_detection.INPUT_PATH = graph_json
    community_detection.OUTPUT_DIR = W9_DIR
    community_detection.main()


def run_week10():
    print("\n" + "=" * 60)
    print("WEEK 10: Interactive Knowledge Graph Visualization (PyVis)")
    print("=" * 60)
    import visualization
    os.makedirs(W10_DIR, exist_ok=True)
    # visualization.py reads/writes via module-level globals; override before calling main()
    visualization.GRAPH_INPUT_PATH = os.path.join(W7_DIR, "knowledge_graph.pkl")
    visualization.HTML_OUTPUT_PATH = os.path.join(W10_DIR, "knowledge_graph.html")
    visualization.META_OUTPUT_PATH = os.path.join(W10_DIR, "visualization_meta.json")
    visualization.main()


def run_week11():
    print("\n" + "=" * 60)
    print("WEEK 11: Filtering & Layer Controls Visualization")
    print("=" * 60)
    import customize
    from pathlib import Path
    os.makedirs(W11_DIR, exist_ok=True)
    # customize.py's main() hardcodes "outputs/week8_outputs/..." and
    # "outputs/week9_outputs/..." as *relative* paths, so it depends on cwd == ROOT.
    _cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        customize.OUTPUT_DIR = Path(W11_DIR)
        customize.main()
    finally:
        os.chdir(_cwd)


def _ensure_eval_compat_symlinks():
    """
    eval.py globs 'outputs/baseline_summaries.json' (prefixed) but also
    'week8_outputs/*.json' / 'week9_outputs/*.json' (NOT prefixed with 'outputs/').
    Both can't resolve from the same cwd given our outputs/week*_outputs layout,
    which is the likely source of the edge-count discrepancy flagged for Week 16.
    We bridge this without touching eval.py by symlinking the week dirs at ROOT.
    """
    links = {
        os.path.join(ROOT, "week8_outputs"): W8_DIR,
        os.path.join(ROOT, "week9_outputs"): W9_DIR,
    }
    created = []
    for link_path, target in links.items():
        if os.path.exists(target) and not os.path.exists(link_path):
            try:
                os.symlink(target, link_path, target_is_directory=True)
                created.append(link_path)
            except OSError as e:
                print(f"[WARN] Could not create compat symlink {link_path} -> {target}: {e}")
    return created


def run_week12():
    print("\n" + "=" * 60)
    print("WEEK 12: Evaluation Experiments")
    print("=" * 60)
    import eval as eval_module
    os.makedirs(W12_DIR, exist_ok=True)

    _cwd = os.getcwd()
    os.chdir(ROOT)
    compat_links = _ensure_eval_compat_symlinks()
    try:
        eval_module.OUTPUT_DIR = W12_DIR
        eval_module.main()
    finally:
        for link_path in compat_links:
            try:
                os.remove(link_path)
            except OSError:
                pass
        os.chdir(_cwd)


WEEK_RUNNERS = {
    1: run_week1,
    2: run_week2,
    3: run_week3,
    4: run_week4,
    5: run_week5,
    6: run_week6,
    7: run_week7,
    8: run_week8,
    9: run_week9,
    10: run_week10,
    11: run_week11,
    12: run_week12,
}

WEEK_DESCRIPTIONS = {
    1: "Fetch papers from arXiv",
    2: "Preprocess dataset",
    3: "Baseline summarization + ROUGE",
    4: "Clustering + joint summarization",
    5: "SciSpacy NER pipeline",
    6: "Entity normalization & deduplication",
    7: "Initial NetworkX knowledge graph",
    8: "Enhanced edge construction (dep + PMI)",
    9: "Community detection & graph analytics",
    10: "Interactive visualization (PyVis)",
    11: "Filtering & layer-controls visualization",
    12: "Evaluation experiments (summary + graph metrics)",
}

# ══════════════════════════════════════════════════════════════════════════
# FINAL DELIVERABLES PACKAGING
# Only the files a reviewer / the manuscript actually needs, copied into one
# flat folder with week-prefixed names so nothing collides or gets missed.
# ══════════════════════════════════════════════════════════════════════════

RELEVANT_FILES = [
    # (source path relative to ROOT, new name in FINAL_DIR)
    (os.path.join(DATA_DIR, "cleaned_dataset.json"),            "week02_cleaned_dataset.json"),
    (os.path.join(DATA_DIR, "dataset_stats.txt"),                "week02_dataset_stats.txt"),
    (os.path.join(OUTPUTS_DIR, "baseline_summaries.json"),       "week03_baseline_summaries.json"),
    (os.path.join(OUTPUTS_DIR, "clustering_results.json"),       "week04_clustering_results.json"),
    (os.path.join(OUTPUTS_DIR, "entity_frequencies.json"),       "week05_entity_frequencies.json"),
    (os.path.join(W6_DIR, "canonical_entities.json"),            "week06_canonical_entities.json"),
    (os.path.join(W6_DIR, "canonical_entities.csv"),             "week06_canonical_entities.csv"),
    (os.path.join(W7_DIR, "knowledge_graph.pkl"),                "week07_knowledge_graph.pkl"),
    (os.path.join(W7_DIR, "metrics_report.json"),                "week07_metrics_report.json"),
    (os.path.join(W7_DIR, "week7_graph.png"),                    "week07_graph_preview.png"),
    (os.path.join(W8_DIR, "enhanced_graph.json"),                "week08_enhanced_graph.json"),
    (os.path.join(W8_DIR, "enhanced_graph.graphml"),              "week08_enhanced_graph.graphml"),
    (os.path.join(W8_DIR, "metrics_report.csv"),                 "week08_metrics_report.csv"),
    (os.path.join(W8_DIR, "week8_graph.png"),                    "week08_graph_preview.png"),
    (os.path.join(W9_DIR, "community_assignments.json"),         "week09_community_assignments.json"),
    (os.path.join(W9_DIR, "analytics_report.json"),              "week09_analytics_report.json"),
    (os.path.join(W9_DIR, "analytics_report.csv"),                "week09_analytics_report.csv"),
    (os.path.join(W9_DIR, "week9_graph.png"),                    "week09_graph_preview.png"),
    (os.path.join(W10_DIR, "knowledge_graph.html"),               "week10_knowledge_graph_interactive.html"),
    (os.path.join(W10_DIR, "visualization_meta.json"),            "week10_visualization_meta.json"),
    (os.path.join(W11_DIR, "knowledge_graph_filtered.html"),      "week11_knowledge_graph_filtered.html"),
    (os.path.join(W11_DIR, "viz_manifest.json"),                  "week11_viz_manifest.json"),
    (os.path.join(W12_DIR, "evaluation_results.json"),            "week12_evaluation_results.json"),
]


def package_deliverables():
    print("\n" + "=" * 60)
    print("PACKAGING FINAL DELIVERABLES")
    print("=" * 60)
    os.makedirs(FINAL_DIR, exist_ok=True)

    copied, missing = [], []
    for src, dst_name in RELEVANT_FILES:
        dst = os.path.join(FINAL_DIR, dst_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied.append(dst_name)
        else:
            missing.append(src)

    manifest_path = os.path.join(FINAL_DIR, "MANIFEST.txt")
    with open(manifest_path, "w") as f:
        f.write("Final Deliverables Manifest\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Included ({len(copied)} files):\n")
        for name in copied:
            f.write(f"  - {name}\n")
        if missing:
            f.write(f"\nMissing / not yet generated ({len(missing)} files):\n")
            for path in missing:
                f.write(f"  - {path}\n")

    print(f"  Copied {len(copied)} files → {FINAL_DIR}/")
    if missing:
        print(f"  [WARN] {len(missing)} expected file(s) not found (see MANIFEST.txt):")
        for path in missing:
            print(f"    - {path}")
    print(f"  Manifest written → {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Research Pipeline: Weeks 1-12",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="\n".join(
            f"  Week {w}: {d}" for w, d in WEEK_DESCRIPTIONS.items()
        ),
    )
    parser.add_argument(
        "--weeks", nargs="+", type=int, default=list(range(1, 13)),
        help="Which weeks to run (default: all 1-12)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print available weeks and exit",
    )
    parser.add_argument(
        "--package-only", action="store_true",
        help="Skip pipeline execution; just (re)package final_deliverables from existing outputs",
    )
    parser.add_argument(
        "--no-package", action="store_true",
        help="Run selected weeks but skip the final_deliverables packaging step",
    )
    args = parser.parse_args()

    if args.list:
        print("Available weeks:")
        for w, d in WEEK_DESCRIPTIONS.items():
            print(f"  {w}: {d}")
        return

    os.makedirs(DATA_DIR,    exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    if args.package_only:
        package_deliverables()
        return

    selected = sorted(set(args.weeks))
    unknown  = [w for w in selected if w not in WEEK_RUNNERS]
    if unknown:
        print(f"[WARN] Unknown week(s) ignored: {unknown}")

    for week_num in selected:
        if week_num not in WEEK_RUNNERS:
            continue
        try:
            WEEK_RUNNERS[week_num]()
        except Exception as exc:
            print(f"\n[ERROR] Week {week_num} failed: {exc}")
            raise

    print("\nAll selected weeks complete.")
    print(f"   Data    → {DATA_DIR}/")
    print(f"   Outputs → {OUTPUTS_DIR}/")

    if not args.no_package:
        package_deliverables()


if __name__ == "__main__":
    main()
    
# import argparse
# import sys
# import os

# ROOT = os.path.dirname(__file__)
# for week in ["week1", "week2", "week3", "week4", "week5", "week6", "week7", "week8", "week9"]:
#     sys.path.insert(0, os.path.join(ROOT, week))

# DATA_DIR    = os.path.join(ROOT, "data")
# OUTPUTS_DIR = os.path.join(ROOT, "outputs")

# # Week-specific output subdirectories
# W6_DIR = os.path.join(OUTPUTS_DIR, "week6_outputs")
# W7_DIR = os.path.join(OUTPUTS_DIR, "week7_outputs")
# W8_DIR = os.path.join(OUTPUTS_DIR, "week8_outputs")
# W9_DIR = os.path.join(OUTPUTS_DIR, "week9_outputs")


# def run_week1():
#     print("\n" + "=" * 60)
#     print("WEEK 1: Fetching papers from arXiv")
#     print("=" * 60)
#     import fetch_papers
#     fetch_papers.main()


# def run_week2():
#     print("\n" + "=" * 60)
#     print("WEEK 2: Preprocessing dataset")
#     print("=" * 60)
#     import preprocess
#     raw_path    = os.path.join(DATA_DIR, "raw_papers.json")
#     output_path = os.path.join(DATA_DIR, "cleaned_dataset.json")
#     report_path = os.path.join(DATA_DIR, "dataset_stats.txt")
#     cleaned = preprocess.preprocess(raw_path, output_path)
#     preprocess.generate_stats_report(cleaned, report_path)


# def run_week3():
#     print("\n" + "=" * 60)
#     print("WEEK 3: Baseline Transformer Summarization + ROUGE")
#     print("=" * 60)
#     import baseline_summarization
#     cleaned_path = os.path.join(DATA_DIR, "cleaned_dataset.json")
#     output_path  = os.path.join(OUTPUTS_DIR, "baseline_summaries.json")
#     baseline_summarization.run_baseline_evaluation(cleaned_path, output_path, n_papers=10)


# def run_week4():
#     print("\n" + "=" * 60)
#     print("WEEK 4: Clustering + Joint Summarization")
#     print("=" * 60)
#     import clustering_summarization
#     cleaned_path = os.path.join(DATA_DIR, "cleaned_dataset.json")
#     output_path  = os.path.join(OUTPUTS_DIR, "clustering_results.json")
#     clustering_summarization.run_clustering_evaluation(cleaned_path, output_path, n_clusters=4)


# def run_week5():
#     print("\n" + "=" * 60)
#     print("WEEK 5: SciSpacy NER Pipeline")
#     print("=" * 60)
#     import ner_pipeline
#     cleaned_path    = os.path.join(DATA_DIR, "cleaned_dataset.json")
#     output_path     = os.path.join(OUTPUTS_DIR, "ner_results.json")
#     global_freq_path = os.path.join(OUTPUTS_DIR, "entity_frequencies.json")
#     ner_pipeline.run_ner_pipeline(cleaned_path, output_path, global_freq_path, n_papers=15)


# def run_week6():
#     print("\n" + "=" * 60)
#     print("WEEK 6: Entity Normalization & Deduplication")
#     print("=" * 60)
#     import entity_normalization
#     os.makedirs(W6_DIR, exist_ok=True)
#     ner_path    = os.path.join(OUTPUTS_DIR, "ner_results.json")
#     output_path = os.path.join(W6_DIR, "canonical_entities.json")
#     entity_normalization.run_normalization(ner_path, output_path)


# def run_week7():
#     print("\n" + "=" * 60)
#     print("WEEK 7: Initial NetworkX Knowledge Graph")
#     print("=" * 60)
#     import networkx_graph
#     os.makedirs(W7_DIR, exist_ok=True)
#     entities_path = os.path.join(W6_DIR, "canonical_entities.json")
#     graph_pkl     = os.path.join(W7_DIR, "knowledge_graph.pkl")
#     metrics_path  = os.path.join(W7_DIR, "metrics_report.json")
#     viz_path      = os.path.join(W7_DIR, "week7_graph.png")
#     networkx_graph.run_graph_construction(entities_path, graph_pkl, metrics_path, viz_path)


# def run_week8():
#     print("\n" + "=" * 60)
#     print("WEEK 8: Enhanced Edge Construction")
#     print("=" * 60)
#     import edge_construction
#     os.makedirs(W8_DIR, exist_ok=True)
#     entities_path   = os.path.join(W6_DIR, "canonical_entities.json")
#     cleaned_path    = os.path.join(DATA_DIR, "cleaned_dataset.json")
#     graph_json      = os.path.join(W8_DIR, "enhanced_graph.json")
#     graph_graphml   = os.path.join(W8_DIR, "enhanced_graph.graphml")
#     metrics_csv     = os.path.join(W8_DIR, "metrics_report.csv")
#     viz_path        = os.path.join(W8_DIR, "week8_graph.png")
#     edge_construction.run_week8_pipeline(
#         entities_path, cleaned_path,
#         graph_json, graph_graphml, metrics_csv, viz_path,
#     )


# def run_week9():
#     print("\n" + "=" * 60)
#     print("WEEK 9: Community Detection & Graph Analytics")
#     print("=" * 60)
#     import community_detection
#     os.makedirs(W9_DIR, exist_ok=True)
#     graph_json = os.path.join(W8_DIR, "enhanced_graph.json")
#     community_detection.INPUT_PATH  = graph_json
#     community_detection.OUTPUT_DIR  = W9_DIR
#     community_detection.main()


# WEEK_RUNNERS = {
#     1: run_week1,
#     2: run_week2,
#     3: run_week3,
#     4: run_week4,
#     5: run_week5,
#     6: run_week6,
#     7: run_week7,
#     8: run_week8,
#     9: run_week9,
# }

# WEEK_DESCRIPTIONS = {
#     1: "Fetch papers from arXiv",
#     2: "Preprocess dataset",
#     3: "Baseline summarization + ROUGE",
#     4: "Clustering + joint summarization",
#     5: "SciSpacy NER pipeline",
#     6: "Entity normalization & deduplication",
#     7: "Initial NetworkX knowledge graph",
#     8: "Enhanced edge construction (dep + PMI)",
#     9: "Community detection & graph analytics",
# }


# def main():
#     parser = argparse.ArgumentParser(
#         description="Research Pipeline: Weeks 1–9",
#         formatter_class=argparse.RawTextHelpFormatter,
#         epilog="\n".join(
#             f"  Week {w}: {d}" for w, d in WEEK_DESCRIPTIONS.items()
#         ),
#     )
#     parser.add_argument(
#         "--weeks", nargs="+", type=int, default=list(range(1, 10)),
#         help="Which weeks to run (default: all 1–9)",
#     )
#     parser.add_argument(
#         "--list", action="store_true",
#         help="Print available weeks and exit",
#     )
#     args = parser.parse_args()

#     if args.list:
#         print("Available weeks:")
#         for w, d in WEEK_DESCRIPTIONS.items():
#             print(f"  {w}: {d}")
#         return

#     os.makedirs(DATA_DIR,    exist_ok=True)
#     os.makedirs(OUTPUTS_DIR, exist_ok=True)

#     selected = sorted(set(args.weeks))
#     unknown  = [w for w in selected if w not in WEEK_RUNNERS]
#     if unknown:
#         print(f"[WARN] Unknown week(s) ignored: {unknown}")

#     for week_num in selected:
#         if week_num not in WEEK_RUNNERS:
#             continue
#         try:
#             WEEK_RUNNERS[week_num]()
#         except Exception as exc:
#             print(f"\n[ERROR] Week {week_num} failed: {exc}")
#             raise

#     print("\n All selected weeks complete.")
#     print(f"   Data    → {DATA_DIR}/")
#     print(f"   Outputs → {OUTPUTS_DIR}/")
#     if 9 in selected:
#         print(f"   W9      → {W9_DIR}/")


# if __name__ == "__main__":
#     main()
 