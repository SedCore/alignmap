"""Command line: ``alignmap fetch | download | run | compare | demo``."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

from .ard import DEFAULT_SOURCES, KNOWN_SOURCES, download_ard
from .corpus import build_corpus, collect_inputs, iter_records, write_jsonl
from .compare import compare_arms, write_compare
from .fetch import CORE_TERMS, EXTENDED_TERMS, fetch_years
from .pipeline import RunConfig, run_pipeline
from .presets import PRESETS, apply_preset
from .toy import make_composition_toy, make_toy_records
from .trends import parse_year_range


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _add_run_options(p: argparse.ArgumentParser) -> None:
    """Every flag defaults to None so that a flag given on the command line overrides a preset."""
    p.add_argument("--preset", choices=list(PRESETS), help="start from a named configuration")
    p.add_argument("--sources", nargs="+", help="keep only these `source` values (e.g. alignmentforum arxiv)")
    p.add_argument("--embedder", choices=["tfidf", "sbert"])
    p.add_argument("--sbert-model", help="sentence-transformers model for --embedder sbert")
    p.add_argument("--max-seq-length", type=int, help="truncate inputs to this many tokens (sbert)")
    p.add_argument("--dims", type=int, help="LSA dimensions for the tfidf embedder")
    p.add_argument("--reducer", choices=["none", "umap"])
    p.add_argument("--umap-components", type=int)
    p.add_argument("--umap-neighbors", type=int)
    p.add_argument("--clusterer", choices=["kmeans", "hdbscan"])
    p.add_argument("--k", help="k-means clusters, or 'auto' to pick by silhouette over --k-range")
    p.add_argument("--k-range", nargs=3, type=int, metavar=("START", "STOP", "STEP"))
    p.add_argument("--min-cluster-size", type=float, help="HDBSCAN; < 1 is a fraction of the documents")
    p.add_argument("--baseline", help="baseline years, e.g. 2021-2022")
    p.add_argument("--recent", help="recent years, e.g. 2025-2026")
    p.add_argument("--min-group-docs", type=int, help="min papers per source group and period for the adjustment")
    p.add_argument("--seed", type=int)
    p.add_argument("--n-boot", type=int)
    p.add_argument("--tag-top", type=int, help="number of community tags used as reference labels")
    p.add_argument("--tag-exclude", nargs="+", help="tags to ignore as too generic (replaces the default list)")
    p.add_argument("--llm-labels", action="store_true",
                   help="name clusters with Claude (needs alignmap[llm] and ANTHROPIC_API_KEY)")
    p.add_argument("--llm-model")
    p.add_argument("--cache-dir", help="cache embeddings here so several runs can share them")


def _config(args, **forced) -> RunConfig:
    cfg = RunConfig()
    if getattr(args, "preset", None):
        cfg = apply_preset(cfg, args.preset)
    updates: dict = {}
    for name in ("embedder", "sbert_model", "max_seq_length", "dims", "reducer", "umap_components",
                 "umap_neighbors", "clusterer", "min_cluster_size", "min_group_docs", "seed", "n_boot",
                 "tag_top", "llm_model"):
        value = getattr(args, name, None)
        if value is not None:
            updates[name] = value
    if getattr(args, "k", None) is not None:
        updates["k"] = None if str(args.k).lower() == "auto" else int(args.k)
    if getattr(args, "k_range", None):
        updates["k_range"] = tuple(args.k_range)
    if getattr(args, "baseline", None):
        updates["baseline"] = parse_year_range(args.baseline)
    if getattr(args, "recent", None):
        updates["recent"] = parse_year_range(args.recent)
    if getattr(args, "sources", None):
        updates["sources"] = tuple(args.sources)
    if getattr(args, "tag_exclude", None):
        updates["tag_exclude"] = tuple(args.tag_exclude)
    if getattr(args, "llm_labels", False):
        updates["llm_labels"] = True
    return replace(cfg, **updates, **forced)


def _cmd_fetch(args) -> int:
    lo, hi = parse_year_range(args.years)
    if args.terms_file:
        terms = [t.strip() for t in Path(args.terms_file).read_text(encoding="utf-8").splitlines() if t.strip()]
    else:
        terms = CORE_TERMS if args.terms == "core" else EXTENDED_TERMS
    paths = fetch_years(
        range(lo, hi + 1), args.out, terms,
        max_per_year=args.max_per_year, refresh=args.refresh, delay=args.delay,
    )
    print(f"\nDone. {len(paths)} files in {args.out}. Next: alignmap run --input {args.out}")
    return 0


def _cmd_download(args) -> int:
    paths = download_ard(args.sources, args.out, refresh=args.refresh)
    print(f"\nDone. {len(paths)} files in {args.out}. Next: alignmap run --input {args.out}")
    return 0


def _provenance(files) -> dict:
    return {"inputs": [{"path": str(f), "sha256": _sha256(f)} for f in files]}


def _cmd_run(args) -> int:
    files = collect_inputs(args.input)
    cfg = _config(args)
    result = run_pipeline(
        iter_records(files), args.out, cfg, provenance=_provenance(files), cache_dir=args.cache_dir
    )
    n_clusters = result["summary"]["n_clusters"]
    print(f"Wrote {Path(args.out) / 'report.md'}  ({len(result['df']):,} papers, {n_clusters} clusters)")
    return 0


def _cmd_compare(args) -> int:
    files = collect_inputs(args.input)
    base = _config(args)
    if args.configs:
        spec = json.loads(Path(args.configs).read_text(encoding="utf-8"))
        arms = {name: replace(base, **over) for name, over in spec.items()}
    else:
        arms = {name: apply_preset(base, name) for name in args.presets}
    df, stats = build_corpus(
        iter_records(files), sources=base.sources, min_chars=base.min_chars, max_chars=base.max_chars
    )
    table, used = compare_arms(df, arms, cache_dir=args.cache_dir, n_rounds=args.stability_rounds)
    out = Path(args.out)
    write_compare(table, used, out / "compare.md", n_docs=len(df), sources=stats["by_source"])
    print(f"Wrote {out / 'compare.md'}")
    return 0


def _cmd_demo(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = (
        make_composition_toy(seed=args.seed or 0) if args.kind == "composition" else make_toy_records(seed=args.seed or 0)
    )
    write_jsonl(records, out / "toy_corpus.jsonl")
    cfg = _config(args, k=4, n_boot=2000)
    run_pipeline(records, out, cfg, provenance={"inputs": ["synthetic (alignmap.toy)"]})
    print(f"Synthetic demo written to {out / 'report.md'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alignmap", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fetch", help="download alignment-related papers from the arXiv API")
    f.add_argument("--years", default="2021-2026", help="e.g. 2021-2026")
    f.add_argument("--terms", choices=["core", "extended"], default="core")
    f.add_argument("--terms-file", help="text file with one search phrase per line (overrides --terms)")
    f.add_argument("--max-per-year", type=int, default=3000)
    f.add_argument("--delay", type=float, default=3.1, help="seconds between requests (arXiv asks for >= 3)")
    f.add_argument("--refresh", action="store_true", help="ignore cached years")
    f.add_argument("--out", default="data/raw")
    f.set_defaults(func=_cmd_fetch)

    d = sub.add_parser("download", help="download the AI Alignment Research Dataset (one JSONL per source)")
    d.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES), choices=list(KNOWN_SOURCES))
    d.add_argument("--out", default="data/ard")
    d.add_argument("--refresh", action="store_true")
    d.set_defaults(func=_cmd_download)

    r = sub.add_parser("run", help="cluster a corpus and report subfield shifts")
    r.add_argument("--input", nargs="+", required=True, help="JSONL files, directories or glob patterns")
    r.add_argument("--out", default="results/run")
    _add_run_options(r)
    r.set_defaults(func=_cmd_run)

    c = sub.add_parser("compare", help="compare methods on the same corpus, scored against community tags")
    c.add_argument("--input", nargs="+", required=True)
    c.add_argument("--out", default="results/compare")
    c.add_argument("--presets", nargs="+", choices=list(PRESETS), default=["lsa-baseline", "2022-style", "modern-fast"])
    c.add_argument("--configs", help="JSON file {arm name: {RunConfig overrides}} instead of --presets")
    c.add_argument("--stability-rounds", type=int, default=3)
    _add_run_options(c)
    c.set_defaults(func=_cmd_compare)

    m = sub.add_parser("demo", help="run the full pipeline on a synthetic corpus (offline)")
    m.add_argument("--out", default="results/demo")
    m.add_argument("--kind", choices=["composition", "trend"], default="composition",
                   help="'composition': venue mix drifts while topics inside each venue barely change")
    _add_run_options(m)
    m.set_defaults(func=_cmd_demo)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, RuntimeError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
