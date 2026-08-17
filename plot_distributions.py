"""
plot_distributions.py — Thesis figure: embedding value distributions per response.

Shows the core point: SBERT RAW embeddings give each response a bell curve that SEPARATES
(different peak/spread, driven by the vector's norm), while L2-NORMALIZED embeddings — what
OpenAI's text-embedding-3 returns — collapse to the same curve. In high dimensions every unit
vector's coordinates look like N(0, 1/d) regardless of content, so normalized responses are
indistinguishable. That is why the OpenAI approach could not tell responses apart; the norm it
throws away is exactly the signal these plots rely on.

Usage
  # A) from an exported full CSV (columns sbert_0..sbert_767, plus model/question/group):
  python plot_distributions.py --csv all_full.csv

  # B) from a CSV of raw responses (needs a 'response' column; optional 'group', 'model'):
  python plot_distributions.py --responses responses.csv

  # C) no data yet — run the built-in demo (embeds a few sentences and plots them):
  python plot_distributions.py --demo

Output: distributions.png (+ a printed peak / norm / std table).
"""

import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

MODEL_NAME = "all-mpnet-base-v2"   # matches embedding_extraction.py


# ── Loading embeddings from the various input shapes ──────────────────────────
def _embed(texts):
    """Compute genuinely RAW SBERT embeddings (Normalize layer dropped) for a list of texts."""
    from sentence_transformers import SentenceTransformer, models
    print(f"Loading {MODEL_NAME} (raw, no Normalize layer) …")
    # Stock all-mpnet-base-v2 ends in a Normalize module that forces unit length; rebuilding
    # from Transformer + mean Pooling keeps the true magnitude so the curves can separate.
    word = models.Transformer(f"sentence-transformers/{MODEL_NAME}")
    pool = models.Pooling(word.get_word_embedding_dimension(), pooling_mode="mean")
    model = SentenceTransformer(modules=[word, pool])
    return np.asarray(model.encode(list(texts), normalize_embeddings=False))


def _pick_label_column(df):
    """Prefer an explicit group; fall back to question, then a constant."""
    for col in ("group", "question", "query_index"):
        if col in df.columns:
            return df[col].astype(str).tolist()
    return ["response"] * len(df)


def load_data(args):
    """Return (X_raw, labels, models) where X_raw is (n_responses × dim)."""
    if args.csv:
        df = pd.read_csv(args.csv)
        sbert_cols = sorted(
            (c for c in df.columns if c.startswith("sbert_")),
            key=lambda c: int(c.split("_")[1]),
        )
        if not sbert_cols:
            sys.exit(f"No sbert_* columns found in {args.csv}. Use --responses instead.")
        X = df[sbert_cols].to_numpy(dtype=float)
        return X, _pick_label_column(df), df.get("model", pd.Series(range(len(df)))).astype(str).tolist()

    if args.responses:
        df = pd.read_csv(args.responses)
        if "response" not in df.columns:
            sys.exit(f"{args.responses} needs a 'response' column.")
        X = _embed(df["response"].tolist())
        models = df.get("model", pd.Series([f"r{i}" for i in range(len(df))])).astype(str).tolist()
        return X, _pick_label_column(df), models

    # --demo
    demo = [
        ("factual",   "gpt",   "The Pythagorean theorem relates the sides of a right triangle."),
        ("factual",   "llama", "In a right triangle, a squared plus b squared equals c squared."),
        ("bank_money", "gpt",  "I withdrew cash from the bank and deposited my paycheck."),
        ("bank_money", "llama","The bank approved my loan and opened a savings account."),
        ("bank_river", "gpt",  "We sat on the river bank and watched the water flow past."),
        ("bank_river", "llama","The boat drifted toward the muddy bank of the stream."),
        ("nonsense",  "gpt",   "Colorless green ideas sleep furiously beneath the xylophone."),
        ("nonsense",  "llama", "Purple sqrt banana the the running sideways lamp cheese."),
    ]
    groups, models, texts = zip(*demo)
    return _embed(texts), list(groups), list(models)


# ── Plotting ──────────────────────────────────────────────────────────────────
def _kde_curve(values, grid):
    """Smooth bell curve (density) of one response's coordinate values."""
    return gaussian_kde(values)(grid)


def figure_distributions(X_raw, labels, models):
    """Build the raw-vs-normalized value-distribution figure. Returns (fig, rows).

    `rows` is a list of (group, model, raw_peak, norm, raw_std) — the numeric evidence
    that the norm (destroyed by normalization) is what separates the responses.
    """
    norms = np.linalg.norm(X_raw, axis=1)
    X_norm = X_raw / np.clip(norms, 1e-12, None)[:, None]   # what OpenAI would hand you (unit length)

    uniq = list(dict.fromkeys(labels))
    cmap = plt.get_cmap("tab10")
    color = {g: cmap(i % 10) for i, g in enumerate(uniq)}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    for ax, data, title in ((axL, X_raw, "RAW SBERT  (curves separate)"),
                            (axR, X_norm, "L2-NORMALIZED  (curves collapse — the OpenAI case)")):
        grid = np.linspace(data.min(), data.max(), 400)
        seen = set()
        for row, lab in zip(data, labels):
            legend = None
            if lab not in seen:            # one legend entry per group, keeps it readable
                legend = lab
                seen.add(lab)
            ax.plot(grid, _kde_curve(row, grid), color=color[lab], alpha=0.7,
                    linewidth=1.6, label=legend)
        ax.set_title(title)
        ax.set_xlabel("embedding value")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
    fig.suptitle("Per-response embedding value distribution — raw vs normalized", fontsize=13)
    fig.tight_layout()

    grid = np.linspace(X_raw.min(), X_raw.max(), 400)
    rows = []
    for i in np.argsort(labels):
        peak = grid[np.argmax(_kde_curve(X_raw[i], grid))]
        rows.append((str(labels[i]), str(models[i]), float(peak), float(norms[i]), float(X_raw[i].std())))
    return fig, rows


def plot(X_raw, labels, models, out="distributions.png"):
    fig, rows = figure_distributions(X_raw, labels, models)
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")
    print(f"\n{'group':<14}{'model':<10}{'raw peak':>10}{'norm':>9}{'raw std':>10}")
    print("-" * 53)
    for g, m, peak, norm, std in rows:
        print(f"{g:<14}{m:<10}{peak:>+10.3f}{norm:>9.2f}{std:>10.3f}")
    print("\nAll normalized norms are 1.0 by construction — that is why the right panel collapses.")


def _pca2(X):
    """2-D PCA via SVD (no sklearn dependency)."""
    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):  # spurious BLAS FP flags
        return Xc @ Vt[:2].T


def figure_scatter(X_raw, labels, models):
    """Build the semantic figure: PCA cluster scatter + grouped cosine heatmap. Returns fig.

    This is where the money/river/ambiguous 'bank' split is visible — the value-histogram
    can't show it, but direction (cosine) can.
    """
    Xn = X_raw / np.clip(np.linalg.norm(X_raw, axis=1, keepdims=True), 1e-12, None)   # unit -> cosine geometry
    coords = _pca2(Xn)

    uniq = list(dict.fromkeys(labels))
    cmap = plt.get_cmap("tab10")
    color = {g: cmap(i % 10) for i, g in enumerate(uniq)}

    order = np.argsort([uniq.index(l) for l in labels])         # cluster rows by group
    lo = [labels[i] for i in order]
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):  # spurious BLAS FP flags
        sim = Xn[order] @ Xn[order].T

    fig, (axS, axH) = plt.subplots(1, 2, figsize=(15, 6.5))
    for g in uniq:
        m = [i for i, l in enumerate(labels) if l == g]
        axS.scatter(coords[m, 0], coords[m, 1], s=60, color=color[g], label=g,
                    alpha=0.85, edgecolor="white", linewidth=0.5)
    axS.set_title("PCA (2-D) of responses — clusters are topics")
    axS.set_xlabel("PC1"); axS.set_ylabel("PC2"); axS.legend(fontsize=8)

    im = axH.imshow(sim, cmap="viridis", vmin=0, vmax=1)
    axH.set_title("Cosine similarity (grouped) — bright blocks = same meaning")
    ticks, pos = [], 0
    for g in uniq:
        c = lo.count(g)
        ticks.append((pos + c / 2 - 0.5, g, pos))
        pos += c
    axH.set_xticks([t[0] for t in ticks]); axH.set_xticklabels([t[1] for t in ticks], rotation=45, ha="right", fontsize=8)
    axH.set_yticks([t[0] for t in ticks]); axH.set_yticklabels([t[1] for t in ticks], fontsize=8)
    for _, _, p in ticks:
        axH.axhline(p - 0.5, color="white", lw=0.6); axH.axvline(p - 0.5, color="white", lw=0.6)
    fig.colorbar(im, ax=axH, fraction=0.046)
    fig.tight_layout()
    return fig


def plot_scatter(X_raw, labels, models, out="scatter.png"):
    fig = figure_scatter(X_raw, labels, models)
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--csv", help="exported full CSV with sbert_* columns")
    src.add_argument("--responses", help="CSV with a 'response' column (embeds on the fly)")
    src.add_argument("--demo", action="store_true", help="use built-in demo sentences")
    ap.add_argument("--out", default="distributions.png")
    ap.add_argument("--scatter-out", default="scatter.png")
    args = ap.parse_args()
    if not (args.csv or args.responses or args.demo):
        print("No input given — running --demo. See --help for CSV inputs.\n")
        args.demo = True

    X_raw, labels, models = load_data(args)
    plot(X_raw, labels, models, out=args.out)              # value distributions (raw vs normalized)
    plot_scatter(X_raw, labels, models, out=args.scatter_out)   # semantic clusters + cosine heatmap


if __name__ == "__main__":
    main()
