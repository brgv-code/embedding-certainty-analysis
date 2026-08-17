"""
context_awareness.py — Demonstrate CONTEXT-AWARE behavior using PRE-NORMALIZATION vectors,
across a large set of ambiguous (polysemous) words.

Context-awareness lives at the WORD level: the raw contextual vector of an ambiguous word is
close within a sense and far across senses. Uses last_hidden_state — before pooling or
normalization — so it answers "plot the vector components before normalization", with no PCA.

Outputs
  word_context_vectors.csv    one row per occurrence: word, sense, sentence, dim_0..dim_767 (RAW)
  context_heatmaps.png        cosine heatmap per word (two bright sense-blocks)
  context_summary.png         within-sense vs across-sense cosine for every word
  prints a summary table

Run:  python context_awareness.py     (add words by editing the WORDS dict)
"""

import csv
import itertools

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel

MODEL = "sentence-transformers/all-mpnet-base-v2"

# ambiguous word -> two senses -> sentences (the word must appear literally)
WORDS = {
    "bank": {
        "money": ["I deposited my paycheck at the bank.", "The bank charged a fee on my account.",
                  "She took out a loan from the bank.", "The bank raised its interest rates."],
        "river": ["We picnicked on the river bank.", "The boat drifted to the far bank.",
                  "Willows lined the muddy bank.", "He fished from the grassy bank."],
    },
    "bat": {
        "animal":   ["A bat flew out of the dark cave.", "The bat hung upside down at dusk.",
                     "The vampire bat feeds at night.", "A fruit bat glided between the trees."],
        "baseball": ["He swung the bat and hit a home run.", "The batter gripped the wooden bat.",
                     "She cracked the ball with an aluminum bat.", "The bat shattered on the fastball."],
    },
    "bark": {
        "tree": ["The tree's bark was rough and gray.", "Beetles burrowed under the bark.",
                 "Cinnamon comes from tree bark.", "The bark peeled off the birch."],
        "dog":  ["The dog let out a loud bark.", "Her puppy would bark at strangers.",
                 "The bark echoed through the yard.", "He heard a sharp bark outside."],
    },
    "spring": {
        "season": ["Flowers bloom in the spring.", "Spring brings warmer weather.",
                   "The birds return every spring.", "We planted seeds in the spring."],
        "coil":   ["The mattress spring snapped.", "A metal spring compressed under load.",
                   "The clock's spring needs winding.", "The trampoline spring stretched."],
    },
    "bass": {
        "fish":  ["He caught a large bass in the lake.", "The bass swam near the reeds.",
                  "We grilled the fresh bass.", "Anglers prize the striped bass."],
        "music": ["The bass guitar rumbled through the hall.", "He plays bass in a jazz band.",
                  "The song's bass line was deep.", "Turn up the bass on the speakers."],
    },
    "crane": {
        "bird":    ["A crane waded in the shallow marsh.", "The crane spread its long wings.",
                    "Cranes migrate south in winter.", "The gray crane stood on one leg."],
        "machine": ["The crane lifted the steel beam.", "A tower crane loomed over the site.",
                    "The crane hoisted the container.", "Operators guided the crane's arm."],
    },
    "seal": {
        "animal": ["The seal swam near the ice floe.", "A seal barked on the rocky shore.",
                   "The gray seal dove for fish.", "Seals basked in the winter sun."],
        "stamp":  ["She pressed a wax seal on the letter.", "The king's seal marked the document.",
                   "He broke the seal on the envelope.", "An official seal certified the deed."],
    },
    "mouse": {
        "animal":   ["A mouse scurried across the kitchen floor.", "The cat chased a tiny mouse.",
                     "A field mouse nibbled the grain.", "The mouse hid behind the stove."],
        "computer": ["I clicked the link with my mouse.", "The wireless mouse needs batteries.",
                     "She dragged the file using the mouse.", "His mouse cursor froze on screen."],
    },
    "plant": {
        "factory":    ["The car plant employs thousands of workers.", "The power plant runs on gas.",
                       "Inspectors toured the chemical plant.", "The assembly plant shut for repairs."],
        "vegetation": ["She watered the plant on the windowsill.", "The plant grew toward the sunlight.",
                       "He repotted the wilting plant.", "A climbing plant covered the fence."],
    },
    "club": {
        "weapon":    ["He swung the heavy wooden club.", "The guard carried a club.",
                      "A stone club lay in the museum.", "She raised the club to defend herself."],
        "nightclub": ["We danced at the club all night.", "The club was packed on Friday.",
                      "A DJ spun records at the club.", "They waited in line outside the club."],
    },
    "bolt": {
        "lightning": ["A bolt of lightning split the sky.", "The bolt struck the old oak tree.",
                      "A bright bolt flashed overhead.", "Thunder followed the bolt instantly."],
        "fastener":  ["Tighten the bolt with a wrench.", "The bolt held the beam in place.",
                      "He loosened the rusty bolt.", "A missing bolt weakened the frame."],
    },
    "pupil": {
        "eye":     ["The doctor examined her dilated pupil.", "His pupil narrowed in the light.",
                    "The pupil of the eye is black.", "Drops widened the patient's pupil."],
        "student": ["The pupil raised her hand in class.", "Every pupil passed the exam.",
                    "The teacher praised the diligent pupil.", "A new pupil joined the school."],
    },
    "ring": {
        "jewelry": ["She wore a diamond ring.", "He slipped the wedding ring on.",
                    "The gold ring was engraved.", "A silver ring circled her finger."],
        "sound":   ["I heard the phone ring.", "The bells ring at noon.",
                    "A sudden ring startled me.", "The alarm gave a shrill ring."],
    },
    "nail": {
        "finger": ["She painted her nail bright red.", "He bit his thumb nail nervously.",
                   "A broken nail snagged the fabric.", "Her nail polish chipped."],
        "metal":  ["Hammer the nail into the board.", "He bent the nail with pliers.",
                   "A rusty nail pierced the plank.", "Drive the nail flush with the wood."],
    },
    "palm": {
        "hand": ["He held the coin in his palm.", "She read the lines on his palm.",
                 "Sweat gathered in my palm.", "The ball fit snugly in her palm."],
        "tree": ["A palm swayed on the sandy beach.", "Coconuts fell from the palm.",
                 "The tall palm shaded the path.", "Palms lined the tropical road."],
    },
    "jam": {
        "food":    ["Spread strawberry jam on the toast.", "She made raspberry jam last summer.",
                    "The jam jar was almost empty.", "He prefers apricot jam."],
        "traffic": ["We were stuck in a traffic jam.", "The jam stretched for miles.",
                    "A jam formed after the crash.", "The rush-hour jam delayed the bus."],
    },
    "bug": {
        "insect":   ["A bug crawled up the wall.", "The garden was full of bugs.",
                     "A tiny bug landed on the leaf.", "She swatted the buzzing bug."],
        "software": ["The developer fixed the software bug.", "A bug crashed the application.",
                     "They reported the bug to support.", "The update patched a security bug."],
    },
    "current": {
        "river":   ["The river current was strong.", "Swimmers fought the ocean current.",
                    "The current carried the raft downstream.", "A rip current pulled him out."],
        "present": ["The current situation is difficult.", "Our current plan needs revision.",
                    "The current CEO joined last year.", "Under current rules, this is banned."],
    },
}


def word_vector(model, tok, sentence, word):
    """Raw (pre-normalization) contextual vector of `word` in `sentence`, or None."""
    enc = tok(sentence, return_tensors="pt", return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    start = sentence.lower().find(word)
    if start < 0:
        return None
    end = start + len(word)
    pos = [i for i, (a, b) in enumerate(offsets) if b > a and a < end and b > start]
    if not pos:
        return None
    device = next(model.parameters()).device          # model may be on MPS/GPU (Streamlit) or CPU (CLI)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        h = model(**enc).last_hidden_state[0].cpu()    # (seq, 768) raw, pre-normalization
    return h[pos].mean(0).numpy()


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def analyze(model, tok, words):
    """Return {word: {vecs, labels, sentences, senses, within, cross, gap}} for the given words.

    `model`/`tok` are a raw HF transformer + fast tokenizer (last_hidden_state, offsets).
    Reusable by both the CLI and the Streamlit app.
    """
    out = {}
    for word, senses in words.items():
        vecs, labels, sents = [], [], []
        for sense, sentences in senses.items():
            for s in sentences:
                v = word_vector(model, tok, s, word)
                if v is None:
                    continue
                vecs.append(v); labels.append(sense); sents.append(s)
        snames = list(senses)
        by = {sn: [v for v, l in zip(vecs, labels) if l == sn] for sn in snames}
        within = float(np.mean([cos(a, b) for sn in snames for a, b in itertools.combinations(by[sn], 2)]))
        cross = float(np.mean([cos(a, b) for a in by[snames[0]] for b in by[snames[1]]]))
        out[word] = {"vecs": np.array(vecs), "labels": labels, "sentences": sents,
                     "senses": snames, "within": within, "cross": cross, "gap": within - cross}
    return out


def main():
    print(f"Loading {MODEL} …")
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModel.from_pretrained(MODEL)
    mdl.eval()

    ncols = 4
    nrows = -(-len(WORDS) // ncols)
    fig_h, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.2 * nrows))
    axes = axes.ravel()

    csv_rows, summary = [], []
    for ax, (word, senses) in zip(axes, WORDS.items()):
        vecs, labels = [], []
        for sense, sentences in senses.items():
            for s in sentences:
                v = word_vector(mdl, tok, s, word)
                if v is None:
                    print(f"  (skipped) '{word}' not located in: {s}")
                    continue
                vecs.append(v); labels.append(sense)
                csv_rows.append([word, sense, s] + [round(float(x), 6) for x in v])

        snames = list(senses)
        by = {sn: [v for v, l in zip(vecs, labels) if l == sn] for sn in snames}
        within = np.mean([cos(a, b) for sn in snames for a, b in itertools.combinations(by[sn], 2)])
        cross = np.mean([cos(a, b) for a in by[snames[0]] for b in by[snames[1]]])
        summary.append((word, within, cross, within - cross))

        V = np.array(vecs); Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
        im = ax.imshow(Vn @ Vn.T, cmap="viridis", vmin=0, vmax=1)
        n0 = labels.count(snames[0])
        ax.axhline(n0 - 0.5, color="w", lw=1); ax.axvline(n0 - 0.5, color="w", lw=1)
        ax.set_title(f"'{word}'  ({snames[0]}|{snames[1]})", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    for ax in axes[len(WORDS):]:
        ax.axis("off")
    fig_h.suptitle("Cosine of each ambiguous word's raw vectors — two bright blocks = two senses",
                   fontsize=13)
    fig_h.tight_layout()
    fig_h.savefig("context_heatmaps.png", dpi=150)
    print("Saved context_heatmaps.png")

    # ── CSV of raw components ─────────────────────────────────────────────────
    with open("word_context_vectors.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["word", "sense", "sentence"] + [f"dim_{i}" for i in range(768)])
        w.writerows(csv_rows)
    print(f"Saved word_context_vectors.csv — {len(csv_rows)} vectors")

    # ── Summary bar chart (sorted by gap) ─────────────────────────────────────
    summary.sort(key=lambda r: r[3], reverse=True)
    words = [r[0] for r in summary]
    x = np.arange(len(words)); bw = 0.4
    fig_s, ax = plt.subplots(figsize=(max(10, 0.7 * len(words)), 5.5))
    ax.bar(x - bw / 2, [r[1] for r in summary], bw, label="same sense", color="#2ca02c")
    ax.bar(x + bw / 2, [r[2] for r in summary], bw, label="different sense", color="#d62728")
    for i, r in enumerate(summary):
        ax.annotate(f"{r[3]:+.2f}", (i, max(r[1], r[2]) + 0.02), ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(words, rotation=45, ha="right")
    ax.set_ylabel("cosine similarity"); ax.set_ylim(0, 1)
    ax.set_title("Context-awareness per word (gap = same − different)")
    ax.legend()
    fig_s.tight_layout(); fig_s.savefig("context_summary.png", dpi=150)
    print("Saved context_summary.png")

    # ── Printed table ─────────────────────────────────────────────────────────
    print(f"\n{'word':<9}{'same-sense':>12}{'diff-sense':>12}{'gap':>8}")
    print("-" * 41)
    for word, wi, cr, gap in summary:
        print(f"{word:<9}{wi:>12.3f}{cr:>12.3f}{gap:>+8.3f}")
    gaps = [r[3] for r in summary]
    print(f"\nmean gap = {np.mean(gaps):+.3f}   all positive: {all(g > 0 for g in gaps)}   "
          f"({len(WORDS)} words, {len(csv_rows)} vectors)")


if __name__ == "__main__":
    main()
