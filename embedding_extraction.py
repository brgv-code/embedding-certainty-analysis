"""
Thesis: Certainty/Uncertainty in LLM Responses

Stage 1 — Ask:    query Gemini, OpenAI, Ollama models in parallel
Stage 2 — Embed:  SBERT + SPLADE embeddings + distribution curve plots 
Stage 3 — Export: download all collected queries at once, organised per query
"""

import json
import time
import concurrent.futures
import streamlit as st
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import requests
from scipy.stats import gaussian_kde
from sentence_transformers import SentenceTransformer, models
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# 2-D PCA helper shared with the standalone plot_distributions.py CLI.
from plot_distributions import _pca2
# Word-level context-awareness study, reused live in the app.
from context_awareness import WORDS as CTX_WORDS, analyze as ctx_analyze

SYNTHETIC_CSV = os.path.join(os.path.dirname(__file__), "synthetic_sentences.csv")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="LLM Certainty Lab", page_icon="🔬", layout="wide")

# ── Constants ─────────────────────────────────────────────────────────────────
OLLAMA_URL    = "http://localhost:11434/api/chat"
OLLAMA_MODELS = ["gemma3:4b", "deepseek-coder:latest", "codellama:latest", "llama3:latest"]
OPENAI_MODEL  = "gpt-4o"
GEMINI_MODEL  = "gemini-2.5-pro"
COLORS = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f"]

# ── Embedding models (cached) ─────────────────────────────────────────────────
@st.cache_resource
def load_sbert():
    # Built WITHOUT the Normalize layer: stock all-mpnet-base-v2 ends in a Normalize module
    # that forces every vector to unit length (same problem as OpenAI), collapsing the value
    # distributions. Transformer + mean Pooling only keeps the true, varying magnitude.
    word = models.Transformer("sentence-transformers/all-mpnet-base-v2")
    pool = models.Pooling(word.get_word_embedding_dimension(), pooling_mode="mean")
    return SentenceTransformer(modules=[word, pool])

@st.cache_resource
def load_splade():
    mid = "naver/splade-cocondenser-selfdistil"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForMaskedLM.from_pretrained(mid)
    mdl.eval()
    return tok, mdl

# ── LLM query functions ───────────────────────────────────────────────────────
def query_openai(prompt, api_key):
    try:
        from openai import OpenAI
        r = OpenAI(api_key=api_key).chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )
        return r.choices[0].message.content.strip(), None
    except Exception as e:
        return "", str(e)

def query_gemini(prompt, api_key, model_id):
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        r = genai.GenerativeModel(model_id).generate_content(prompt)
        return r.text.strip(), None
    except Exception as e:
        return "", str(e)

def query_ollama(prompt, model):
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }, timeout=120)
        r.raise_for_status()
        return r.json()["message"]["content"].strip(), None
    except requests.exceptions.ConnectionError:
        return "", "Ollama not running — start with: ollama serve"
    except Exception as e:
        return "", str(e)

def run_all_models(prompt, openai_key, gemini_key, gemini_model_id, selected_ollama):
    def timed(fn, *args):
        t0 = time.time()
        text, err = fn(*args)
        return {"response": text, "error": err, "elapsed": round(time.time() - t0, 2)}

    results = {}
    with concurrent.futures.ThreadPoolExecutor() as pool:
        futures = {}
        if openai_key:
            futures[OPENAI_MODEL] = pool.submit(timed, query_openai, prompt, openai_key)
        if gemini_key:
            futures[gemini_model_id] = pool.submit(timed, query_gemini, prompt, gemini_key, gemini_model_id)
        for m in selected_ollama:
            futures[m] = pool.submit(timed, query_ollama, prompt, m)
        for name, fut in futures.items():
            results[name] = fut.result()
    return results

# ── Embedding functions ───────────────────────────────────────────────────────
def embed_sbert(text, model):
    # Raw, un-normalized on purpose: L2-normalizing forces every vector to length 1, which
    # collapses each response's value distribution to the same bell curve (why OpenAI failed).
    return model.encode(text, normalize_embeddings=False)

def embed_splade(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    vec = torch.max(torch.log(1 + torch.relu(logits)), dim=1).values
    return vec.squeeze(0).numpy()

def top_splade_tokens(vec, tokenizer, top_n=25):
    indices = np.argsort(vec)[::-1][:top_n]
    return [(tokenizer.convert_ids_to_tokens([i])[0], float(vec[i])) for i in indices]

# ── Plot helpers ──────────────────────────────────────────────────────────────
def dist_plot(embeddings: dict, title: str):
    fig = go.Figure()
    for idx, (label, vec) in enumerate(embeddings.items()):
        fig.add_trace(go.Histogram(
            x=vec, name=label[:50], opacity=0.55, nbinsx=80,
            marker_color=COLORS[idx % len(COLORS)], histnorm="probability density",
        ))
    fig.update_layout(
        title=title, xaxis_title="Value", yaxis_title="Density",
        barmode="overlay", legend=dict(orientation="h", y=-0.28), height=420,
    )
    return fig

def token_bar(vec, tokenizer, label):
    tokens, scores = zip(*top_splade_tokens(vec, tokenizer))
    fig = go.Figure(go.Bar(x=list(tokens), y=list(scores), marker_color="#ff7f0e"))
    fig.update_layout(title=f"Top SPLADE tokens — {label[:50]}",
                      xaxis_title="Token", yaxis_title="Activation", height=380)
    return fig

# ── Interactive collection-wide plots (Plotly — zoom/pan/hover) ───────────────
def _group_colors(labels):
    uniq = list(dict.fromkeys(labels))
    return uniq, {g: COLORS[i % len(COLORS)] for i, g in enumerate(uniq)}

def plotly_pca_scatter(X, labels, models, texts):
    Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    coords = _pca2(Xn)
    uniq, color = _group_colors(labels)
    fig = go.Figure()
    for g in uniq:
        idx = [i for i, l in enumerate(labels) if l == g]
        fig.add_trace(go.Scatter(
            x=coords[idx, 0], y=coords[idx, 1], mode="markers", name=g,
            marker=dict(size=11, color=color[g], line=dict(width=1, color="white")),
            text=[f"{models[i]}<br>{texts[i][:90]}" for i in idx],
            hovertemplate="%{text}<extra>" + g + "</extra>",
        ))
    fig.update_layout(title="PCA (2-D) — clusters are topics (drag to zoom)",
                      xaxis_title="PC1", yaxis_title="PC2", height=560,
                      legend=dict(orientation="h", y=-0.2))
    return fig

def plotly_cosine_heatmap(X, labels, texts=None):
    """One cell per group PAIR (not per response) — average cosine, so it stays readable."""
    Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    groups = list(dict.fromkeys(labels))
    idx = {g: [i for i, l in enumerate(labels) if l == g] for g in groups}
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):  # spurious BLAS FP flags
        sim = Xn @ Xn.T
    G = len(groups)
    M = np.zeros((G, G))
    for i, gi in enumerate(groups):
        for j, gj in enumerate(groups):
            block = sim[np.ix_(idx[gi], idx[gj])]
            if i == j:                       # within-group: average excluding the self-1s
                n = len(idx[gi])
                M[i, j] = (block.sum() - n) / (n * (n - 1)) if n > 1 else 1.0
            else:
                M[i, j] = block.mean()
    fig = go.Figure(go.Heatmap(
        z=M, x=groups, y=groups, colorscale="Viridis", zmin=0, zmax=1,
        text=[[f"{M[i, j]:.2f}" for j in range(G)] for i in range(G)],
        texttemplate="%{text}", textfont={"size": 12},
        hovertemplate="%{y} ↔ %{x}: %{z:.2f}<extra></extra>"))
    fig.update_layout(title="Average similarity between groups — bright diagonal = self-similar",
                      height=460, yaxis=dict(autorange="reversed"))
    return fig

def plotly_value_distributions(X, labels):
    norms = np.linalg.norm(X, axis=1)
    Xn = X / np.clip(norms, 1e-12, None)[:, None]
    uniq, color = _group_colors(labels)
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "RAW SBERT (curves separate)", "L2-NORMALIZED (collapse — the OpenAI case)"))
    for col, data in ((1, X), (2, Xn)):
        grid = np.linspace(data.min(), data.max(), 300)
        seen = set()
        for row, lab in zip(data, labels):
            show = lab not in seen; seen.add(lab)
            fig.add_trace(go.Scatter(
                x=grid, y=gaussian_kde(row)(grid), mode="lines",
                line=dict(color=color[lab], width=1.5), opacity=0.7,
                name=lab, legendgroup=lab, showlegend=(show and col == 1)),
                row=1, col=col)
    fig.update_layout(title="Per-response value distribution — raw vs normalized",
                      height=480, legend=dict(orientation="h", y=-0.25))
    return fig

# ── Context-awareness plots (word-level, Plotly) ──────────────────────────────
def ctx_summary_fig(results):
    words = sorted(results, key=lambda w: results[w]["gap"], reverse=True)
    fig = go.Figure()
    fig.add_bar(x=words, y=[results[w]["within"] for w in words], name="same sense", marker_color="#2ca02c")
    fig.add_bar(x=words, y=[results[w]["cross"] for w in words], name="different sense", marker_color="#d62728")
    fig.update_layout(barmode="group", yaxis_title="cosine similarity", yaxis_range=[0, 1],
                      title="Context-awareness per word — same-sense high, different-sense low",
                      height=460, legend=dict(orientation="h", y=-0.2))
    return fig

def ctx_heatmap_fig(word, r):
    V = r["vecs"]
    Vn = V / np.clip(np.linalg.norm(V, axis=1, keepdims=True), 1e-12, None)
    order = np.argsort([r["senses"].index(l) for l in r["labels"]])
    lab = [r["labels"][i] for i in order]
    sent = [r["sentences"][i] for i in order]
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        sim = Vn[order] @ Vn[order].T
    fig = go.Figure(go.Heatmap(
        z=sim, zmin=0, zmax=1, colorscale="Viridis",
        hovertext=[[f"{lab[i]} / {lab[j]}<br>{sent[i][:55]}<br>{sent[j][:55]}<br>cos={sim[i, j]:.2f}"
                    for j in range(len(order))] for i in range(len(order))],
        hoverinfo="text"))
    fig.update_layout(title=f"'{word}' — bright = same sense, dark = different  (gap {r['gap']:+.2f})",
                      height=480, yaxis=dict(autorange="reversed"))
    return fig

def ctx_csv(results):
    import io, csv as _csv
    buf = io.StringIO(); w = _csv.writer(buf)
    w.writerow(["word", "sense", "sentence"] + [f"dim_{i}" for i in range(768)])
    for word, r in results.items():
        for v, sense, s in zip(r["vecs"], r["labels"], r["sentences"]):
            w.writerow([word, sense, s] + [round(float(x), 6) for x in v])
    return buf.getvalue()

# ── Export helpers (operate on the full collection) ───────────────────────────
def collection_to_full_csv(collection):
    """One row per (query_index, model). All SBERT dims + sparse SPLADE cols."""
    rows = []
    for qi, entry in enumerate(collection):
        for r in entry["records"]:
            row = {
                "query_index": qi,
                "question": r["question"],
                "model": r["model"],
                "response": r["response"],
            }
            for i, v in enumerate(r["sbert"]):
                row[f"sbert_{i}"] = round(float(v), 6)
            for i in np.nonzero(r["splade"])[0]:
                row[f"splade_{i}"] = round(float(r["splade"][i]), 6)
            rows.append(row)
    return pd.DataFrame(rows).to_csv(index=False)

def collection_to_json(collection):
    """Nested structure: list of queries, each with their model records."""
    out = []
    for qi, entry in enumerate(collection):
        query_obj = {
            "query_index": qi,
            "question": entry["question"],
            "models": [],
        }
        for r in entry["records"]:
            nz = np.nonzero(r["splade"])[0].tolist()
            query_obj["models"].append({
                "model": r["model"],
                "response": r["response"],
                "sbert_embedding": [round(float(v), 6) for v in r["sbert"]],
                "splade_sparse": {str(i): round(float(r["splade"][i]), 6) for i in nz},
            })
        out.append(query_obj)
    return json.dumps(out, indent=2)

def collection_to_summary_csv(collection):
    """Lightweight: no raw vectors. Stats only."""
    rows = []
    for qi, entry in enumerate(collection):
        for r in entry["records"]:
            rows.append({
                "query_index": qi,
                "question": r["question"][:120],
                "model": r["model"],
                "response": r["response"][:200],
                "sbert_mean": round(float(np.mean(r["sbert"])), 6),
                "sbert_std":  round(float(np.std(r["sbert"])),  6),
                "splade_nonzero": int(np.count_nonzero(r["splade"])),
                "splade_max":     round(float(np.max(r["splade"])), 4) if getattr(r["splade"], "size", 0) else 0.0,
            })
    return pd.DataFrame(rows).to_csv(index=False)

# ── Session state init ────────────────────────────────────────────────────────
if "collection" not in st.session_state:
    st.session_state["collection"] = []   # list of {question, records:[...]}
if "ask_counter" not in st.session_state:
    st.session_state["ask_counter"] = 0   # increments each time Ask is clicked

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    n_collected = len(st.session_state["collection"])
    st.metric("Queries collected", n_collected)
    if n_collected and st.button("🗑 Clear collection", use_container_width=True):
        st.session_state["collection"] = []
        st.rerun()

    st.divider()
    st.header("Demo data")
    st.caption("Load example sentences to see the plots — no API key needed.")
    if st.button("📥 Load synthetic dataset", use_container_width=True):
        if not os.path.exists(SYNTHETIC_CSV):
            st.error("synthetic_sentences.csv not found — run make_synthetic_dataset.py first.")
        else:
            sdf = pd.read_csv(SYNTHETIC_CSV)
            model = load_sbert()          # cached; SBERT-only load for the demo
            new_collection, done, total = [], 0, len(sdf)
            prog = st.progress(0.0, text="Embedding synthetic sentences…")
            for gname, gdf in sdf.groupby("group"):
                records = []
                for _, r in gdf.iterrows():
                    text = str(r["response"])
                    records.append({
                        "model":    str(r.get("model", "syn")),
                        "question": str(gname),
                        "response": text,
                        "sbert":    embed_sbert(text, model),
                        "splade":   np.array([]),
                    })
                    done += 1
                    prog.progress(done / total)
                new_collection.append({"question": str(gname), "records": records})
            st.session_state["collection"] = new_collection
            st.success(f"Loaded {total} sentences in {len(new_collection)} groups — see tab 2.")
            st.rerun()

    st.divider()
    st.header("API Keys")
    openai_key = st.text_input("OpenAI", type="password", placeholder="sk-…")
    gemini_key = st.text_input("Gemini", type="password", placeholder="AIza…")
    gemini_model_id = st.text_input(
        "Gemini model ID", value=GEMINI_MODEL,
        help="gemini-2.5-pro · gemini-2.5-flash · gemini-1.5-pro",
    )

    st.divider()
    st.header("Ollama Models")
    selected_ollama = [m for m in OLLAMA_MODELS if st.checkbox(m, value=True, key=f"ol_{m}")]

    st.divider()
    st.header("Embedding Models")
    with st.spinner("Loading SBERT…"):
        sbert_model = load_sbert()
    st.success("SBERT ready")
    compute_splade = st.checkbox(
        "Also compute SPLADE (interpretable, slower)", value=False,
        help="Sparse lexical embedding — off by default; SBERT is the focus of the analysis.")
    splade_tok = splade_model = None
    if compute_splade:
        with st.spinner("Loading SPLADE…"):
            splade_tok, splade_model = load_splade()
        st.success("SPLADE ready")

# ── App title ─────────────────────────────────────────────────────────────────
st.title("🔬 LLM Certainty Lab")

tab_ask, tab_embed, tab_export, tab_context = st.tabs([
    "1 · Ask Models",
    "2 · Embeddings & Plots",
    "3 · Export",
    "4 · Context-Awareness",
])

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Ask
# ══════════════════════════════════════════════════════════════════════════════
with tab_ask:
    question = st.text_area("Your question", height=80,
                            placeholder="e.g. What is the Pythagorean theorem?")

    active = []
    if openai_key: active.append(OPENAI_MODEL)
    if gemini_key: active.append(gemini_model_id)
    active += selected_ollama

    if not active:
        st.warning("Add an API key or enable at least one Ollama model in the sidebar.")

    if st.button("▶ Ask all models", type="primary",
                 disabled=not question.strip() or not active):
        with st.spinner(f"Querying {len(active)} model(s) in parallel…"):
            answers = run_all_models(
                question.strip(), openai_key, gemini_key, gemini_model_id, selected_ollama
            )
        st.session_state["ask_counter"] += 1
        st.session_state["answers"] = {"question": question.strip(), "results": answers}

    # ── Display answers ───────────────────────────────────────────────────────
    if "answers" in st.session_state:
        data    = st.session_state["answers"]
        results = data["results"]

        st.markdown(f"**Q:** {data['question']}")
        st.divider()

        cols = st.columns(min(len(results), 4))
        for idx, (model_name, r) in enumerate(results.items()):
            with cols[idx % len(cols)]:
                if r["error"]:
                    st.error(f"**{model_name}** ({r['elapsed']}s)\n\n{r['error']}")
                else:
                    st.markdown(f"**{model_name}** `{r['elapsed']}s`")
                    st.text_area("", value=r["response"], height=300,
                                 key=f"ans_{st.session_state['ask_counter']}_{model_name}",
                                 label_visibility="collapsed")

        st.divider()

        col_embed, col_status = st.columns([1, 2])
        with col_embed:
            embed_clicked = st.button("➡ Embed & add to collection", type="primary")

        # Check if this question is already in the collection
        already_collected = any(
            e["question"] == data["question"]
            for e in st.session_state["collection"]
        )
        with col_status:
            if already_collected:
                st.info("This question is already in the collection.")

        if embed_clicked and not already_collected:
            good = {m: r for m, r in results.items() if r["response"] and not r["error"]}
            records = []
            prog = st.progress(0)
            for idx, (model_name, r) in enumerate(good.items()):
                text = r["response"]
                records.append({
                    "model":    model_name,
                    "question": data["question"],
                    "response": text,
                    "sbert":    embed_sbert(text, sbert_model),
                    "splade":   embed_splade(text, splade_tok, splade_model) if compute_splade else np.array([]),
                })
                prog.progress((idx + 1) / len(good))

            st.session_state["collection"].append({
                "question": data["question"],
                "records":  records,
            })
            # Also store as "current" for the plots tab
            st.session_state["current_records"] = records
            st.success(
                f"Added to collection — "
                f"{len(st.session_state['collection'])} query/queries total. "
                "Go to **2 · Embeddings & Plots** or ask another question."
            )
        elif embed_clicked and already_collected:
            st.warning("Already in collection. Ask a different question to add more.")

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Embeddings & Plots
# ══════════════════════════════════════════════════════════════════════════════
with tab_embed:
    collection = st.session_state["collection"]

    if not collection:
        st.info("Load the **synthetic dataset** from the sidebar, or ask a question in "
                "**1 · Ask Models**, to see the data and the semantic map here.")
    else:
        all_records = [(e["question"], r) for e in collection for r in e["records"]]

        # ── The loaded data — one tab per group, side by side ─────────────────
        st.subheader(f"Loaded data — {len(collection)} groups · {len(all_records)} responses")
        groups = list(dict.fromkeys(q for q, _ in all_records))
        for tab, g in zip(st.tabs([g[:28] for g in groups]), groups):
            with tab:
                st.dataframe(
                    pd.DataFrame([{"model": r["model"], "response": r["response"]}
                                 for q, r in all_records if q == g]),
                    use_container_width=True, hide_index=True,
                )
        if st.button("🗑 Clear all", key="clear_embed"):
            st.session_state["collection"] = []
            st.rerun()

        if len(all_records) >= 2:
            X = np.array([r["sbert"] for _, r in all_records], dtype=float)
            labels = [q[:24] for q, _ in all_records]     # each group/question is a cluster
            mdls   = [r["model"] for _, r in all_records]
            texts  = [r["response"] for _, r in all_records]

            # ── Value distribution (shown first) ──────────────────────────────
            st.divider()
            st.subheader("Value distribution")
            st.caption("Each response's embedding values as a curve — raw vs L2-normalized.")
            st.plotly_chart(plotly_value_distributions(X, labels), use_container_width=True)

            # ── Semantic map ──────────────────────────────────────────────────
            st.divider()
            st.subheader("Semantic map")
            st.caption("Same meaning clusters together, different meaning lands apart. "
                       "Drag to zoom, hover to read a response.")
            st.plotly_chart(plotly_pca_scatter(X, labels, mdls, texts), use_container_width=True)
            st.plotly_chart(plotly_cosine_heatmap(X, labels, texts), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — Export
# ══════════════════════════════════════════════════════════════════════════════
with tab_export:
    collection = st.session_state["collection"]

    if not collection:
        st.info("Collect at least one query in Stage 1 first.")
    else:
        total_records = sum(len(e["records"]) for e in collection)
        st.subheader(f"Download — {len(collection)} queries · {total_records} embeddings")

        # ── Save raw vectors to a file on disk ────────────────────────────────
        st.markdown("#### Save raw vectors to disk")
        default_path = os.path.join(os.path.dirname(__file__), "raw_vectors.csv")
        save_path = st.text_input("File path", value=default_path, key="save_path")
        if st.button("💾 Save full CSV (raw SBERT + sparse SPLADE)"):
            try:
                with open(save_path, "w") as fh:
                    fh.write(collection_to_full_csv(collection))
                st.success(f"Saved {total_records} rows → {save_path}")
            except Exception as exc:
                st.error(f"Could not save: {exc}")
        st.divider()

        # ── Per-query download ────────────────────────────────────────────────
        st.markdown("#### Per-query downloads")
        for qi, entry in enumerate(collection):
            q_label = f"Q{qi+1}: {entry['question'][:60]}"
            with st.expander(q_label):
                c1, c2, c3 = st.columns(3)
                single = [entry]
                with c1:
                    st.download_button(
                        "⬇ CSV (full vectors)",
                        collection_to_full_csv(single).encode(),
                        f"q{qi+1}_full.csv", "text/csv",
                        key=f"csv_full_{qi}",
                    )
                with c2:
                    st.download_button(
                        "⬇ JSON",
                        collection_to_json(single).encode(),
                        f"q{qi+1}.json", "application/json",
                        key=f"json_{qi}",
                    )
                with c3:
                    st.download_button(
                        "⬇ Summary CSV",
                        collection_to_summary_csv(single).encode(),
                        f"q{qi+1}_summary.csv", "text/csv",
                        key=f"csv_sum_{qi}",
                    )

        # ── Bulk download (all queries) ───────────────────────────────────────
        st.divider()
        st.markdown("#### Download all queries at once")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button(
                "⬇ all_full.csv",
                collection_to_full_csv(collection).encode(),
                "all_full.csv", "text/csv",
            )
        with c2:
            st.download_button(
                "⬇ all_embeddings.json",
                collection_to_json(collection).encode(),
                "all_embeddings.json", "application/json",
            )
        with c3:
            st.download_button(
                "⬇ all_summary.csv",
                collection_to_summary_csv(collection).encode(),
                "all_summary.csv", "text/csv",
            )

        st.divider()
        st.subheader("JSON Preview (first query)")
        st.json(json.loads(collection_to_json(collection[:1])))

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — Context-Awareness (word-level, pre-normalization vectors)
# ══════════════════════════════════════════════════════════════════════════════
with tab_context:
    st.caption(
        "The raw (pre-normalization) contextual vector of an ambiguous word is **close within a "
        "sense** and **far across senses**. This is context-awareness — shown with cosine, no PCA. "
        "Uses the transformer's token vectors before pooling/normalization."
    )
    chosen = st.multiselect(
        "Ambiguous words", options=list(CTX_WORDS), default=list(CTX_WORDS)[:8],
        help="Each word has two senses × sample sentences (defined in context_awareness.py).")

    if st.button("▶ Run context-awareness analysis", type="primary", disabled=not chosen):
        with st.spinner(f"Embedding {len(chosen)} words…"):
            # reuse the already-loaded SBERT transformer (no second model download)
            hf_model = sbert_model[0].auto_model
            hf_tok = sbert_model[0].tokenizer
            st.session_state["ctx_results"] = ctx_analyze(
                hf_model, hf_tok, {w: CTX_WORDS[w] for w in chosen})

    results = st.session_state.get("ctx_results")
    if results:
        gaps = [results[w]["gap"] for w in results]
        st.metric("Mean gap (same − different)", f"{np.mean(gaps):+.3f}",
                  help="Positive for every word means each word moves with its context.")
        st.plotly_chart(ctx_summary_fig(results), use_container_width=True)

        word = st.selectbox("Inspect a word's cosine heatmap", options=list(results))
        st.plotly_chart(ctx_heatmap_fig(word, results[word]), use_container_width=True)

        st.dataframe(pd.DataFrame(
            [(w, round(results[w]["within"], 3), round(results[w]["cross"], 3), round(results[w]["gap"], 3))
             for w in sorted(results, key=lambda w: results[w]["gap"], reverse=True)],
            columns=["word", "same-sense", "different-sense", "gap"]), use_container_width=True)

        st.download_button("⬇ Download raw word vectors (CSV, pre-normalization)",
                           ctx_csv(results).encode(), "word_context_vectors.csv", "text/csv")
    else:
        st.info("Pick some words and click **Run** to compute the vectors and plots.")
