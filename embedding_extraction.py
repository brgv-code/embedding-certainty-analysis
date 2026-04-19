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
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

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
    return SentenceTransformer("all-mpnet-base-v2")

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
    return model.encode(text, normalize_embeddings=True)

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
                "splade_max":     round(float(np.max(r["splade"])), 4),
            })
    return pd.DataFrame(rows).to_csv(index=False)

# ── Session state init ────────────────────────────────────────────────────────
if "collection" not in st.session_state:
    st.session_state["collection"] = []   # list of {question, records:[...]}
if "ask_counter" not in st.session_state:
    st.session_state["ask_counter"] = 0   # increments each time Ask is clicked

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
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
    with st.spinner("Loading SPLADE…"):
        splade_tok, splade_model = load_splade()
    st.success("SPLADE ready")

    st.divider()
    n_collected = len(st.session_state["collection"])
    st.metric("Queries collected", n_collected)
    if n_collected and st.button("🗑 Clear collection", use_container_width=True):
        st.session_state["collection"] = []
        st.rerun()

# ── App title ─────────────────────────────────────────────────────────────────
st.title("🔬 LLM Certainty Lab")

tab_ask, tab_embed, tab_export = st.tabs([
    "1 · Ask Models",
    "2 · Embeddings & Plots",
    "3 · Export",
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
                    "splade":   embed_splade(text, splade_tok, splade_model),
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
        st.info('Ask a question in **1 · Ask Models**, then click "Embed & add to collection".')
    else:
        # Let the user pick which query to inspect
        questions = [f"Q{i+1}: {e['question'][:80]}" for i, e in enumerate(collection)]
        selected_q = st.selectbox("Inspect query", options=questions)
        qi = questions.index(selected_q)
        records = collection[qi]["records"]

        col_remove, _ = st.columns([1, 4])
        with col_remove:
            if st.button("Remove this query from collection", key="remove_q"):
                st.session_state["collection"].pop(qi)
                st.rerun()

        st.divider()

        # ── SBERT ─────────────────────────────────────────────────────────────
        st.subheader("SBERT — Distribution Curves (768 dims)")
        st.caption(
            "Overlapping curves → models agree on this question. "
            "Spread apart → models disagree."
        )
        st.plotly_chart(
            dist_plot({r["model"]: r["sbert"] for r in records},
                      f"SBERT · {collection[qi]['question'][:60]}"),
            use_container_width=True,
        )

        # ── SPLADE ────────────────────────────────────────────────────────────
        st.subheader("SPLADE — Distribution Curves (active dims)")
        st.plotly_chart(
            dist_plot({r["model"]: r["splade"][r["splade"] > 0] for r in records},
                      f"SPLADE · {collection[qi]['question'][:60]}"),
            use_container_width=True,
        )

        # ── SPLADE top tokens ─────────────────────────────────────────────────
        st.subheader("SPLADE — Top Activated Tokens")
        model_names = [r["model"] for r in records]
        sel_model = st.selectbox("Pick a model", options=model_names, key="tok_model")
        rec = next(r for r in records if r["model"] == sel_model)
        st.plotly_chart(token_bar(rec["splade"], splade_tok, sel_model),
                        use_container_width=True)

        # ── Summary table ─────────────────────────────────────────────────────
        st.subheader("Numeric Summary")
        st.dataframe(pd.DataFrame([{
            "Model": r["model"],
            "SBERT mean": round(float(np.mean(r["sbert"])), 5),
            "SBERT std":  round(float(np.std(r["sbert"])),  5),
            "SPLADE non-zero": int(np.count_nonzero(r["splade"])),
            "SPLADE max":      round(float(np.max(r["splade"])), 4),
        } for r in records]), use_container_width=True)

        # ── Raw vector preview ────────────────────────────────────────────────
        st.divider()
        st.subheader("Raw Embedding Vectors")

        prev_model = st.selectbox("Pick a model to inspect", options=model_names, key="vec_model")
        prev_rec = next(r for r in records if r["model"] == prev_model)

        st.markdown(f"**SBERT — {prev_model}** (768 values)")
        sbert_vec = prev_rec["sbert"]
        sbert_df = pd.DataFrame(
            [sbert_vec.astype(float)],
            columns=[str(i) for i in range(len(sbert_vec))],
        )
        st.dataframe(sbert_df, use_container_width=True)

        st.markdown(f"**SPLADE — {prev_model}** (active tokens only, sorted by value)")
        splade_vec = prev_rec["splade"]
        nz_idx = np.nonzero(splade_vec)[0]
        nz_tokens = [splade_tok.convert_ids_to_tokens([int(i)])[0] for i in nz_idx]
        nz_values = splade_vec[nz_idx].astype(float)
        order = np.argsort(nz_values)[::-1]
        splade_df = pd.DataFrame(
            [nz_values[order]],
            columns=[nz_tokens[i] for i in order],
        )
        st.dataframe(splade_df, use_container_width=True)

        # ── Collection overview ───────────────────────────────────────────────
        if len(collection) > 1:
            st.divider()
            st.subheader(f"Collection overview ({len(collection)} queries)")
            overview = []
            for i, entry in enumerate(collection):
                for r in entry["records"]:
                    overview.append({
                        "Q#": i + 1,
                        "Question": entry["question"][:70],
                        "Model": r["model"],
                        "SBERT std": round(float(np.std(r["sbert"])), 5),
                        "SPLADE non-zero": int(np.count_nonzero(r["splade"])),
                    })
            st.dataframe(pd.DataFrame(overview), use_container_width=True)

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
