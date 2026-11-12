
import re

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="SmartResume AI", page_icon="📄", layout="wide")

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

STOP = set(
    """a an the and or of to in on for with by at from as is are be been being this that
    these those your you we our their its it will can may must should would could also such
    than then so into onto over under between among within without across about per via
    both either neither each other any all some more most many much very well etc ie eg who
    whom which what when where why how not no nor if else while during after before up down
    have has had do does using use used like them they he she his her s
    le la les un une des du de d l et ou en dans sur pour avec par au aux ce cette ces votre
    vos nous notre leur leurs son sa ses est sont être ainsi comme plus très aussi sans sous
    entre chez vers qui que quoi dont où""".split()
)

GENERIC = set(
    """strong solid good great excellent best better new nice key main various several
    multiple real experience experiences knowledge understanding ability abilities skills
    skill proficiency familiarity required preferred working work works related closely
    including include includes team role looking join responsibilities responsibility
    qualifications qualification degree years year build evaluate develop write deploy
    collect clean present track monitor design analyze extract solve ensure contribute
    manage create implement maintain perform provide support help need needs able
    communication autonomous curious summary profile education languages projects interests
    references title job description fields field problems problem
    compétences compétence connaissance connaissances expérience maîtrise bonne bon
    capacité formation profil langues projets""".split()
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {background-color: #0B1F3A;}
    [data-testid="stSidebar"] * {color: #ffffff;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Phase 1-2 : PDF extraction + cleaning ----------
@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)


def extract_text(uploaded_file):
    data = uploaded_file.read()
    doc = fitz.open(stream=data, filetype="pdf")
    return "\n".join(page.get_text() for page in doc)


def clean(text):
    text = re.sub(r"[•▪●■◦►]", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


# ---------- Phase 3 : dynamic skill extraction (no dictionary) ----------
def extract_phrases(text):
    """RAKE-style: split segments at stopwords, trim generic words,
    keep single words only if they look technical."""
    segments = re.split(r"[\n\r,;:•|/()\[\]]+|\s-\s|\.\s", text)
    found = []
    for seg in segments:
        toks = re.findall(r"\w[\w+#.\-]*", seg)
        toks = [t.strip(".-") for t in toks]
        toks = [t for t in toks if t]
        runs, cur = [], []
        for i, t in enumerate(toks):
            if t.lower() in STOP:
                if cur:
                    runs.append(cur)
                cur = []
            else:
                cur.append((t, i))
        if cur:
            runs.append(cur)
        for run in runs:
            while run and run[0][0].lower() in GENERIC:
                run = run[1:]
            while run and run[-1][0].lower() in GENERIC:
                run = run[:-1]
            if not run:
                continue
            if len(run) == 1:
                tok, idx = run[0]
                technical = (
                    any(c.isupper() for c in tok[1:])
                    or bool(re.search(r"[+#_\d]", tok))
                    or (tok[0].isupper() and idx > 0)
                    or len(toks) == 1
                )
                if not technical:
                    continue
                cands = [[tok]]
            elif len(run) <= 5:
                cands = [[t for t, _ in run]]
            else:
                w = [t for t, _ in run]
                cands = [w[i : i + 3] for i in range(len(w) - 2)]
            for c in cands:
                p = " ".join(c).lower()
                if len(p) < 2 or re.match(r"^\d", p) or re.search(r"\d{4,}|@|http", p):
                    continue
                found.append(p)
    return list(dict.fromkeys(found))


def top_keyphrases(text, model, top_n=25):
    """Rank candidate phrases by similarity to the whole text (KeyBERT-like),
    then drop near-duplicates."""
    cands = extract_phrases(text)[:800]
    if not cands:
        return []
    emb = model.encode(cands, normalize_embeddings=True)
    doc = emb.mean(axis=0)
    doc = doc / (np.linalg.norm(doc) + 1e-9)
    scores = emb @ doc
    chosen = []
    for i in np.argsort(-scores):
        if all(float(emb[i] @ emb[j]) < 0.85 for j in chosen):
            chosen.append(i)
        if len(chosen) == top_n:
            break
    return [cands[i] for i in chosen]


def doc_vec(text, model, size=60):
    words = text.split()
    chunks = [" ".join(words[i : i + size]) for i in range(0, len(words), size)] or [text]
    emb = model.encode(chunks, normalize_embeddings=True)
    v = emb.mean(axis=0)
    return v / (np.linalg.norm(v) + 1e-9)


# ---------- Phase 4 : Approach A (TF-IDF) vs Approach B (Transformers) ----------
def run_analysis(cv_text, jd_text):
    model = load_model()
    jd_skills = top_keyphrases(jd_text, model, 25)
    cv_phrases = extract_phrases(cv_text)[:800]
    if not jd_skills or not cv_phrases:
        return None

    # Approach B: Sentence Transformers
    jd_emb = model.encode(jd_skills, normalize_embeddings=True)
    cv_emb = model.encode(cv_phrases, normalize_embeddings=True)
    sims_b = jd_emb @ cv_emb.T
    best_b = sims_b.max(axis=1)
    best_phrase_b = [cv_phrases[i] for i in sims_b.argmax(axis=1)]
    global_b = float(doc_vec(cv_text, model) @ doc_vec(jd_text, model))

    # Approach A: TF-IDF
    vec = TfidfVectorizer(token_pattern=r"(?u)[\w+#]+").fit(jd_skills + cv_phrases)
    sims_a = cosine_similarity(vec.transform(jd_skills), vec.transform(cv_phrases))
    best_a = sims_a.max(axis=1)
    tf = TfidfVectorizer(ngram_range=(1, 2)).fit_transform([cv_text, jd_text])
    global_a = float(cosine_similarity(tf)[0, 1])

    return {
        "jd_skills": jd_skills,
        "best_a": best_a.tolist(),
        "best_b": best_b.tolist(),
        "best_phrase_b": best_phrase_b,
        "global_a": global_a,
        "global_b": global_b,
    }


def classify(sim, strong, partial):
    if sim >= strong:
        return "strong"
    if sim >= partial:
        return "partial"
    return "missing"


def coverage(sims, strong, partial):
    labels = [classify(s, strong, partial) for s in sims]
    return (labels.count("strong") + 0.5 * labels.count("partial")) / max(len(labels), 1) * 100


# ---------- Phase 5 : UI ----------
def circle(score):
    color = "#16a34a" if score >= 70 else "#f59e0b" if score >= 40 else "#dc2626"
    st.markdown(
        f"""
        <div style="width:170px;height:170px;border-radius:50%;
             background:conic-gradient({color} {score * 3.6}deg,#e5e7eb 0deg);
             display:flex;align-items:center;justify-content:center;margin:auto">
          <div style="width:130px;height:130px;border-radius:50%;background:white;
               display:flex;align-items:center;justify-content:center;
               font-size:34px;font-weight:700;color:#0B1F3A">{score:.0f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chips(items, icon, color):
    if not items:
        st.caption("—")
        return
    html = "".join(
        f'<span style="display:inline-block;background:{color}20;border:1px solid {color};'
        f'color:{color};padding:4px 10px;border-radius:14px;margin:3px;font-size:14px">'
        f"{icon} {sk} ({s:.2f})</span>"
        for sk, s in items
    )
    st.markdown(html, unsafe_allow_html=True)


def show_results(res, strong, partial):
    skills, sims = res["jd_skills"], res["best_b"]
    groups = {"strong": [], "partial": [], "missing": []}
    for sk, s in zip(skills, sims):
        groups[classify(s, strong, partial)].append((sk, s))
    overall = coverage(sims, strong, partial)

    st.divider()
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Overall Match")
        circle(overall)
    with c2:
        st.subheader("Smart Summary")
        st.write(
            f"The CV covers **{len(groups['strong'])}** required skills strongly, "
            f"**{len(groups['partial'])}** partially, and **{len(groups['missing'])}** are missing "
            f"(out of {len(skills)} skills extracted from the job description)."
        )
        if groups["missing"]:
            st.write("Main gaps: " + ", ".join(sk for sk, _ in groups["missing"][:5]))

    st.subheader("Strong skills")
    chips(groups["strong"], "✓", "#16a34a")
    st.subheader("Partial skills")
    chips(groups["partial"], "~", "#f59e0b")
    st.subheader("Missing skills")
    chips(groups["missing"], "✗", "#dc2626")


def page_analysis(strong, partial):
    st.title("📄 Resume Analysis")
    col1, col2 = st.columns(2)
    with col1:
        cv_file = st.file_uploader("Upload your CV (PDF)", type=["pdf"])
    with col2:
        jd_file = st.file_uploader("Or upload Job_description.txt", type=["txt"])
    jd_text = st.text_area("Job Description", height=200, placeholder="Paste the job description here...")
    if jd_file is not None:
        jd_text = jd_file.read().decode("utf-8", errors="ignore")

    if st.button("Analyze Resume", type="primary"):
        if cv_file is None or len(jd_text.strip()) < 30:
            st.warning("Please upload a CV (PDF) and provide a job description.")
            return
        cv_text = clean(extract_text(cv_file))
        if len(cv_text) < 100:
            st.error("No readable text found. This PDF may be scanned (image). Please use a text-based PDF.")
            return
        with st.spinner("Analyzing... (the first run downloads the AI model, please wait)"):
            res = run_analysis(cv_text, clean(jd_text))
        if res is None:
            st.error("Could not extract skills from the texts.")
            return
        st.session_state["results"] = res

    res = st.session_state.get("results")
    if res:
        show_results(res, strong, partial)


def page_comparison(strong, partial):
    st.title("⚖️ AI Model Comparison")
    res = st.session_state.get("results")
    if not res:
        st.info("Run an analysis first in the 'Resume Analysis' page.")
        return

    cov_a = coverage(res["best_a"], strong, partial)
    cov_b = coverage(res["best_b"], strong, partial)
    glob_a, glob_b = res["global_a"] * 100, res["global_b"] * 100

    fig = go.Figure()
    cats = ["Global similarity", "Skill coverage"]
    fig.add_bar(name="A: TF-IDF", x=cats, y=[glob_a, cov_a], marker_color="#94a3b8")
    fig.add_bar(name="B: Sentence Transformers", x=cats, y=[glob_b, cov_b], marker_color="#0B1F3A")
    fig.update_layout(barmode="group", yaxis_title="Score (%)", yaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

    m1, m2 = st.columns(2)
    m1.metric("TF-IDF skill coverage", f"{cov_a:.0f}%")
    m2.metric("Transformers skill coverage", f"{cov_b:.0f}%")

    df = pd.DataFrame(
        {
            "JD skill": res["jd_skills"],
            "TF-IDF sim": [round(x, 2) for x in res["best_a"]],
            "Transformers sim": [round(x, 2) for x in res["best_b"]],
            "Best CV match (B)": res["best_phrase_b"],
        }
    )
    gain = df[(df["Transformers sim"] >= partial) & (df["TF-IDF sim"] < partial)]
    st.subheader("Per-skill comparison")
    st.dataframe(df, use_container_width=True)

    st.subheader("Academic analysis")
    st.write(
        f"- **TF-IDF** measures lexical overlap: two phrases are similar only if they share the same words. "
        f"It cannot link 'ML' to 'Machine Learning' or a French phrase to its English equivalent.\n"
        f"- **Sentence Transformers** map phrases into a semantic space where similar meanings are close, "
        f"even with different words or languages.\n"
        f"- In this analysis, **{len(gain)}** skill(s) were detected semantically by Transformers "
        f"but missed by TF-IDF."
    )
    if len(gain):
        st.dataframe(gain, use_container_width=True)


def page_about():
    st.title("ℹ️ About the Project")
    st.write(
        "**SmartResume AI** compares a CV (PDF) with a job description in any field "
        "(AI, petrochemistry, medicine, engineering...). Skills are extracted dynamically "
        "by AI, without any predefined skill dictionary."
    )
    st.markdown(
        "- PyMuPDF: PDF text extraction\n"
        "- Candidate phrases + multilingual embeddings: dynamic skill extraction\n"
        "- Approach A: TF-IDF + cosine similarity (scikit-learn)\n"
        "- Approach B: Sentence Transformers + cosine similarity\n"
        "- Streamlit + Plotly: interface"
    )


st.sidebar.title("SmartResume AI")
page = st.sidebar.radio("Navigation", ["Resume Analysis", "AI Model Comparison", "About the Project"])
st.sidebar.markdown("---")
strong_t = st.sidebar.slider("Strong threshold", 0.50, 0.95, 0.75, 0.01)
partial_t = st.sidebar.slider("Partial threshold", 0.20, 0.90, 0.50, 0.01)

if page == "Resume Analysis":
    page_analysis(strong_t, partial_t)
elif page == "AI Model Comparison":
    page_comparison(strong_t, partial_t)
else:
    page_about()
