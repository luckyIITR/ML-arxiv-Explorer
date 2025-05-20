import streamlit as st
from reranker.pipeline import search_with_rerank

st.set_page_config(page_title="ML ArXiv Paper Explorer", layout="wide")
st.title("📚 ML ArXiv Paper Explorer")
st.markdown("Search ML papers by meaning, not just keywords.")

query = st.text_input("🔍 Enter your query", placeholder="e.g., transformer model for text classification")

if query:
    with st.spinner("Searching and re-ranking..."):
        results = search_with_rerank(query, top_k_retrieval=20, top_n_final=5)

    for i, res in enumerate(results):
        st.markdown(f"### {i+1}. {res['title']}")
        st.markdown(f"**Score:** {res['score']:.4f}")
        st.markdown(res["abstract"])
        st.markdown("---")
