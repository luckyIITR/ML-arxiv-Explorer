from search_engine import search
import streamlit as st

st.title("Semantic Search on ML arXiv Papers")

query = st.text_input("Enter your query:", placeholder="e.g., attention mechanism for NLP")

if query:
    results = search(query)

    st.write("### Top Matching Papers")
    for i, row in results.iterrows():
        st.markdown(f"**Title**: {row['title']}")
        st.markdown(f"**Score**: {row['scores']:.4f}")
        st.markdown(f"**Abstract**: {row['abstract']}")
        st.markdown("---")
