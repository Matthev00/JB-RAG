import streamlit as st

from src.config import EMBEDDING_MODEL
from src.retriever.faiss_search import FAISSRetriever

retriever = FAISSRetriever(embedding_model=EMBEDDING_MODEL)
retriever.load_index("escrcpy")

st.title("Code Search Application")

search_type = st.radio(
    "Select the type of search you want to perform:",
    ("Radius search", "Top K search"),
)

if search_type == "Radius search":
    radius = st.slider(
        "Select radius for search:", min_value=0.1, max_value=1.0, step=0.05, value=0.26
    )
    expand_query_type = st.selectbox(
        "Select query expansion type:", ("None", "WordNet", "Candidate Terms"), index=2
    )
elif search_type == "Top K search":
    top_k = st.slider(
        "Select the number of top results:", min_value=1, max_value=20, step=1, value=11
    )
    expand_query_type = st.selectbox(
        "Select query expansion type:", ("None", "WordNet", "Candidate Terms"), index=2
    )

query = st.text_input("Enter your query:")

if st.button("Search"):
    if query.strip() == "":
        st.warning("Please enter a query.")
    else:
        if search_type == "Radius search":
            expand_query = None
            if expand_query_type == "WordNet":
                expand_query = "wordnet"
            elif expand_query_type == "Candidate Terms":
                expand_query = "candidate_terms"
            results = retriever.search(
                query, radius=radius, expand_query_type=expand_query
            )
        elif search_type == "Top K search":
            expand_query = None
            if expand_query_type == "WordNet":
                expand_query = "wordnet"
            elif expand_query_type == "Candidate Terms":
                expand_query = "candidate_terms"

            results = retriever.search(
                query, top_k=top_k, expand_query_type=expand_query
            )

        if results:
            st.success(f"Found {len(results)} results:")
            for result in results:
                st.write(f"- **{result['relative_path']}**")
        else:
            st.warning("No results found.")
