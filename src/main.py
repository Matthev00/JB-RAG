import streamlit as st

from src.app_utils.answer_summary import generate_summary
from src.config import EMBEDDING_MODEL
from src.retriever.faiss_search import FAISSRetriever


def main():
    """
    Streamlit application for code search with optional LLM summarization.
    To run this script 'streamlit run src/main.py --server.fileWatcherType=none'
    """
    retriever = FAISSRetriever(embedding_model=EMBEDDING_MODEL)
    retriever.load_index("escrcpy")

    st.title("Code Search Application")

    search_type = st.radio(
        "Select the type of search you want to perform:",
        ("Radius search", "Top K search"),
    )

    if search_type == "Radius search":
        radius = st.slider(
            "Select radius for search:",
            min_value=0.1,
            max_value=1.0,
            step=0.05,
            value=0.26,
        )
        expand_query_type = st.selectbox(
            "Select query expansion type:",
            ("None", "WordNet", "Candidate Terms"),
            index=2,
        )
    elif search_type == "Top K search":
        top_k = st.slider(
            "Select the number of top results:",
            min_value=1,
            max_value=20,
            step=1,
            value=11,
        )
        expand_query_type = st.selectbox(
            "Select query expansion type:",
            ("None", "WordNet", "Candidate Terms"),
            index=2,
        )

    query = st.text_input("Enter your query:")
    use_llm = st.checkbox(
        "🤖 Use LLM explanation",
        value=False,
        help="Generates a natural language summary",
    )

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
                    st.markdown(f"- **{result['relative_path']}**")

                if use_llm:
                    with st.spinner("Generating explanation with LLM..."):
                        explanation = generate_summary(query, results)
                        st.markdown("### 🤖 LLM Explanation")
                        st.info(explanation)
            else:
                st.warning("No results found.")


if __name__ == "__main__":
    main()
