import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/search"

st.set_page_config(page_title="Patent Search Dashboard", layout="wide")

st.title("🔍 Patent Search Dashboard")
st.markdown("Search for patents using natural language queries. Powered by ChromaDB and FastAPI.")

query = st.text_input("Enter your search query", placeholder="e.g., machine learning in agriculture")

if st.button("Search") and query.strip():
    with st.spinner("Searching..."):
        try:
            response = requests.post(API_URL, json={"query": query.strip()})
            response.raise_for_status()
            results = response.json()

            if not results:
                st.warning("No results found.")
            else:
                for i, result in enumerate(results, start=1):
                    with st.container():
                        st.subheader(f"🔹 Result {i}")
                        st.markdown(f"**Title:** {result['title']}")
                        st.markdown(f"**Patent Number:** `{result['patent_number']}`")
                        st.markdown(f"**Quality Score:** `{round(result['quality_score'], 4)}`")
                        with st.expander("🔎 Source Info"):
                            st.json(result["source_info"])
                        st.markdown("---")

        except Exception as e:
            st.error(f"❌ Failed to retrieve results: {e}")
else:
    st.info("Enter a search query and click 'Search' to begin.")

