import streamlit as st
import requests

def query_rag_api(base_url, temperature, k, chunk_overlap, rerank_k, index_type, manual_keywords, user_query, history_limit):
  """
  Query the Gemini RAG API with user-provided parameters
  """
  # Output container for displaying the status and results
  st.text("Checking API Health Status...")
  try:
    health_response = requests.get(f"{base_url}/health")
    health_response.raise_for_status()
    st.success(f"API Health Status: {health_response.json()}")
  except requests.exceptions.RequestException as e:
      st.error(f"API Health Check failed: {e}")
      return

  # Add manual keywords
  if manual_keywords:
    st.text("Adding manual keywords...")
    try:
      keywords_response = requests.post(
          f"{base_url}/add-keywords",
          json={"keywords": manual_keywords}
      )
      keywords_response.raise_for_status()
      st.success(f"Manual Keywords Added")
    except requests.RequestException as e:
      st.error(f"Failed to add manual keywords: {e}")

  # Set parameters
  st.text("Setting parameters...")
  try:
    params_response = requests.post(
        f"{base_url}/set-parameters",
        json={
            "temperature": temperature,
            "k": k,
            "chunk_overlap": chunk_overlap,
            "rerank_k": rerank_k,
            "index_type": index_type
        }
    )
    params_response.raise_for_status()
    st.success(f"Parameters Update: {params_response.json()}")
  except requests.RequestException as e:
    st.error(f"Failed to update parameters: {e}")
    return

  # Send query with detailed retrieval
  st.text("Sending query...")
  try:
    query_params = {"query": user_query}
    # Include history context if a limit is set
    if history_limit:
      query_params["history_context_limit"] = history_limit

    query_response = requests.post(
        f"{base_url}/query-with-details",
        params=query_params
    )
    query_response.raise_for_status()
    result = query_response.json()

    # Display the response
    st.subheader("Query Response")
    st.write("**Answer:**", result.get('answer', 'No answer returned'))

    st.subheader("Retrieved Documents")
    for i, doc in enumerate(result.get('retrieved_documents', []), 1):
      st.write(f"**Document {i}:**")
      st.write(f"Source: {doc.get('source', 'Unknown')}")
      st.write(f"Content Preview: {doc.get('content_preview', 'No preview available')}")

  except requests.RequestException as e:
    st.error(f"Query failed: {e}")

  # Retrieve conversation history
  if history_limit:
    st.text("Retrieving conversation history...")
    try:
      history_response = requests.get(f"{base_url}/conversation-history?limit={history_limit}")
      history_response.raise_for_status()
      history_data = history_response.json()

      st.subheader("Recent Conversation History")
      for entry in history_data:
        st.write(f"**Timestamp:** {entry.get('timestamp', 'N/A')}")
        st.write(f"**Query:** {entry.get('query', 'N/A')}")
        st.write(f"**Answer:** {entry.get('answer', 'N/A')}")
        st.write("---")
    except requests.RequestException as e:
      st.error(f"Failed to retrieve conversation history: {e}")


# Streamlit App
st.title("Gemini RAG API Client")

# Input fields for user-modifiable parameters
base_url = st.text_input("Base URL", "https://your-api-url.here")
temperature = st.slider("Temperature", 0.0, 1.0, 0.1)
k = st.number_input("Number of Documents to Retrieve (k): Max 20", min_value=1, max_value=20, value=5)
chunk_overlap = st.number_input("Chunk Overlap (Max 50)", min_value=0, max_value=50, value=10)
rerank_k = st.number_input("Rerank k (Max 20)", min_value=1, max_value=20, value=5)
index_type = st.selectbox("Index Type", options=["rerank", "basic"])
manual_keywords = st.text_input("Manual Keywords (comma-separated)")
user_query = st.text_area("User Query", "Enter your query, e.g., 'What are the tax regulations in Malaysia?'")
show_history = st.checkbox("Show Conversation History", value=True)

# Dynamically control history limit input
if show_history:
    history_limit = st.number_input(
        "Conversation History Limit (Max 10)",
        min_value=1,
        max_value=10,
        value=1
    )
else:
    history_limit = None  # No limit when conversation history is not shown

# Convert manual keywords from a string to a list
manual_keywords_list = [kw.strip() for kw in manual_keywords.split(",") if kw.strip()]

# Button to trigger API call
if st.button("Send Query"):
     query_rag_api(
        base_url=base_url,
        temperature=temperature,
        k=k,
        chunk_overlap=chunk_overlap,
        rerank_k=rerank_k,
        index_type=index_type,
        manual_keywords=manual_keywords_list,
        user_query=user_query,
        history_limit=history_limit
    )
