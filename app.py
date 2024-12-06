import streamlit as st
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

executor = ThreadPoolExecutor(max_workers=3)

def async_request(url, json_data=None, method="POST", timeout=240):
  if method == "POST":
    return requests.post(url, json=json_data, timeout=timeout)
  return requests.get(url, timeout=timeout) 

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

  # Set parameters with improved error handling
  st.text("Setting parameters...")
  try:
    # Use a shorter timeout for initial request
    params_response = requests.post(
      f"{base_url}/set-parameters",
      json={
        "temperature": temperature,
        "k": k,
        "chunk_overlap": chunk_overlap,
        "rerank_k": rerank_k,
        "index_type": index_type
      },
      timeout=60  # Shorter, more reasonable timeout
    )
    params_response.raise_for_status()
    st.success(f"Parameters Update: {params_response.json()}")
  except requests.Timeout:
    st.warning("Parameter setting timed out. Retrying with longer timeout...")
    try:
      # Retry with a much longer timeout
      params_response = requests.post(
        f"{base_url}/set-parameters",
        json={
          "temperature": temperature,
          "k": k,
          "chunk_overlap": chunk_overlap,
          "rerank_k": rerank_k,
          "index_type": index_type
        },
        timeout=240  # Extended timeout
      )
      params_response.raise_for_status()
      st.success(f"Parameters Update: {params_response.json()}")
    except requests.RequestException as e:
      st.error(f"Failed to update parameters after retry: {e}")
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

# index_type and rerank_k
index_type = st.selectbox("Index Type", options=["rerank", "basic"])
if index_type == "rerank":
  rerank_k = st.number_input("Rerank k (Max 20)", min_value=1, max_value=20, value=5)
else:
  rerank_k = None # No rerank_k is needed for basic indexing
  st.info("Rerank k is not applicable for basic indexing.")

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
