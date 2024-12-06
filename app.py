import streamlit as st
import requests

# Default Parameters
DEFAULT_PARAMETERS = {
    'base_url': 'https://your-api-url.here',
    'temperature': 0.1,
    'k': 5,
    'chunk_overlap': 10,
    'index_type': 'basic',
    'rerank_k': None
}

def check_api_health(base_url):
    """Check API health status"""
    try:
        health_response = requests.get(f"{base_url}/health", timeout=30)
        health_response.raise_for_status()
        return health_response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Health Check failed: {e}")
        return None

def set_api_parameters(base_url, params):
    """Set API parameters"""
    try:
        response = requests.post(
            f"{base_url}/set-parameters",
            json={
                "temperature": params['temperature'],
                "k": params['k'],
                "chunk_overlap": params['chunk_overlap'],
                "rerank_k": params['rerank_k'],
                "index_type": params['index_type']
            },
            timeout=240
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to set parameters: {e}")
        return None

def add_manual_keywords(base_url, keywords):
    """Add manual keywords"""
    if not keywords:
        return None
    try:
        response = requests.post(
            f"{base_url}/add-keywords",
            json={"keywords": keywords},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to add keywords: {e}")
        return None

def query_rag_api(base_url, user_query, manual_keywords, history_limit):
    """Execute RAG query"""
    try:
        query_params = {"query": user_query}
        if history_limit:
            query_params["history_context_limit"] = history_limit

        query_response = requests.post(
            f"{base_url}/query-with-details",
            params=query_params,
            timeout=240
        )
        query_response.raise_for_status()
        return query_response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Query failed: {e}")
        return None

def get_conversation_history(base_url, history_limit):
    """Retrieve conversation history"""
    if not history_limit:
        return None
    try:
        history_response = requests.get(
            f"{base_url}/conversation-history?limit={history_limit}",
            timeout=30
        )
        history_response.raise_for_status()
        return history_response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to retrieve conversation history: {e}")
        return None

def main():
    st.title("Gemini RAG API Client")

    # Sidebar for Parameters
    with st.sidebar:
        st.header("API Configuration")
        base_url = st.text_input("Base URL", DEFAULT_PARAMETERS['base_url'])
        
        st.subheader("Retrieval Parameters")
        temperature = st.slider(
            "Temperature", 
            0.0, 1.0, 
            DEFAULT_PARAMETERS['temperature']
        )
        k = st.number_input(
            "Documents to Retrieve (k)", 
            min_value=1, max_value=20, 
            value=DEFAULT_PARAMETERS['k']
        )
        chunk_overlap = st.number_input(
            "Chunk Overlap", 
            min_value=0, max_value=50, 
            value=DEFAULT_PARAMETERS['chunk_overlap']
        )
        
        index_type = st.selectbox(
            "Index Type", 
            options=["basic", "rerank"], 
            index=0
        )
        
        rerank_k = None
        if index_type == "rerank":
            rerank_k = st.number_input(
                "Rerank k", 
                min_value=1, max_value=20, 
                value=5
            )
        
        # Submit Parameters Button
        params_submitted = st.button("Submit Parameters")
        
        # Store submitted parameters in session state
        if params_submitted:
            st.session_state.submitted_params = {
                'base_url': base_url,
                'temperature': temperature,
                'k': k,
                'chunk_overlap': chunk_overlap,
                'index_type': index_type,
                'rerank_k': rerank_k
            }
            
            # Attempt to set API parameters
            health_status = check_api_health(base_url)
            if health_status:
                set_status = set_api_parameters(base_url, st.session_state.submitted_params)
                if set_status:
                    st.sidebar.success("Parameters successfully submitted!")

    # Main Page Content
    if 'submitted_params' in st.session_state:
        st.header("Query RAG API")
        
        manual_keywords = st.text_input("Manual Keywords (comma-separated)")
        manual_keywords_list = [kw.strip() for kw in manual_keywords.split(",") if kw.strip()]
        
        user_query = st.text_area("User Query", "Enter your query")
        
        show_history = st.checkbox("Show Conversation History", value=False)
        history_limit = st.number_input(
            "Conversation History Limit", 
            min_value=1, max_value=10, 
            value=1
        ) if show_history else None
        
        if st.button("Execute Query"):
            # Add manual keywords if provided
            if manual_keywords_list:
                add_manual_keywords(
                    st.session_state.submitted_params['base_url'], 
                    manual_keywords_list
                )
            
            # Execute query
            result = query_rag_api(
                st.session_state.submitted_params['base_url'], 
                user_query, 
                manual_keywords_list, 
                history_limit
            )
            
            if result:
                st.subheader("Query Response")
                st.write("**Answer:**", result.get('answer', 'No answer'))
                
                st.subheader("Retrieved Documents")
                for i, doc in enumerate(result.get('retrieved_documents', []), 1):
                    st.write(f"**Document {i}:**")
                    st.write(f"Source: {doc.get('source', 'Unknown')}")
                    st.write(f"Content Preview: {doc.get('content_preview', 'No preview')}")
            
            # Retrieve conversation history if enabled
            if show_history:
                history = get_conversation_history(
                    st.session_state.submitted_params['base_url'], 
                    history_limit
                )
                if history:
                    st.subheader("Conversation History")
                    for entry in history:
                        st.write(f"**Timestamp:** {entry.get('timestamp', 'N/A')}")
                        st.write(f"**Query:** {entry.get('query', 'N/A')}")
                        st.write(f"**Answer:** {entry.get('answer', 'N/A')}")
                        st.write("---")
    else:
        st.info("Please submit parameters in the sidebar before proceeding.")

if __name__ == "__main__":
    main()
