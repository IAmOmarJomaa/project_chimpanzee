import streamlit as st
import requests
import pandas as pd
import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8001")

st.set_page_config(page_title="Project Chimpanzee", layout="wide")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "The_Joe_Rogan_Experience_logo.png")

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR ---
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=200)
    else:
        st.warning("Logo file not found.")
        
    st.title("The JRE Brain")
    st.markdown("Powered by **Llama 3.2**, **Neo4j**, and **LanceDB**.")
    
    st.divider()
    st.subheader("System Status")
    try:
        # Simple health check
        status = requests.get(f"{API_URL}/", timeout=2)
        if status.status_code == 200:
            st.success("API: Online 🟢")
        else:
            st.warning(f"API: Unstable ({status.status_code}) 🟡")
    except:
        st.error("API: Offline 🔴")
        st.stop()

# --- MAIN CHAT ---
st.title("Virtual Joe Rogan 🎙️")

# Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("Ask Joe anything..."):
    # 1. Show User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Get AI Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Pulling up that clip..."):
            try:
                # Prepare payload
                payload = {
                    "query": prompt,
                    "history": st.session_state.messages[:-1] # Send context
                }
                
                # Call API
                response = requests.post(f"{API_URL}/chat", json=payload)
                data = response.json()
                
                ai_text = data.get("response", "Error: No response")
                sources = data.get("sources", [])
                
                # Display Answer
                message_placeholder.markdown(ai_text)
                
                # Save to History
                st.session_state.messages.append({"role": "assistant", "content": ai_text})
                
                # 3. Show Sources (Expandable)
                with st.expander("📚 Sources / Evidence"):
                    st.write("I read these chunks to answer you:")
                    st.json(sources)
                    
            except Exception as e:
                message_placeholder.error(f"Connection Error: {e}")

# --- GRAPH SEARCH TAB (Bonus) ---
with st.expander("🕸️ Explore the Knowledge Graph"):
    entity = st.text_input("Enter an Entity (e.g., Elon, Aliens):")
    if st.button("Search Graph"):
        try:
            res = requests.post(f"{API_URL}/search/graph", json={"entity": entity, "limit": 10})
            if res.status_code == 200:
                results = res.json().get("results", [])
                if results:
                    df = pd.DataFrame(results)
                    st.dataframe(df)
                else:
                    st.warning("No connections found.")
            else:
                st.error("Graph search failed.")
        except Exception as e:
            st.error(f"Error: {e}")