import streamlit as st
import requests
import pandas as pd

# --- CONFIG ---
API_URL = "http://127.0.0.1:8001"
st.set_page_config(page_title="Project Chimpanzee", layout="wide")

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/b/b9/Joe_Rogan_Experience_logo.jpg", width=200)
    st.title("The JRE Brain")
    st.markdown("Powered by **Llama 3.2**, **Neo4j**, and **LanceDB**.")
    
    st.divider()
    st.subheader("System Status")
    try:
        status = requests.get(f"{API_URL}/").json()
        st.success("API: Online 🟢")
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
        res = requests.post(f"{API_URL}/search/graph", json={"entity": entity, "limit": 10})
        if res.status_code == 200:
            df = pd.DataFrame(res.json()["results"])
            if not df.empty:
                st.dataframe(df)
            else:
                st.warning("No connections found.")