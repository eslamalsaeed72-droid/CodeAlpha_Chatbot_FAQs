# CELL 9 – Streamlit English FAQ chatbot

st.set_page_config(page_title="English FAQ Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 English FAQ Chatbot")
st.markdown("Semantic FAQ assistant powered by **Sentence-BERT (all-MiniLM-L6-v2)**.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

col1, col2 = st.columns([0.85, 0.15])
with col1:
    user_input = st.text_input("Ask a question (English only):", placeholder="How can I track my order?")
with col2:
    send = st.button("Send", use_container_width=True)

if send and user_input.strip():
    result = matcher.get_intelligent_answer(user_input.strip())
    st.session_state.chat_history.append(
        {
            "user": user_input.strip(),
            "answer": result["answer"],
            "confidence": result["confidence"],
            "tone": result["tone"],
            "intent": result["analysis"]["intent"]["type"],
        }
    )

st.markdown("---")

for msg in reversed(st.session_state.chat_history):
    st.markdown(f"**You:** {msg['user']}")
    st.markdown(f"**Bot:** {msg['answer']}")
    cols = st.columns(3)
    cols[0].metric("Confidence", f"{msg['confidence']:.1%}")
    cols[1].metric("Tone", msg["tone"].title())
    cols[2].metric("Intent", msg["intent"])

st.markdown("---")
st.caption("English FAQ Chatbot · Sentence-BERT all-MiniLM-L6-v2")
