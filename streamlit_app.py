import streamlit as st
import time
from model import ChatbotModel

# Page configuration
st.set_page_config(
    page_title="Toko Kue Manis Chatbot",
    page_icon="🍰",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .stChatFloatingInputContainer {
        bottom: 20px;
    }
    .user-avatar {
        background-color: #ff4b4b;
        color: white;
        padding: 8px;
        border-radius: 50%;
    }
    .bot-avatar {
        background-color: #f0f2f6;
        color: black;
        padding: 8px;
        border-radius: 50%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Model Loading with Caching ---
# This is CRITICAL for performance. It prevents reloading the 15s model on every action.
@st.cache_resource
def get_chatbot_model():
    print("Loading model... (this should only happen once)")
    return ChatbotModel()

# Load the model (cached)
try:
    with st.spinner("Sedang menyiapkan asisten toko kue... (Mohon tunggu sebentar untuk inisialisasi pertama)"):
        chatbot = get_chatbot_model()
except Exception as e:
    st.error(f"Error initializing model: {e}")
    st.stop()

# Sidebar for Quick Actions
with st.sidebar:
    st.title("🍰 Toko Kue Manis")
    st.markdown("Selamat datang! Silakan tanya apa saja tentang kue kami.")
    
    st.subheader("Coba tanyakan:")
    quick_actions = [
        "Ada kue apa saja?",
        "Harga kue ulang tahun",
        "Cara pemesanan",
        "Metode pembayaran",
        "Area pengiriman"
    ]
    
    for action in quick_actions:
        if st.button(action, use_container_width=True):
            # Add user message to state and trigger rerun to process it
            st.session_state.messages.append({"role": "user", "content": action})
            # We don't need explicit rerun in newer Streamlit versions if we handle logic right after
            # but for button clicks to show in chat immediately, we often append to state.

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "confidence" in message:
             st.caption(f"Confidence: {message['confidence']}%")

# Chat Logic
# Check if the last message is from user and needs a response (for button clicks)
last_message_is_user = len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user"
# OR check for new input
if prompt := st.chat_input("Ketik pesan Anda..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    last_message_is_user = True

# Generate response if needed
if last_message_is_user:
    # Check if we just responded to this exact message to prevent double-firing (basic check)
    # Actually, simpler logic: if last message is user, simple as that.
    # We just need to make sure we haven't ALREADY responded to it.
    # In Streamlit, the script reruns top to bottom.
    # If we append user msg, we rerun.
    # Then we hit this block. We assume we haven't responded yet because we push response to state.
    
    # However, we need to be careful with the "button" logic above.
    # If button was clicked, we appended to messages. The script continues or reruns?
    # st.button returns True on the click run. We appended.
    # If we are in the run where button is True:
    # 1. We appended user msg.
    # 2. We render history (including new user msg).
    # 3. We hit this block: last_message_is_user is True.
    # 4. We generate response.
    # 5. We append response.
    # Perfect.

    # But wait, if we are in the run where button is True, we haven't rendered the new user msg yet in the loop above
    # because the loop runs before the button logic if button is below loop?
    # Ah, standard pattern: Place callbacks or check logic before rendering or use st.rerun().
    
    # Refined Loop Logic:
    # 1. Sidebar buttons (update state).
    # 2. Chat input (update state).
    # 3. Render all messages.
    # 4. If last message is user, generate response, append, and RERUN to show it? 
    #    Or just render it immediately using `with st.chat_message("assistant"):`.
    
    user_text = st.session_state.messages[-1]["content"]
    
    with st.chat_message("assistant"):
        with st.spinner("Mengetik..."):
            # Simulate slight delay for natural feel or just wait for model
            time.sleep(0.5) 
            
            result = chatbot.predict(user_text)
            response_text = result["response"]
            confidence = result.get("confidence", 0) * 100
            
            st.markdown(response_text)
            st.caption(f"Confidence: {confidence:.1f}%")
            
    # Add to history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response_text,
        "confidence": f"{confidence:.1f}"
    })
