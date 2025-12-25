# Import the necessary libraries
import streamlit as st
import PyPDF2
import io
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from pypdf import PdfReader, PdfWriter
from google import genai

# --- 1. Page Configuration and Title ---
st.set_page_config(page_title="NuAi Chatbot", page_icon="🧠", layout="wide")
st.title("🤖 NuAi")
st.caption("Chatbot cerdas dengan kemampuan analisis dokumen menggunakan LangGraph dan Gemini.")

# --- 2. Sidebar for Settings ---
with st.sidebar:
    st.subheader("Settings")
    google_api_key = st.text_input("Google AI API Key", type="password")
    reset_button = st.button("Reset Conversation", help="Clear all messages and start fresh")

# --- BARU: Fungsi untuk Memproses Dokumen ---
@st.cache_data # Gunakan cache agar tidak memproses ulang file yang sama
def process_document(uploaded_file):
    """Mengekstrak teks dari file yang diunggah (PDF, DOCX, TXT)."""
    text = ""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension == "pdf":
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.getvalue()))
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif file_extension == "docx":
            doc = docx.Document(io.BytesIO(uploaded_file.getvalue()))
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif file_extension == "txt":
            text = uploaded_file.getvalue().decode("utf-8")
        
        st.success(f"Dokumen '{uploaded_file.name}' berhasil diproses!")
        return text
    except Exception as e:
        st.error(f"Gagal memproses file: {e}")
        return None

# --- 3. API Key and Agent Initialization ---
if not google_api_key:
    st.info("Please add your Google AI API key in the sidebar to start chatting.", icon="🗝️")
    st.stop()

if ("agent" not in st.session_state) or (getattr(st.session_state, "_last_key", None) != google_api_key):
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            google_api_key=google_api_key,
            temperature=0.7,
    )
        )
        st.session_state.agent = create_react_agent(
            model=llm,
            tools=[],
           
            prompt="You are a helpful, friendly assistant. Respond concisely and clearly. If context from a document is provided, you MUST base your answer on that context."
        )
        st.session_state._last_key = google_api_key
        st.session_state.pop("messages", None)
        st.session_state.pop("document_context", None) # <-- BARU: Hapus konteks dokumen juga
    except Exception as e:
        st.error(f"Invalid API Key or configuration error: {e}")
        st.stop()

# --- 4. Chat History and Document Context Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_context" not in st.session_state: 
    st.session_state.document_context = None 

if reset_button:
    st.session_state.pop("agent", None)
    st.session_state.pop("messages", None)
    st.session_state.pop("document_context", None) 
    st.rerun()

# --- 5. Display Past Messages ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


uploaded_file = st.file_uploader(
    "Unggah dokumen untuk dianalisis (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt"]
)

if uploaded_file is not None:
    # Proses file dan simpan konteksnya di session state
    # Ini hanya akan berjalan jika file baru diunggah
    if st.session_state.document_context is None: 
        with st.spinner(f"Menganalisis '{uploaded_file.name}'..."):
            st.session_state.document_context = process_document(uploaded_file)
            # Menambahkan pesan sistem setelah dokumen diproses
            if st.session_state.document_context:
                system_message = f"Dokumen '{uploaded_file.name}' siap dianalisis. Silakan ajukan pertanyaan Anda."
                st.session_state.messages.append({"role": "assistant", "content": system_message})
                st.rerun() # Muat ulang untuk menampilkan pesan baru

# --- 6. Handle User Input and Agent Communication ---
prompt = st.chat_input("Tanyakan sesuatu...") # <-- DIMODIFIKASI: Teks placeholder diubah

if prompt:
    # 1. Add and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- DIMODIFIKASI: Sisipkan Konteks Dokumen ke dalam Prompt ---
    final_prompt = prompt
    if st.session_state.document_context:
        # Jika ada konteks, gabungkan dengan prompt pengguna
        final_prompt = f"""
        Berdasarkan konteks dokumen berikut, jawablah pertanyaan pengguna.
        ---
        Konteks Dokumen:
        {st.session_state.document_context}
        ---
        Pertanyaan Pengguna: {prompt}
        """
    # -----------------------------------------------------------

    # 3. Get the assistant's response.
    try:
        # Format pesan untuk LangChain
        messages = []
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                # Gunakan final_prompt untuk pesan pengguna terakhir
                if msg["content"] == prompt:
                     messages.append(HumanMessage(content=final_prompt))
                else:
                     messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        # Panggil agent dengan pesan yang sudah dimodifikasi
        with st.spinner("NuAi sedang berpikir..."):
            response = st.session_state.agent.invoke({"messages": messages})
        
        if "messages" in response and len(response["messages"]) > 0:
            answer = response["messages"][-1].content
        else:
            answer = "Maaf, saya tidak dapat memberikan respons."

    except Exception as e:
        answer = f"Terjadi kesalahan: {e}"

    # 4. Display the assistant's response.
    with st.chat_message("assistant"):
        st.markdown(answer)
    
    # 5. Add the assistant's response to the message history list.
    st.session_state.messages.append({"role": "assistant", "content": answer})
