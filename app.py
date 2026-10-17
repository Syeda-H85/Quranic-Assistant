import os
import sys
import glob
import subprocess
import streamlit as st
from streamlit.runtime import exists
from dotenv import load_dotenv
from rag import get_client, build_index, retrieve, answer

# If we press the play button in VS Code, start the website automatically
if not exists():
    subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
    sys.exit()

load_dotenv()

# page title and heading
st.set_page_config(page_title="Quranic AI Assistant", page_icon="📖")
st.title("📖 Quranic AI Assistant")
st.caption("Ask questions about a surah. Answers come only from the PDF.")


def get_api_key():
    # on Streamlit Cloud the key is in secrets, on my laptop it is in .env
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.getenv("GEMINI_API_KEY")


api_key = get_api_key()
if not api_key:
    st.error("GEMINI_API_KEY not found. Add it to the .env file.")
    st.stop()

client = get_client(api_key)

# find all PDFs inside the data folder
pdfs = sorted(glob.glob("data/*.pdf"))
if len(pdfs) == 0:
    st.error("Put a surah PDF inside the data folder.")
    st.stop()


def nice_name(path):
    # data/surah_alaq.pdf  ->  Surah Alaq
    name = os.path.basename(path)
    name = name.replace(".pdf", "").replace("_", " ")
    return name.title()


pdf_path = st.sidebar.selectbox("Choose surah", pdfs, format_func=nice_name)


@st.cache_resource(show_spinner="Reading the PDF...")
def load_index(path):
    # this runs only once, then Streamlit remembers the result
    return build_index(client, path)


chunks, vectors = load_index(pdf_path)

# sample questions in the sidebar
st.sidebar.subheader("Sample questions")
samples = [
    "What is the main message of this surah?",
    "What did Allah create humans from?",
    "What does this surah say about someone who stops a servant from praying?",
    "What is the command given at the end of the surah?",
]

clicked_question = None
for sample in samples:
    if st.sidebar.button(sample):
        clicked_question = sample

# keep the chat history so old messages stay on the screen
if "history" not in st.session_state:
    st.session_state.history = []

for item in st.session_state.history:
    with st.chat_message("user"):
        st.write(item["question"])
    with st.chat_message("assistant"):
        st.write(item["answer"])
        with st.expander("Sources from the PDF"):
            for source in item["sources"]:
                st.write(source)
                st.divider()

# the question comes from the chat box or from a sample button
question = st.chat_input("Ask a question about the surah...")
if question is None:
    question = clicked_question

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            sources = retrieve(client, question, chunks, vectors)
            final_answer = answer(client, question, sources)

        st.write(final_answer)
        with st.expander("Sources from the PDF"):
            for source in sources:
                st.write(source)
                st.divider()

    st.session_state.history.append(
        {"question": question, "answer": final_answer, "sources": sources}
    )