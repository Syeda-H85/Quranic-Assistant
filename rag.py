import os
import re
import time
import numpy as np
from pypdf import PdfReader
from google import genai
from google.genai import types
from dotenv import load_dotenv

# load the API key from the .env file
load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-3.5-flash")


def get_client(api_key):
    # connect to Gemini using our key
    return genai.Client(api_key=api_key)


def load_chunks(pdf_path):
    # Step 1: read all the text from the PDF
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text = text + " " + page_text

    # remove extra spaces and new lines
    text = " ".join(text.split())

    if text.strip() == "":
        raise ValueError("No text found in this PDF. Use a PDF where you can select the text.")

    # Step 2: split the text at every verse number (like 1.Read or 10.a servant)
    verses = re.split(r"(?<!\d)(?=\d{1,3}\.\s?[A-Za-z˹])", text)
    verses = [v.strip() for v in verses if v.strip() != ""]

    # Step 3: join 3 verses into one chunk, and repeat 1 verse in the next chunk
    # so that the meaning is not lost between chunks
    chunks = []
    i = 0
    while i < len(verses):
        chunk = " ".join(verses[i:i + 3])
        chunks.append(chunk)
        if i + 3 >= len(verses):
            break
        i = i + 2

    return chunks


def embed(client, texts, task_type):
    # turn text into numbers (vectors) so we can compare meaning
    all_vectors = []

    for i in range(0, len(texts), 90):
        batch = texts[i:i + 90]

        # try up to 4 times in case the internet connection drops
        for attempt in range(4):
            try:
                result = client.models.embed_content(
                    model=EMBED_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type=task_type),
                )
                break
            except Exception as error:
                if attempt == 3:
                    raise error
                time.sleep(2)

        for item in result.embeddings:
            all_vectors.append(item.values)

    vectors = np.array(all_vectors, dtype="float32")

    # make every vector length 1 so the comparison is fair
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


def build_index(client, pdf_path):
    # make chunks from the PDF and turn each chunk into a vector
    chunks = load_chunks(pdf_path)
    vectors = embed(client, chunks, "RETRIEVAL_DOCUMENT")
    return chunks, vectors


def retrieve(client, question, chunks, vectors, top_k=6):
    # turn the question into a vector
    question_vector = embed(client, [question], "RETRIEVAL_QUERY")[0]

    # compare the question with every chunk (higher score = more similar)
    scores = vectors @ question_vector

    # pick the best matching chunks
    best_positions = np.argsort(scores)[::-1][:top_k]

    best_chunks = []
    for position in best_positions:
        best_chunks.append(chunks[position])
    return best_chunks


def answer(client, question, context_chunks):
    # put the best chunks together as context
    context = "\n\n".join(context_chunks)

    prompt = f"""You are a respectful Quran study assistant.
Answer the question using ONLY the context below. It comes from an English translation of a surah.
If the question asks for the main message or a summary, summarize the context in 3 or 4 short points.
If the answer is clearly not in the context, say: "I could not find this in the surah provided."
Do not give religious rulings. Keep the answer short and simple.
Context:
{context}

Question: {question}

Answer:"""

    # try up to 4 times in case the internet connection drops
    for attempt in range(4):
        try:
            response = client.models.generate_content(model=CHAT_MODEL, contents=prompt)
            return response.text
        except Exception as error:
            if attempt == 3:
                raise error
            time.sleep(2)