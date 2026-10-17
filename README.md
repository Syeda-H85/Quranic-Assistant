# Quranic AI Assistant using RAG

This is my final assignment project. It is a chatbot that answers questions about a surah of the Quran using its English translation. It only answers from the PDF I gave it.

## How it works
1. The surah PDF is read and split into small chunks of 3 verses.
2. Each chunk is turned into numbers (embeddings) using the Gemini API.
3. When I ask a question, the question is also turned into numbers.
4. The 6 closest chunks are found.
5. Gemini reads only those chunks and writes the answer.

## Tools used
- Python
- Streamlit (for the website)
- Google Gemini API
- pypdf (to read the PDF)
- NumPy

## How to run it on your computer
1. Clone this repo
2. Make a virtual environment:
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
3. Make a file called `.env` and add your key:
```
GEMINI_API_KEY=your_key_here
```
4. Run the app:
```
streamlit run app.py
```

## Example questions
- What did Allah create humans from?
- What does this surah say about someone who stops a servant from praying?
- What is the command given at the end of the surah?

## Note
This assistant only repeats and summarizes the translation in the PDF. It does not give religious rulings. For that, please ask a scholar.

I built the basic idea and structure of this project myself. I also took some help from AI for parts of the code, and then I tested it, fixed the errors, and changed things to make it work for my surah.

Live app: [paste your Streamlit link here]

Author: Syeda Huda
Course: Python with Gen Ai