from pdf_loader import load_pdf
from embedder import chunk_text, create_faiss_index, model
from retriever import retrieve
from ollama_llm import ask_ollama

PDF_PATH = "data/sample.pdf"

print("Loading PDF...")
text = load_pdf(PDF_PATH)

chunks = chunk_text(text)
index = create_faiss_index(chunks)

print("PDF Agent ready (Ollama) 🚀")

while True:
    query = input("\nAsk a question (or 'exit'): ")
    if query.lower() == "exit":
        break

    context = retrieve(query, index, chunks, model)
    context_text = "\n".join(context)

    prompt = f"""
You are a helpful assistant.
Answer the question using ONLY the context below.

Context:
{context_text}

Question:
{query}
"""

    answer = ask_ollama(prompt)
    print("\nAnswer:\n", answer)
