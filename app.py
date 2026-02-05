import gradio as gr
import os

from app.pdf_loader import load_pdf
from app.embedder import chunk_text, create_faiss_index, model
from app.retriever import retrieve
from app.llm.hf_llm import HuggingFaceLLM

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = "google/flan-t5-large"  # Reliable and lightweight

llm = HuggingFaceLLM(HF_TOKEN, HF_MODEL)

def process_pdf(pdf, question):
    text = load_pdf(pdf.name)
    chunks = chunk_text(text)
    index = create_faiss_index(chunks)

    context = retrieve(question, index, chunks, model)
    context_text = "\n".join(context)

    prompt = f"""
Answer the question using only the context below.

Context:
{context_text}

Question:
{question}
"""

    return llm.generate(prompt)

interface = gr.Interface(
    fn=process_pdf,
    inputs=[
        gr.File(label="Upload PDF"),
        gr.Textbox(label="Question")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="📄 PDF Question Answering Agent",
    description="Lightweight RAG-based PDF agent"
)

interface.launch()
