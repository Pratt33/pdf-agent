import gradio as gr
import os
import asyncio
from huggingface_hub import InferenceClient
from app.pdf_loader import load_pdf
from app.embedder import chunk_text, create_faiss_index, model, warmup_model
from app.retriever import retrieve
from app.cache import pdf_cache

client = InferenceClient(api_key=os.getenv("HF_TOKEN"))
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

async def process_pdf(pdf, question):
    if not pdf:
        yield "⚠️ Please upload a PDF file first."
        return
    if not question.strip():
        yield "⚠️ Please enter a question."
        return
    
    try:
        # Check cache first
        pdf_hash = pdf_cache.get_hash(pdf.name)
        cached_data = pdf_cache.get(pdf_hash)
        
        if cached_data:
            yield "✓ Using cached PDF data...\n"
            chunks = cached_data['chunks']
            index = cached_data['index']
        else:
            # Progressive loading with status updates
            yield "📄 Loading PDF...\n"
            text = await asyncio.to_thread(load_pdf, pdf.name)
            
            yield "✂️ Chunking text...\n"
            chunks = await asyncio.to_thread(chunk_text, text)
            
            yield "🔢 Creating embeddings (batch processing)...\n"
            index = await asyncio.to_thread(create_faiss_index, chunks)
            
            # Cache the processed data
            pdf_cache.set(pdf_hash, {'chunks': chunks, 'index': index})
            yield "💾 PDF cached for faster future queries...\n"
        
        yield "🔍 Retrieving relevant context...\n"
        context_chunks = await asyncio.to_thread(retrieve, question, index, chunks, model)
        context = "\n".join(context_chunks)

        prompt = f"Use the context to answer.\nContext: {context}\nQuestion: {question}"
        
        yield "🤖 Generating answer...\n"
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        
        # Return final answer
        yield response.choices[0].message.content
        
    except Exception as e:
        yield f"❌ Error: {str(e)}"

# Custom CSS for better styling
css = """
.gradio-container {
    font-family: 'Arial', sans-serif;
}
#title {
    text-align: center;
    font-size: 2.5em;
    font-weight: bold;
    margin-bottom: 10px;
}
#description {
    text-align: center;
    font-size: 1.1em;
    color: #666;
    margin-bottom: 20px;
}
"""

# Create the UI using Blocks for better layout
with gr.Blocks() as demo:
    gr.Markdown(
        """
        <div id="title">📄 PDF Question Answering Agent</div>
        <div id="description">Upload a PDF and ask questions - powered by RAG & LLaMA 3.2</div>
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(
                label="📎 Upload PDF Document",
                file_types=[".pdf"],
                type="filepath"
            )
            question_input = gr.Textbox(
                label="❓ Your Question",
                placeholder="What would you like to know about this document?",
                lines=3
            )
            submit_btn = gr.Button("🔍 Get Answer", variant="primary", size="lg")
            clear_btn = gr.Button("🔄 Clear", size="sm")
        
        with gr.Column(scale=1):
            output = gr.Textbox(
                label="💡 Answer",
                placeholder="Your answer will appear here...",
                lines=10
            )
    
    gr.Markdown(
        """
        ### 💡 Tips:
        - Upload any PDF document (research papers, reports, books, etc.)
        - Ask specific questions about the content
        - The model uses RAG (Retrieval Augmented Generation) for accurate answers
        - **Cache enabled**: Same PDF = instant responses! (stores last 2 PDFs)
        """
    )
    
    # Examples
    gr.Examples(
        examples=[
            ["What is the main topic of this document?"],
            ["Summarize the key findings."],
            ["What are the conclusions?"]
        ],
        inputs=question_input
    )
    
    # Event handlers
    submit_btn.click(
        fn=process_pdf,
        inputs=[pdf_input, question_input],
        outputs=output
    )
    
    clear_btn.click(
        fn=lambda: (None, "", ""),
        outputs=[pdf_input, question_input, output]
    )

# Pre-warm models on startup
print("🔥 Warming up models...")
warmup_model()
print("✓ Models ready!")

demo.launch(css=css, theme=gr.themes.Soft())