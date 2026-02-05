import gradio as gr
import os
from huggingface_hub import InferenceClient
from app.pdf_loader import load_pdf
from app.embedder import chunk_text, create_faiss_index, model
from app.retriever import retrieve

client = InferenceClient(api_key=os.getenv("HF_TOKEN"))
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

def process_pdf(pdf, question):
    if not pdf:
        return "⚠️ Please upload a PDF file first."
    if not question.strip():
        return "⚠️ Please enter a question."
    
    try:
        text = load_pdf(pdf.name)
        chunks = chunk_text(text)
        index = create_faiss_index(chunks)
        context = "\n".join(retrieve(question, index, chunks, model))

        prompt = f"Use the context to answer.\nContext: {context}\nQuestion: {question}"
        
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error: {str(e)}"

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

demo.launch(css=css, theme=gr.themes.Soft())