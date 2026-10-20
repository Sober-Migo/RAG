"""Gradio UI for the RAG PDF Q&A app."""
import gradio as gr

from app.chain import answer_question
from app.config import SERVER_NAME, SERVER_PORT


def create_app() -> gr.Blocks:
    with gr.Blocks(title="RAG PDF Q&A") as demo:
        gr.Markdown(
            """
            # RAG PDF Q&A
            Upload a PDF and ask questions grounded in the document.

            **Stack:** Local ONNX embeddings · Chroma · LangChain · **Groq** · Gradio

            *Tip:* Set `GROQ_API_KEY` in Space secrets for generated answers.
            Without it, the app returns the most relevant passages (extractive mode).
            """
        )
        with gr.Row():
            pdf_input = gr.File(
                label="Upload PDF",
                file_count="single",
                file_types=[".pdf"],
                type="filepath",
            )
        query_input = gr.Textbox(
            label="Question",
            lines=2,
            placeholder="Ask anything about the uploaded document…",
        )
        submit_btn = gr.Button("Ask", variant="primary")

        gr.Markdown("### Answer")
        answer_output = gr.Markdown(
            value="*Answer will appear here.*",
            elem_classes=["rag-answer"],
        )

        submit_btn.click(
            fn=answer_question,
            inputs=[pdf_input, query_input],
            outputs=answer_output,
        )
        query_input.submit(
            fn=answer_question,
            inputs=[pdf_input, query_input],
            outputs=answer_output,
        )
    return demo


def launch(share: bool = False) -> None:
    app = create_app()
    app.queue(default_concurrency_limit=2)
    app.launch(server_name=SERVER_NAME, server_port=SERVER_PORT, share=share)
