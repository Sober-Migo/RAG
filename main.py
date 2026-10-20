"""Entry point for local runs and Hugging Face Spaces.

Spaces (Gradio SDK) loads this file and expects a top-level `demo` object.
"""
from app.ui import create_app

demo = create_app()

if __name__ == "__main__":
    # Local / Kaggle: share optional. Spaces ignores this path's share flag
    # and serves through its own proxy — do not force share=True on Spaces.
    demo.queue(default_concurrency_limit=2)
    demo.launch(server_name="0.0.0.0", server_port=7860)
