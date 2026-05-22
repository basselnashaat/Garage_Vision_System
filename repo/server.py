"""
Hugging Face Spaces entrypoint.

Spaces expect `app.py` at the repository root. It re-exports the Gradio demo
defined in app_gradio.py (same UI, GRADIO_SPACE env already set there).
"""

from app_gradio import demo

if __name__ == "__main__":
    demo.launch()
