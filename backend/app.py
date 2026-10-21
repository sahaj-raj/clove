import uvicorn
from server import app

# Hugging Face Gradio Spaces execute `python app.py` on port 7860
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
