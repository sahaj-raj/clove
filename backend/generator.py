import os
from dotenv import load_dotenv
from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv(override=True)

client = genai.Client()

FALLBACK_MODELS = [
    "gemini-flash-lite-latest",
    "gemini-flash-latest",
    "gemini-2.5-flash",
]

# Resilience: Automatically retries up to 3 times with exponential backoff if an API call fails
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    reraise=True
)
def _call_gemini_with_retry(model_name: str, prompt: str):
    """Makes the network call to Google Gemini with automatic retries on rate limits."""
    return client.models.generate_content_stream(
        model=model_name,
        contents=prompt
    )

def stream_gemini_responses(prompt: str):
    for model_name in FALLBACK_MODELS:
        try:
            # 1. Call Gemini through the resilient retry wrapper
            response = _call_gemini_with_retry(model_name, prompt)

            # 2. Stream tokens chunk-by-chunk
            for chunk in response:
                if chunk.text:
                    yield chunk.text
            return
        except Exception as e:
            print(f"[Resilience] {model_name} failed after retries, failing over to next model...")
            continue
            
    yield "Error: All models are currently experiencing high demand. Please try again shortly."

if __name__ == "__main__":
    test_prompt = "Explain in 2 sentences why citation anchoring prevents hallucinations in RAG."
    print("Streaming responses from gemini:\n")
    for token in stream_gemini_responses(test_prompt):
        print(token, end="", flush=True)
    print("\n")
