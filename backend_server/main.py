import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables
load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
HF_MODEL_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"

# Access HF_TOKEN from environment
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("No HF_TOKEN found in environment variables!")

# Pydantic model for request validation
class Question(BaseModel):
    question: str

@app.get("/")
async def home():
    return {"message": "Hello from the Hugging Face LLaMA backend from Aaryan Purohit!"}

@app.post("/api/ask")
async def ask_llama(question_data: Question):
    """
    This endpoint receives a question and returns an answer from the LLaMA model.
    """
    try:
        # Construct the prompt
        prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
            "You are a helpful and smart assistant. You accurately provide an answer "
            "to the provided user query.<|eot_id|><|start_header_id|>user<|end_header_id|>"
            f" Here is the query: ```{question_data.question}```. Provide a precise and concise answer."
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        )

        parameters = {
            "max_new_tokens": 500,
            "temperature": 0.01,
            "top_k": 50,
            "top_p": 0.95,
            "return_full_text": False
        }

        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        }

        payload = {
            "inputs": prompt,
            "parameters": parameters
        }

        # Make the POST request to the Hugging Face Inference API
        response = requests.post(HF_MODEL_URL, headers=headers, json=payload)
        response_json = response.json()

        # Check for errors from the HF API
        if isinstance(response_json, dict) and response_json.get("error"):
            raise HTTPException(status_code=500, detail=response_json["error"])

        # Extract the generated text from the first object
        generated_text = response_json[0]["generated_text"].strip()

        return {"answer": generated_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
