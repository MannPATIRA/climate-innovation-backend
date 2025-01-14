import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

HF_MODEL_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"

# 3) Access HF_TOKEN from environment
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    # Optional: raise an error or warning if not found
    raise ValueError("No HF_TOKEN found in environment variables!")
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Hello from the Hugging Face LLaMA backend!"})

@app.route('/api/ask', methods=['POST'])
def ask_llama():
    """
    This endpoint receives a JSON body with 'question' 
    and returns a JSON response with 'answer'.
    """
    try:
        data = request.get_json()
        user_question = data.get('question', None)

        if not user_question:
            return jsonify({"error": "No question provided"}), 400

        # Construct the prompt the same way the article tutorial showed:
        # Here’s an example prompt that sets up system and user messages:
        prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
            "You are a helpful and smart assistant. You accurately provide an answer "
            "to the provided user query.<|eot_id|><|start_header_id|>user<|end_header_id|>"
            f" Here is the query: ```{user_question}```. Provide a precise and concise answer."
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        )

        # You can customize parameters:
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
            return jsonify({"error": response_json["error"]}), 500

        # Typically, the HF API returns an array. Extract the generated text from the first object:
        generated_text = response_json[0]["generated_text"].strip()

        return jsonify({"answer": generated_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
