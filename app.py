from flask import Flask, request, jsonify
from transformers import pipeline

# Initialize the Flask app
app = Flask(__name__)

# Load a small pre-trained language model pipeline
# Using Hugging Face Transformers pipeline for text generation
text_generator = pipeline("text-generation", model="gpt2")

@app.route('/')
def home():
    return "Welcome to the MiniLLM API!"

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.get_json()
    prompt = data.get('prompt', '')
    max_length = data.get('max_length', 50)

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    try:
        # Generate text
        result = text_generator(prompt, max_length=max_length, num_return_sequences=1)
        return jsonify({"generated_text": result[0]['generated_text']})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
