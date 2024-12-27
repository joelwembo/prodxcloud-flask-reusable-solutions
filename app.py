from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the all-MiniLM-L6-v2 model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Example product catalog
products = [
    {"id": 1, "name": "Wireless Headphones", "description": "High-quality wireless headphones with noise cancellation."},
    {"id": 2, "name": "Gaming Laptop", "description": "Powerful gaming laptop with the latest GPU and high refresh rate display."},
    {"id": 3, "name": "Smartwatch", "description": "Feature-rich smartwatch with health tracking and notifications."},
    {"id": 4, "name": "Bluetooth Speaker", "description": "Portable Bluetooth speaker with deep bass and long battery life."},
    {"id": 5, "name": "E-book Reader", "description": "Lightweight e-book reader with a high-resolution display and weeks of battery life."},
    {"id": 6, "name": "4K TV", "description": "Ultra-HD television with vibrant colors and smart features."},
    {"id": 7, "name": "Noise-Canceling Earbuds", "description": "Compact earbuds with excellent noise isolation."},
    {"id": 8, "name": "Smartphone", "description": "High-performance smartphone with a stunning display and long battery life."},
    {"id": 9, "name": "Tablet", "description": "Portable tablet perfect for reading, gaming, and productivity."},
    {"id": 10, "name": "Mechanical Keyboard", "description": "Durable mechanical keyboard with customizable RGB lighting."},
    # Add 190 more entries
]
# Example User Behaviors
user_behaviors = [
    {"user_id": 1, "product_id": 1, "action": "viewed", "timestamp": "2024-12-27T10:00:00"},
    {"user_id": 1, "product_id": 4, "action": "purchased", "timestamp": "2024-12-27T10:30:00"},
    {"user_id": 2, "product_id": 2, "action": "viewed", "timestamp": "2024-12-27T11:00:00"},
    {"user_id": 2, "product_id": 3, "action": "viewed", "timestamp": "2024-12-27T11:05:00"},
    {"user_id": 2, "product_id": 5, "action": "purchased", "timestamp": "2024-12-27T11:20:00"},
    {"user_id": 3, "product_id": 10, "action": "rated", "rating": 4, "timestamp": "2024-12-27T11:45:00"},
    {"user_id": 3, "product_id": 9, "action": "viewed", "timestamp": "2024-12-27T11:50:00"},
    # Add more interactions
]
# Precompute embeddings for product descriptions
product_embeddings = model.encode([p["description"] for p in products])

@app.route('/')
def home():
    return "Welcome to the Product Recommendation API! Use /recommend to get product recommendations."

@app.route('/recommend', methods=['POST'])
def recommend_products():
    data = request.get_json()
    user_query = data.get('query', '')

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    try:
        # Generate embedding for the user query
        query_embedding = model.encode(user_query)

        # Compute cosine similarity between the query and product embeddings
        similarities = np.dot(product_embeddings, query_embedding) / (
            np.linalg.norm(product_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Sort products by similarity score
        recommended_indices = similarities.argsort()[::-1]
        recommendations = [{"product": products[i], "score": float(similarities[i])} for i in recommended_indices]

        return jsonify({"recommendations": recommendations[:3]})  # Return top 3 recommendations
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
