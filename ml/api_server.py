"""
Flask API Server for Crime Classification Model
Loads the trained DistilBERT model and exposes REST API endpoint
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from utils.preprocess import clean_text

app = Flask(__name__)
CORS(app)  # Allow React frontend to call this API

# =========================
# Load Model at Startup
# =========================
print("Loading model...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_model")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "datasets", "label_map.json")

# Check if model exists
if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError(f"Model not found at {MODEL_DIR}")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()  # Set to evaluation mode

# Load label mapping
with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    label_map = json.load(f)

print(f"✅ Model loaded successfully on {device}")
print(f"✅ {len(label_map['label_to_name'])} categories available")


# =========================
# API Endpoints
# =========================

@app.route('/api/classify', methods=['POST'])
def classify_complaint():
    """
    Classify a complaint text into one of 12 crime categories
    
    Request Body:
    {
        "complaint": "text of the complaint"
    }
    
    Response:
    {
        "success": true,
        "category": "theft",
        "category_full": "Theft & Robbery (BNS 303–309)",
        "confidence": 95.6
    }
    """
    try:
        # Get complaint text from request
        data = request.get_json()
        complaint_text = data.get('complaint', '')
        
        if not complaint_text or complaint_text.strip() == '':
            return jsonify({
                'success': False,
                'error': 'Complaint text is required'
            }), 400
        
        # Clean the text
        cleaned_text = clean_text(complaint_text)
        
        # Tokenize
        encoding = tokenizer(
            cleaned_text,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        )
        
        # Move to device
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            prediction = torch.argmax(outputs.logits, dim=1).item()
            probabilities = torch.softmax(outputs.logits, dim=1)[0]
            confidence = probabilities[prediction].item()
        
        # Get category names
        category_short = label_map["label_to_name"][str(prediction)]
        category_full = label_map["name_to_full"][category_short]
        
        # Return result
        return jsonify({
            'success': True,
            'category': category_short,
            'category_full': category_full,
            'confidence': round(confidence * 100, 2),
            'label_id': prediction
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Check if API is running"""
    return jsonify({
        'status': 'ok',
        'message': 'Crime Classification API is running',
        'device': str(device),
        'num_categories': len(label_map['label_to_name'])
    })


@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get list of all available crime categories"""
    return jsonify({
        'success': True,
        'categories': label_map
    })


# =========================
# Run Server
# =========================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Crime Classification API Server")
    print("="*60)
    print("Server running on: http://localhost:5000")
    print("\nAvailable endpoints:")
    print("  POST /api/classify    - Classify complaint text")
    print("  GET  /api/health      - Health check")
    print("  GET  /api/categories  - List all categories")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
