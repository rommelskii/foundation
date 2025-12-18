from src.core.detector import YOLOService
from src.core.pipeline import PipelineProcessor 
from flask import Flask, jsonify, request
import numpy as np

"""
file: main.py

This file contains the routes of the back-end service.
"""

app = Flask(__name__)
model = YOLOService()

@app.route("/inference", methods=['POST'])
def inference():
    data = request.get_json(silent=True)
    if 'image_b64' not in data:
        return jsonify({"error": "malformed json, missing key/value"}), 400

    img_b64 = data.get('image_b64')
    processed_img = PipelineProcessor.Process_Frame(img_b64, model)
    
    return jsonify(processed_img), 200
     
@app.route("/status", methods=['GET'])
def status():
    return jsonify("ok"), 200

if __name__ == "__main__":
    app.run(port=8080)
