from flask import Flask, request, jsonify

from preprocess import getPreprocessedData
from torch_utils import getPrediction

app = Flask(__name__)

def allowed_file(file_name):
    ALLOWED_EXTENSIONS = {'wav'}
    return '.' in file_name and file_name.rsplit('.', 1)[1].lower() in  ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supported'})
        
        try:
            audio_bytes = file.read()
            signals = getPreprocessedData(audio_bytes=audio_bytes, mfcc=True)
            prediction = getPrediction(signals)
            return jsonify({'result': prediction})
        except:
            return jsonify({'error': 'error during prediction'})

    return jsonify({'error': 'the request was not a POST'})