from flask import request, jsonify, Flask, render_template
import json
import keras
from os import environ
from api_model_definition import build_cnn_return_preds, losses
import numpy as np
from keras.utils.np_utils import to_categorical
import uuid
import pickle
import math
app = Flask(__name__)
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    body = json.loads(request.get_json())
    data = body.get('dataset')
    n_authors = body.get('n_authors')
    vocab_size = body.get('vocab_size')
    input_shape = (int(body.get('input_shape')),)
    model, attn_maps = build_cnn_return_preds(input_shape=input_shape, n_authors=n_authors, vocab_size=vocab_size)
    x_set = np.asarray(data.get('x'))
    y_set = np.asarray(data.get('y'))
    y_set = np.expand_dims(to_categorical(y_set), 1)
    model.fit(x_set, y_set, epochs=10, batch_size=100, shuffle='batch')
    model_id = uuid.uuid4()
    model.save(f'./user_models/{model_id}')
    return jsonify({ 'model_id': model_id })

@app.route('/predict', methods=['POST'])
def predict():
    body = json.loads(request.get_json())
    loss = losses([])
    sample = body.get('sample')
    model = keras.models.load_model(f'./user_models/{body.get("model_id")}', custom_objects={'combined_loss': loss})
    prediction = model.predict(np.asarray(sample))
    prediction = int(math.ceil(prediction.reshape(40)[0]))
    return jsonify({'prediction': prediction})

app.run(debug=True)
