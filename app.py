from fastai import *
from fastai.vision import *
import fastai
import sys

from io import BytesIO
from typing import List, Dict, Union, ByteString, Any

import os
import flask

import requests
import torch
import json
from flask import Flask, jsonify, request, Response

app = Flask(__name__)

def load_model(path=".", model_name="modelweights.pth"):
    learn = load_learner(path, fname=model_name)
    return learn

def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image


"""
API
"""

@app.route("/predict", methods=['POST'])
def predict():
    if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(512, 385))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = model.predict(image)
            results = imagenet_utils.decode_predictions(preds)
            data["predictions"] = []

            # loop over the results and add them to the list of
            # returned predictions
            for (imagenetID, label, prob) in results[0]:
                r = {"label": label, "probability": float(prob)}
                data["predictions"].append(r)

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


"""
END API
"""

model = load_model('models')

if __name__ == "__main__":
    app.run(port=8080, debug=True)
