

import sys
from base64 import b64decode
from keras.applications import imagenet_utils
from io import BytesIO
from typing import List, Dict, Union, ByteString, Any
import os
import flask
import re
import requests
import torch
import json
import numpy
from flask import Flask, jsonify, request, Response
from src.model import Model

import torchvision.transforms as transforms
from fastai.vision import *

from PIL import Image
app = Flask(__name__)

def load_model(path="C:\\Users\\zhiji\\Documents\\Projects\\ImplementAI\\Image-Classifier\\data\\models\\"):
    model = load_learner(path, "trained_model.pkl")
    return model

def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    # image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image

""" 
API
"""

@app.route("/predict", methods=['POST'])
def predict():
    print(len(request.files))

    if request.files:
                request.files['photo']
                img = open_image(request.files['photo'])
            # read the image in PIL format

            # preprocess the image and prepare it for classification

            # classify the input image and then initialize the list
            # of predictions to return to the client
                preds = model.predict(img)
                print(preds)
                # results = imagenet_utils.decode_predictions(preds) 
                data["predictions"] = preds

                # loop over the results and add them to the list of
                # returned predictions
                # for (imagenetID, label, prob) in results[0]:
                #     r = {"label": label, "probability": float(prob)}
                #     data["predictions"].append(r)

                # indicate that the request was a success
                data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


"""
END API
"""

model = load_model()

if __name__ == "__main__":
    app.run(port=3001, debug=True)
