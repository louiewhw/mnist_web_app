from flask import Flask, jsonify, request, render_template, redirect, url_for, make_response
import pickle
import base64
import re
from io import BytesIO
from PIL import Image
import numpy as np

dt, dt_score = pickle.load(open('DecisionTree.pkl', 'rb'))
rf, rf_score = pickle.load(open('RandomForest.pkl', 'rb'))

svc, svc_score = pickle.load(open('SVC.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', dt_prediction=123)


@app.route("/predict", methods=['POST'])
def predict():

    req = request.get_json()[22:]
    imgdata = base64.b64decode(req)

    img = Image.open(BytesIO(imgdata)).convert('L')
    img = img.resize((28, 28))
    img = np.array(img).reshape(1,784)


    dt_predict = dt.predict(img)[0].tostring()
    dt_predict = int.from_bytes(dt_predict, 'big')

    svc_predict = svc.predict(img)[0].tostring()
    svc_predict = int.from_bytes(svc_predict, 'big')


    rf_predict = rf.predict(img)[0].tostring()
    rf_predict = int.from_bytes(rf_predict, 'big')

    res = make_response(jsonify({'dt':dt_predict, 'svc': svc_predict, 'rf': rf_predict}))

    return res




if __name__ == "__main__":
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.run()
