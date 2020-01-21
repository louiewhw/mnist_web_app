from flask import Flask, jsonify, request, render_template
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

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        svc_predict = get_predict()
        

    else:
        svc_predict = '?'
    return render_template('index.html', dt_prediction=svc_predict)


# @app.route("/predict", methods=['POST'])
def get_predict():

    req = request.get_json()[22:]
    imgdata = base64.b64decode(req)

    img = Image.open(BytesIO(imgdata)).convert('L')
    img = img.resize((28, 28))
    img = np.array(img).reshape(1,784)

    svc_predict = predict(img)
    print(svc_predict)
    return svc_predict
    



def predict(arr):
    svc_predict = svc.predict(arr)
    
    return svc_predict


if __name__ == "__main__":
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.run()
