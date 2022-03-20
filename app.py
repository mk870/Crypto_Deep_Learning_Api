import pickle
import re
import numpy as np
from flask import  Flask,request, jsonify, make_response
import tensorflow as tf
from flask_cors import CORS


with open('btcscaler.pkl','rb') as f:
    btcscaler = pickle.load(f)

with open('Ethscaler.pkl','rb') as f:
    ethscaler = pickle.load(f)

with open('litescaler.pkl','rb') as f:
    litescaler = pickle.load(f)

btcmodel = tf.keras.models.load_model('btcmodel.h5')
ethmodel = tf.keras.models.load_model('ethmodel.h5')
litemodel = tf.keras.models.load_model('litemodel.h5')

def prediction(days,inputs,model):
  pred = []
  inputs2 = np.reshape(inputs,(1,60,1))
  
  for i in range(days):
    if i == 0:
      a = model.predict(inputs2)
      pred.append(a.tolist()[0][0])
      inputs2 = np.append(inputs2,pred[0])
      inputs2 = inputs2.tolist()
      inputs2 = np.reshape(inputs2,(1,61,1))
    elif i>0:
      b = model.predict(np.reshape(inputs2[0][i:],(1,60,1)))
      pred.append(b.tolist()[0][0])
      inputs2 = np.append(inputs2,pred[i])
      inputs2 = inputs2.tolist()
      inputs2 = np.reshape(inputs2,(1,(61+i),1))
  return pred


app = Flask(__name__)
CORS(app, supports_credentials=True)
@app.route("/",methods = ['GET'])
def hello():
    return jsonify({"response":"hello this is the crypto app"})

@app.route("/bitcoin",methods = ['POST'])
def bitcoinpred():
  predprices =[]
  if request.method == 'POST':
    req = request.get_json()
    inputs = req['modelInputs']
    days = req['days']
    inputs = btcscaler.transform(np.reshape(inputs,(-1,1)))
    scaledpred = prediction(days,inputs,btcmodel)
    arr = btcscaler.inverse_transform(np.reshape(scaledpred,(-1, 1)))
    arr = arr.tolist() 
    for i in arr:
        predprices.append(i[0])
    res = make_response(jsonify({'predictions':predprices}))
    return res
  

@app.route("/ethereum",methods = ['POST'])
def ethereumpred():
  if request.method == 'POST':
    req = request.get_json()
    inputs = req['modelInputs']
    days = req['days']
    inputs = ethscaler.transform(np.reshape(inputs,(-1,1)))
    scaledpred = prediction(days,inputs,ethmodel)
    arr = ethscaler.inverse_transform(np.reshape(scaledpred,(-1, 1)))
    arr = arr.tolist() 
    predprices = []
    for i in arr:
        predprices.append(i[0])
    res = make_response(jsonify({'predictions':predprices}))
    return res

@app.route("/litecoin",methods = ['POST'])
def litepred():
  if request.method == 'POST':
    req = request.get_json()
    inputs = req['modelInputs']
    days = req['days']
    inputs = litescaler.transform(np.reshape(inputs,(-1,1)))
    scaledpred = prediction(days,inputs,litemodel)
    arr = litescaler.inverse_transform(np.reshape(scaledpred,(-1, 1)))
    arr = arr.tolist() 
    predprices = []
    for i in arr:
        predprices.append(i[0])
    res =make_response(jsonify({'predictions':predprices}))
    return res

if __name__== "__main__":
    app.run(debug = True)