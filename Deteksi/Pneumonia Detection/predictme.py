from flask import Flask, render_template, request,jsonify
from tensorflow.keras.models import load_model
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model
import cv2
import numpy as np
import base64
from PIL import Image
import io
import re
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

img_size=180

app = Flask(__name__) 

app.add_url_rule('/photos/<path:filename>', endpoint='photos', view_func=app.send_static_file)

sess = tf.compat.v1.Session()
graph = tf.get_default_graph()

set_session(sess)
model=load_model(os.path.join("./model/","pneum.h5"))

label_dict={0:'Pneumonia Negative', 1:'Pneumonia Positive'}
def preprocess(img):

	img=np.array(img)

	if(img.ndim==3):
		gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	else:
		gray=img

	gray=gray/255
	resized=cv2.resize(gray,(img_size,img_size))
	resized=resized.reshape(-1,img_size,img_size,1)
	return resized	

@app.route("/")
def index():
	return(render_template("index.html"))

@app.route("/predict", methods=["POST"])
def predict():
	print('HERE')
	global graph
	global sess
	with graph.as_default():
		message = request.get_json(force=True)
		encoded = message['image']
		decoded = base64.b64decode(encoded)
		dataBytesIO=io.BytesIO(decoded)
		dataBytesIO.seek(0)
		image = Image.open(dataBytesIO)

		test_image=preprocess(image)
		set_session(sess)
		prediction = model.predict(test_image)
		result = int(np.around(prediction[0]))
		accuracy=float(np.max(prediction,axis=1)[0])

		label=label_dict[result]

		print(prediction,result,accuracy)

		response = {'prediction': {'result': label,'accuracy': accuracy}}
		

	return jsonify(response)

app.run(debug=True)

#<img src="" id="img" crossorigin="anonymous" width="400" alt="Image preview...">