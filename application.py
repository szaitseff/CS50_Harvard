# Interactive web app "Digit Recognizer"

from flask import Flask, render_template, request
import numpy as np
from imageio import imread  	# image reading
from skimage import transform	# image processing
from re import search       	# regular expressions
from base64 import b64decode	# decode dataURL
from keras import models        # ML API
import tensorflow as tf         # ML Framework
from custom import Custom_model, layer_dims
import pickle      # load weights for the custom model


# Define helper functions for image processing:
def decodeImage(dataURL):
	""" Decode dataURL from base64 into png file
	"""
	# Get image string from data URL:
	imgstr = search(r'base64,(.*)', str(dataURL)).group(1)
	# Decode imgstr from base64 and write as png to disk
	with open('output.png', 'wb') as output:
		output.write(b64decode(imgstr))

def convertImage(filename):
	""" Convert png file into tensor (ndarray)
	"""
	# Read png file from disk as ndarray
	x = imread(filename, pilmode='L')
	# Bit-wise black-white inversion to match MNIST dataset
	x = np.invert(x)
	# Resize the array to match MNIST dataset
	x = transform.resize(x, (28, 28))
	return x


# Initialize a pre-trained Keras model:
global k_model, graph
with open("model/k_model.json") as f1:
	k_model = models.model_from_json(f1.read())
k_model.load_weights("model/k_weights.h5")
graph = tf.get_default_graph()

# Initialized a pre-trained Custom model:
global c_model
c_model = Custom_model(layer_dims)
with open("model/c_weights.pickle", "rb") as f2:
	c_model.parameters = pickle.load(f2)


# Initialize the flask web app:
app = Flask(__name__)


# Ensure templates are auto-reloaded
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Ensure responses aren't cached
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response

@app.route("/")
def index():
    """Render landing page with canvas"""
    return render_template("index.html")

@app.route("/keras/", methods=['GET','POST'])
def kerasPredict():
	""" When the kerasPredict method is called, we're going to input
	the user drawn character as an image into the Keras model,
	perform inference, and return the classification.
	"""
	# Get the dataURL of the image:
	dataURL = request.get_data()
	# Decode dataURL from base64 into png file:
	decodeImage(dataURL)
	# Convert png into a ndarray of shape=(28, 28):
	x = convertImage('output.png')
	# Reshape the array for Keras input
	x = x.reshape(1, 28, 28, 1)
	# Perform classification with the Keras model:
	with graph.as_default():
		probs = k_model.predict(x)
		response = np.argmax(probs)
		prob = np.amax(probs)
		print(prob)
	return f'{response}, probability = {prob*100:.0f}%'

@app.route("/custom/", methods=['GET','POST'])
def customPredict():
	""" When the customPredict method is called, we're going to input
	the user drawn character as an image into the Custom model,
	perform inference, and return the classification.
	"""
	# Get the dataURL of the image:
	dataURL = request.get_data()
	# Decode dataURL from base64 into png file:
	decodeImage(dataURL)
	# Convert png into a ndarray of shape=(28, 28):
	x = convertImage('output.png')
	# Reshape the array for Custom model input
	x = x.reshape(784, 1)
	# Perform classification with the Custom model:
	probs, _ = c_model.forward_propagation(x, keep_prob=1)
	response = np.argmax(probs)
	prob = np.amax(probs)
	print(prob)
	return f'{response}, probability = {prob*100:.0f}%'



if __name__ == "__main__":
	app.run()