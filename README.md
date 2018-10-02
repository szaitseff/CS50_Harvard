## My CS50 Project is an interactive web application called "Digit Recognizer".
### You can see a video presentation of my project on YouTube [here](https://www.youtube.com/watch?v=ein2VnyxFTQ)

### Front-end:
**Canvas**
The application is based on a simple Flask server that renders the only landing web page defined in 'templates/index.html' file.
This page contains a drawing panel, based on the JavaScript Canvas element powered by the code in 'static/scripts.js' file,
which contains all external JavaScript code. You can draw black and white images (presumably digits from 0 to 9) on the Canvas.
There are also two buttons right below the Canvas called "Ask Keras Model" and "Ask Custom Model", and the third button beneath
them called <CLEAR>.

**scripts.js**
The purpose of the CLEAR button is to clear any drawing from the Canvas, and a few lines of JavaScript code perform this function.
For each of the 'asking' buttons we bind an event handler that sends Ajax requests to the relevant Flask app functions.
So you can ask two pretrained machine learning models to classify handwritten digits on the Canvas.


### Back-end:
**application.py**
The Flask app in the 'application.py' file has two functions: 'kerasPredict' and 'customPredict', which can process
images with handwritten digits into numpy arrays, feed the relevant machine learning models and get classification of the image,
providing response to the Ajax requests from the 'asking' buttons on the landing page. The images written on Canvas have size
of 280x280 pixels, then they are decoded by the application into png files and converted to numpy arrays of size 28x28 pixels
to make predictions with the machine learning models.

**Kaggle**
The machine learning models are the 'brains' of the application. I trained them on Kaggle as part of the
[Digit Recognizer competition](https://www.kaggle.com/c/digit-recognizer) on the infamous MNIST dataset of tens
of thousands of handwritten images. The architecture of both machine learning models is the same - a simple fully-connected
neural network with one hidden layer. Each image in the MNIST dataset contains a handwritten digit in the range [0-9]
and has size of 28x28 pixels (784 pixels in total). Therefore we have an input layer of 784 pixels, a hidden layer of 512 neurons,
and an output layer of 10 digit classes.

The difference between the two models is that the first neural network was implemented using Keras API with Tensorflow backend.
To build a machine learning model with the Keras library takes just a few lines of code. On the contrast, the second model
was built 'from scratch' using pure Python code and Numpy arrays. I called it the "Custom model". Both models have been trained
in a Jupyter notebook on this [Kaggle kernel](https://www.kaggle.com/szaitseff/under-the-hood-a-dense-net-w-mnist-dataset). Both
models showed a comparable performance with c.98% accuracy on the validation set.

**project/model/ folder and custom.py**
To be used in the web application, the Keras model/weights have been saved into 'k_model.json' and 'k_weights.h5'
files from the Jupyter notebook on Kaggle and uploaded to 'project/model' folder.
For the Custom model, I copied the Python code required for predictions into 'project/custom.py' file, which is then imported
into the application, and loaded the pre-trained model parameters to the 'project/model/c_weights.pickle' file.

The image processing by my app is likely to differ from processing of the original MNIST images, which were truly handwritten
on paper and then digitized. Therefore, the "production" accuracy of my models in the web application is expected lower
than the training accuracy on images from MNIST. But it doesn't mean we are not be able to have fun with "Digit Recognizer" :)