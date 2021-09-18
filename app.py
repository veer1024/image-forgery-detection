from __future__ import division, print_function
import os
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

## importing tensorflow 
# coding=utf-8
import sys
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
#from keras.models import load_model
from tensorflow.keras.models import load_model
#from keras.preprocessing import image
from tensorflow.keras.preprocessing import image
app = Flask(__name__)
Upload_folder = "./uploaded_images/"
model_folder = "./saved_model/"
Model_path = "./saved_model/my_model.h5"

### loading the trained model or saved model

model = load_model(Model_path)
print("**********************************************************************Model Summary************************************************************************************")
print(model.summary())
print("**********************************************************************Model Summary************************************************************************************")
"""
image_path = "./uploaded_images/finger.png"
img = image.load_img(image_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)
print("image shape ***************************************************************8")
print(img_batch.shape)

prediction = model.predict(img_batch)
print("*****************************************************Prediction image value****************************************************")
print(prediction)
print("*****************************************************Prediction image value****************************************************")
"""
@app.route("/<page>",methods=["GET","POST"])
def upload_image(page):
     if request.method == "POST":
        print("Got a POST request ")
        if page == "index.html":
          print("GOt a POST request with index")
          image_file = request.files["image"]
          if image_file:
              image_location = os.path.join(
                      Upload_folder,
                      image_file.filename
              )
              image_file.save(image_location)
              print(image_location)
              filename = image_location
              ## loding image 
              img = image.load_img(image_location, target_size=(128, 128))
              img_array = image.img_to_array(img)
              img_batch = np.expand_dims(img_array, axis=0)
              prediction = model.predict(img_batch)
              print("********************************************************Prediction****************************************************************************************")
              print(prediction)
              print(page)
              print("********************************************************Prediction****************************************************************************************")
              os.remove(filename)
              message = "message for file"
              prediction_message = "Prediction count value in between of 0 and 1 is :" + str(prediction)
              ## setting value of message 
              if prediction < 0.5:
                 message = "Their is no forged detected in the image"
              else:
                 message = "Their might be some forged in the image"
               
              return render_template("index.html",prediction=prediction_message, message=message)
        elif page == "admin.html":
            print("Got a post reqeust with admin")
            model_file = request.files["model_file"]
            print(model_file)
            if model_file:
              model_location = os.path.join(
                      model_folder,
                      model_file.filename
              )
              model_file.save(model_location)
              print(model_location)
              filename = model_location
              ## loding model 
              #status = model.predict(filename)
              print("********************************************************Prediction****************************************************************************************")
              #print(prediction)
              print(filename+" uploaded!!")
              print("********************************************************Prediction****************************************************************************************")
              #os.remove(filename)
              message = "Model Uploaded successfully!!"
              prediction_message = "Status: Running" #+ str(prediction)
              ## setting value of message 
              """
              if prediction < 0.5:
                 message = "Their is no forged detected in the image"
              else:
                 message = "Their might be some forged in the image"
              """
              return render_template("admin.html",prediction=prediction_message, message=message)
        else:
            print("Route not found")
     if page == "index.html":
         return render_template("index.html")
     if page == "admin.html":
         return render_template("admin.html")
     else:
         return render_template("index.html")
               
if __name__ == "__main__":
   app.run(host='0.0.0.0',port=1234,debug=True)
