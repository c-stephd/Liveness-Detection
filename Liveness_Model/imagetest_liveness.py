# This file contains the code to test the liveness model with a real or spoof image
# All you need is the path of the image to be tested


import cv2
import os
import numpy as np
import argparse
import PIL
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import model_from_json


# Define command line argument to input the image path
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", type=str, required= True,
	help="path to to the image to be tested")
args = vars(ap.parse_args())


root_dir = os.getcwd()

# Load Liveness Model graph
json_file = open('liveness_model.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load liveness model weights 
model.load_weights('liveness_model.h5')


# Defining the image test function

def image_test(img_path):
  predictor = {'category': 'value'}

  img = load_img(img_path,target_size=(160,160))
  img = img_to_array(img)
  img = np.expand_dims(img,axis=0)
  img = img / 255.0

  prediction = model.predict(img)

  if prediction > 0.5:
    prediction_class = 'spoof'
  else:
    prediction_class = 'real'

  predictor['category'] = prediction_class
  print(prediction_class) 

  #except Exception as e:
  #  pass

  return predictor['category']
 
  

result = image_test(args['path'])

print('The image is a {} image'.format(result))
