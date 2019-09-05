from keras.models import load_model
from keras.models import model_from_json
import tensorflow as tf
from keras import backend as K
from utils import *
import cv2
from build_predicator import *
from utils import *
import matplotlib.pyplot as plt
import sys
from glob import glob
import csv

args=sys.argv
input_dir=args[1]
try:
   output_csv=args[2]
except:
   output_csv='results.csv'

input_size=416
max_box_per_image = 10
anchors=[0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
labels=["face"]
#image_path   = "image/olivier.jpg"


# load json and create model
json_file = open('freeze_graph/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json, custom_objects={"tf": tf})
# load weights into new model
#print("Start loading model from disk")
loaded_model.load_weights("freeze_graph/model.h5")
print("Loaded model from disk")

g1=Graph()

image_paths=glob(input_dir+'/*png')
image_paths.extend(glob(input_dir+'/*jpg'))
image_paths=sorted(image_paths)

results=[]
for image_path in tqdm(image_paths):
    image = cv2.imread(image_path)
    image2 = cv2.resize(image, (input_size, input_size))
    image2 = normalize(image2)
    input_image = image2[:,:,::-1]
    input_image = np.expand_dims(input_image, 0)
    dummy_array = dummy_array = np.zeros((1,1,1,1,max_box_per_image,4))
    input_data=[input_image, dummy_array]
    netout = loaded_model.predict([input_image, dummy_array])[0]
    boxes  = decode_netout2(netout, labels,anchors)
    if len(boxes) != 0:
        listImg,idx=getFacesList(image, boxes)
        listPrediction=g1.classify_age([listImg])
        temp=listPrediction[0][3]
        if temp[0]>temp[1]:
           gender="Female"
        else:
           gender="Male"
        result=[image_path,gender,listPrediction[0][1]]
        results.append(result)
        
with open('results.csv', mode='w') as f:
    writer = csv.writer(f)
    writer.writerows(results)


