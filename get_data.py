import numpy as np
import os, time
import cv2                       
import random
import pandas as pd
import matplotlib.pyplot as plt

DATADIR = "/home/balta/Desktop/salt_kaggle/train"
CATEGORIES = ['images', 'masks']            
IMG_SIZE = 128                    

images = []
masks = []

print("starting")                        

for category in CATEGORIES:
	path = DATADIR + "/" + category + "/"
	for image in os.listdir(path):
		try:
			img_array = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)
			img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
			
			if category == 'images':
				images.append(img_array)

			elif category == 'masks':
				masks.append(img_array)

		except Exception as e:
			print(e)
                 
print("oh fk numpy")
images = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE,1)   
print("images {}".format(len(images)))        
masks = np.array(masks).reshape(-1, IMG_SIZE, IMG_SIZE,1)   
print("masks {}".format(len(masks))) 
        

print("Normalizing...")
images = images / 255.0
masks = masks / 255.0

np.save("images.npy", images)
np.save("masks.npy", masks)
