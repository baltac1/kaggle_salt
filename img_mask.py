import numpy as np
import os, time
import cv2                       
import keras
import imageio
from PIL import Image
import matplotlib.pyplot as plt


DATADIR = "/home/balta/Desktop/salt_kaggle/"
CATEGORIES = ['images', 'masks']            
IMG_SIZE = 128
IMG_NAMES = []                  
test_images = []

i = 0
print("starting")  
path = DATADIR + "images/"
for image in os.listdir(path):
	try:
		img_array = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)
		img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
		test_images.append(img_array)
		i+=1
		if i%3000 == 0:
			print(i)

	except Exception as e:
		print(e)
i = 0


print('names')
path = DATADIR + "images/" 
for image in os.listdir(path):
	try:
		IMG_NAMES.append(str(image))
		i+=1
		if i%3000 == 0:
			print(i)

	except Exception as e:
		print(e)

print('loading model')
model = keras.models.load_model('/home/balta/Desktop/salt_kaggle/128x128-50-epochs-32-batch_size_unet.h5')

test_images = np.array(test_images).reshape(len(test_images), IMG_SIZE, IMG_SIZE,1)
print('normalizing')
test_images = test_images / 255.0

print('generating decoded images')                  
decoded_imgs = model.predict(test_images)
	
i = 0
for i in range(len(decoded_imgs)):
	img = decoded_imgs[i]
	imageio.imwrite('{}_mask.png'.format(IMG_NAMES[i]), img)
	if i%3000 == 0:
		print(i)