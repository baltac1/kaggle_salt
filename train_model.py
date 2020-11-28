from unet import unet
import numpy as np
from keras.callbacks import TensorBoard
import os 
import cv2
import matplotlib.pyplot as plt
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


IMG_SIZE = 128
CHANNELS = 1
BATCH_SIZE = 32
EPOCHS = 100
DATADIR = '/home/balta/Desktop/salt_kaggle/'

images = np.load('images.npy')
masks = np.load('masks.npy')


model = unet(IMG_SIZE, IMG_SIZE, CHANNELS)
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(images, masks, test_size=0.15, random_state=42)

callbacks = [
    EarlyStopping(patience=5, verbose=1),
    ReduceLROnPlateau(patience=3, verbose=1),
    ModelCheckpoint('model-tgs-salt-1.h5', verbose=1, save_best_only=True, save_weights_only=True)
]

results = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callbacks,
                    validation_data=(X_valid, y_valid))


model.save('{}x{}-{}-epochs-{}-batch_size_unet.h5'.format(IMG_SIZE, IMG_SIZE, EPOCHS, BATCH_SIZE))

'''

DATADIR = '/home/balta/Desktop/salt_kaggle/images'
CATEGORIES = ['']
test_images =[]
for category in CATEGORIES:
	path = DATADIR + "/"
	for image in os.listdir(path):
		for i in range(10):
			try:
				img_array = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)
				img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
				
				test_images.append(img_array)

			except Exception as e:
				print(e)
		break
test_images = np.array(test_images).reshape(-1, IMG_SIZE, IMG_SIZE,1)
test_images = test_images / 255.0

model = keras.models.load_model('100x100-100-epochs-64-batch_size.h5')
decoded_imgs = model.predict(test_images)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(test_images[i].reshape(100, 100))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(100, 100))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
'''
