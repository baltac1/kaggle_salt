import numpy as np 
import pandas as pd 
from tqdm import tqdm
import keras, time
import cv2, os

IMG_SIZE = 101
DATADIR = "/home/balta/Desktop/salt_kaggle/masks/"
names = []
pred = []
data_dict = {}
model = keras.models.load_model('/home/balta/Desktop/salt_kaggle/128x128-50-epochs-32-batch_size_unet.h5')

test_fns = os.listdir(DATADIR)
for image in os.listdir(DATADIR):
	img_array = cv2.resize((cv2.imread(DATADIR + image, cv2.IMREAD_GRAYSCALE)), (101, 101))
	retval, img_array = cv2.threshold(img_array, 165, 255, cv2.THRESH_BINARY)
	pred.append(img_array)
pred = np.array(pred)/255.0
print("images read, normalized")
pred = np.expand_dims(pred, axis=3)



def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs

print("creating dict")
pred_dict = {fn[:-4]:RLenc(np.round(pred[i,:,:,0])) for i,fn in tqdm(enumerate(test_fns))}


import pandas as pd


print("outputting")
sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv('submission.csv')