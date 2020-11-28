from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K

def unet(WIDTH, HEIGHT, CHANNELS):

	im_width  = WIDTH
	im_height = HEIGHT
	border = 5
	im_chan = CHANNELS # Number of channels: first is original and second cumsum(axis=0)
	#path_train = '../input/train/'
	#path_test = '../input/test/'
		
	input_img = Input((im_height, im_width, im_chan), name='img')

	#encoder
		# Build U-Net model

	c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (input_img)
	c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
	p1 = MaxPooling2D((2, 2)) (c1)

	c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
	c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
	p2 = MaxPooling2D((2, 2)) (c2)

	c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
	c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
	p3 = MaxPooling2D((2, 2)) (c3)

	c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
	c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
	p4 = MaxPooling2D(pool_size=(2, 2)) (c4)


	c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
	c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)

	u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
	#check out this skip connection thooooo
	u6 = concatenate([u6, c4])
	c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
	c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

	u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
	u7 = concatenate([u7, c3])
	c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
	c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

	u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
	u8 = concatenate([u8, c2])
	c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
	c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

	u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
	u9 = concatenate([u9, c1], axis=3)
	c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
	c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

	outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

	model = Model(inputs=[input_img], outputs=[outputs])
	model.compile(optimizer='adam', loss='binary_crossentropy') #, metrics=[mean_iou]) # The mean_iou metrics seens to leak train and test values...
	model.summary()

	return model