#**************imports important****************
import os,cv2
import numpy as np
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras import backend as K
K.set_image_dim_ordering('tf')
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten 
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization , SeparableConv2D
from keras.optimizers import SGD , RMSprop

# Define data path of dataset
PATH = os.getcwd()
data_path = PATH + '/dataset_train100'
data_dir_list = os.listdir(data_path)

#here we define the hyperparameters
img_rows=128
img_cols=128
num_channel=3
epochs=50
batch_size=150
# here we Define the number of classes
num_classes = 28
img_data_list=[] #this list we are going to use it after ... we are going to put data in this list
#this function consist a  resize the image to a fixed size(64, 64), then flatten the image into a list of raw pixel intensities
def image_to_feature_vector(image, size=(64, 64)):
		return cv2.resize(image, size).flatten()

for dataset in data_dir_list:
	img_list=os.listdir(data_path+'/'+ dataset)
	print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
	for img in img_list:
		input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
		
		input_img_flatten=image_to_feature_vector(input_img,(64,64))
		img_data_list.append(input_img_flatten)

img_data = np.array(img_data_list)
# Making sure that the values are float so that we can get decimal numbers after division to not loose  information
img_data = img_data.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
img_data /= 255
print (img_data.shape)
print("all the images has been loaded")
num_of_samples = img_data.shape[0]
print(num_of_samples)
labels = np.ones((num_of_samples,),dtype='int64')
i=0
t=0
#this loop while for take just 100 images in each class in our dataset
while i < len(labels):
	labels[i:i+100]=t
	t=t+1
	i=i+100

# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels)
print(type(Y))
# here we Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)
print("shuffle is working")
# Split the dataset into training set(takes 80% from dataset) and test set(takes 20% from dataset)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
print("preparing training and test data")
X_train = np.reshape(X_train, (len(X_train),  64, 64, 3))
X_test = np.reshape(X_test, (len(X_test), 64, 64, 3))

cnn4 = Sequential()
cnn4.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(64,64,3)))
cnn4.add(BatchNormalization())

cnn4.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
cnn4.add(BatchNormalization())
cnn4.add(MaxPooling2D(pool_size=(2, 2)))
cnn4.add(Dropout(0.25))


cnn4.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
cnn4.add(BatchNormalization())
cnn4.add(Dropout(0.25))

cnn4.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
cnn4.add(BatchNormalization())
cnn4.add(MaxPooling2D(pool_size=(2, 2)))
cnn4.add(Dropout(0.25))

cnn4.add(Flatten())

cnn4.add(Dense(512, activation='relu'))
cnn4.add(BatchNormalization())
cnn4.add(Dropout(0.5))

cnn4.add(Dense(128, activation='relu'))
cnn4.add(BatchNormalization())
cnn4.add(Dropout(0.5))

cnn4.add(Dense(128, activation='relu'))
cnn4.add(BatchNormalization())
cnn4.add(Dropout(0.5))

cnn4.add(Dense(28, activation='softmax'))

# construct the image generator for data augmentation .. in order to solve the problem of overffiting
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

#here we apply gradient decent (we chose for that adam optimize to find the global minimum  ) .. and we update parametrs until we find the best parametrs
# initiate RMSprop optimizer
opt = RMSprop(lr=0.0001, decay=1e-6)
sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
cnn4.compile(loss='categorical_crossentropy', optimizer='Adam',metrics=["accuracy"]
# Viewing model_configuration .. we save our model in file 'modelsummary.txt'
with open('modelsummary.txt', 'w') as f:
    with redirect_stdout(f):
        cnn4.summary()
cnn4.summary()
print("the general architecture has been shown")
cnn4.get_config()
cnn4.layers[0].get_config()
cnn4.layers[0].input_shape
cnn4.layers[0].output_shape
cnn4.layers[0].get_weights()
np.shape(cnn4.layers[0].get_weights()[0])
cnn4.layers[0].trainable

filename='model_train_new.csv'
csv_log=callbacks.CSVLogger(filename, separator=',', append=False)
filepath="Best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5"
checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = csv_log
print("helloooooooooooo")
BS = 150
#fitting the model
H = cnn4.fit_generator(
	aug.flow(X_train, y_train, batch_size=150),
	validation_data=(X_test, y_test),
	steps_per_epoch=len(X_train) // BS,
	epochs=100, verbose=1)

# visualizing losses and accuracy
train_loss=H.history['loss']
#val_loss=hist.history['val_loss']
train_acc=H.history['acc']
#val_acc=hist.history['val_acc']
xc=range(epochs)
cnn4.save("model.h5", overwrite=True)
#%%
# Evaluating the model

score = cnn4.evaluate(X_test, y_test, verbose=0)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])
# summarize history for accuracy
plt.plot(train_acc)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(train_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

