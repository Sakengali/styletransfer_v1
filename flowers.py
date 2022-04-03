import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import PIL

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

#images 
import pathlib
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz" #str file to save the url
flowers = tf.keras.utils.get_file('flower_photos',origin=dataset_url, untar=True)  # script to get the dataset from tf keras dataset called 'flower photos' of 'that' url, and extract it from 'tar' format
flowers = pathlib.Path(flowers) # script to turn the path of flower (stored as str) to a PATH file.

#reduced ds for faster testing
#flowers = pathlib.Path(r'C:\Users\Professional\.keras\datasets\flower_classification\flowers_extremeshorted') # r' is necessary

print(type(flowers))
image_n = list(flowers.glob('*/*.jpg'))
print(type(image_n), len(image_n))

#dandelion = list(flowers.glob('dandelion/*'))
#PIL.Image.open(str(dandelion[1]))              #turn path dandelion 1 into str, and open the image in that path


#dataset
batch_size = 32  #why use batches in each iteration? to not overfit, to speed up?
image_h = 180
image_w = 180

tr_set = tf.keras.utils.image_dataset_from_directory(
    flowers,
    validation_split = 0.2,
    subset = 'training',
    seed = 123,                                #keras generates random initial w using numpy, and takes on seed: indicating this seed - make sure get same initial ws each run
    image_size = (image_h, image_w),
    batch_size=batch_size
)

val_set = tf.keras.utils.image_dataset_from_directory(
    flowers,
    validation_split = 0.2,
    subset='validation',
    seed=123,                            
    image_size=(image_h,image_w),
    batch_size=batch_size
)

classes = tr_set.class_names; class_n = len(classes)
print("classes: ", classes)

#img visualization

#plt.figure(figsize=(16,16))
#for img, labels in tr_set.take(1): #for pxls and classes of one batch?
 #   for i in range(16):
  #      plt.subplot(4,4,i+1)
   #     plt.imshow(img[i].numpy().astype('uint8'))
    #    #print(img[i].numpy().astype('uint8'))
     #   plt.title(tr_set.class_names[labels[i]])
      #  plt.axis('off')

#configuring data
tr_set = tr_set.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)  #q shuffling is intuitively compr, but why shuffle the cach?
val_set = val_set.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

#data aug
data_aug = Sequential([
    layers.RandomFlip("horizontal", input_shape = (image_h, image_w, 3)),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.1),
])


model = Sequential([
    data_aug,
    layers.Rescaling(1./255, input_shape=(image_h, image_w, 3)), #feature scaling
    layers.Conv2D(16,3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32,3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64,3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(0.2),
    layers.Dense(128, activation='relu'),
    layers.Dense(class_n)
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()

epochs = 3
d_analyse = model.fit(
    tr_set,
    validation_data = val_set,
    epochs = epochs
)

model.save("all_flower_model")

#model d_analyze
tr_acc = d_analyse.history['accuracy']
tr_loss = d_analyse.history['loss']

val_acc = d_analyse.history['val_accuracy']
val_loss = d_analyse.history['val_loss']

epoch_range = range(epochs)
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(epoch_range, tr_loss, label='train loss')
plt.plot(epoch_range, val_loss, label='validation loss')
plt.title("loss curve")
plt.xlabel("epochs"); plt.ylabel("losses"); plt.legend(loc='upper right')

plt.subplot(1,2,2)
plt.plot(epoch_range, tr_acc, label='train accuracy')
plt.plot(epoch_range, val_acc, label='validation accuracy')
plt.title("accuracy curve")
plt.xlabel("epochs"); plt.ylabel("accuracy"); plt.legend(loc='lower right')
