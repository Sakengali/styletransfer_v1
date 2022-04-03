import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from matplotlib import image

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#img

orig_h, orig_w, _ = image.imread('img_orig.jpg').shape    #to get init sizes of orig img
image_h = 400; image_w = round(image_h*orig_w/orig_h)   #resizing to h =400

img_orig = keras.utils.load_img('img_orig.jpg', target_size=(image_h, image_w)) # keras.utils.load_img loads from pathS
img_style = keras.utils.load_img('img_style_pink.jpg', target_size=(image_h, image_w)) #same for img_style


#auxiliary fs to process img for vgg, and reprocess img from vgg output
def process(img):
    img = keras.utils.img_to_array(img)       #img data turned into np array i guess
    img = np.expand_dims(img, axis=0)         # dims are expanded - it is now: shape: 1,h,w,3 -> i guess it is needed for processing by vgg
    img = keras.applications.vgg19.preprocess_input(img)  #rgb to bgr, adds zero-center values
    plt.imshow(img.reshape(image_h, image_w, 3)); plt.text(200,300, "that's how vgg processes img to")  #reshaping to get h x w x 3 array and show processed img
    return img

def reprocess(img):
    img = img.reshape((image_h, image_w, 3))
    
    img[:,:,0] += 103.939 #reversing vgg zero-centering
    img[:,:,1] += 116.779
    img[:,:,2] += 123.68
    img = img[:,:,::-1]  #bgr to rgb
    #img[:,:,0], img[:,:,2] = img[:,:,2], img[:,:,0] #bgr to rgb by collet
    img = np.clip(img,0,255).astype('uint8')
    plt.imshow(img); plt.text(200,350, 'processed back')
    return img

#to show how imgs are processed & then reprocessed
#plt.figure(figsize=(8,4))
#plt.subplot(1,2,1)
#duck = process(img_orig) 
#plt.subplot(1,2,2)
#reprocess(duck)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#model

#importing vgg19 convnet model - transfer learning
vgg19_conv = keras.applications.vgg19.VGG19(
    weights="imagenet",                
    include_top=False,                                   
)

#syntax to get the activation values (output matrices) of conv + pool layers
outputs_dict = dict([(layer.name, layer.output) for layer in vgg19_conv.layers]); # taking layer names and outputs of loaded model
feature_extraction = keras.Model(inputs = vgg19_conv.inputs, outputs = outputs_dict)
#feature_extraction.summary()




#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#cost f

#aux fs for cost comp
def content_loss(img_orig, stylated_img):
    return tf.reduce_sum(tf.square(img_orig-stylated_img))


def gram_matrix(output_mtrx):    #output_mtrx of a conv layer
    output_mtrx = tf.transpose(output_mtrx, (2,0,1))
    output_mtrx = tf.reshape(output_mtrx, (output_mtrx.shape[0], -1))          #reshaping to vector - '-1' in reshape == unspecified, that is concluded by prog. tried my own way by making to vector then gramming - overrided memory error
    #print(output_mtrx.shape); print(tf.transpose(output_mtrx).shape)
    gram_matrix = tf.matmul(output_mtrx, tf.transpose(output_mtrx))
    return gram_matrix

#gram_matrix(np.matrix([[1,2,3],[1,4,5]])) #checking

def style_loss(img_style, stylated_img):
    S = gram_matrix(img_style)
    G = gram_matrix(stylated_img)
    num_kernels = img_style.shape[2]
    return tf.reduce_sum( 1/(2*image_h*image_w*num_kernels) * tf.square(S - G))

#Collet also implemented regularizatio loss - i spose it is Lambda_l of Ng. will try later.

#layers whose values use for style cost:
layers_style_cost = [
    "block1_conv2",
    "block2_conv2",
    "block3_conv4",
    "block4_conv4",
    "block5_conv4"
]
#beta
beta = 5e-8

#layer for content
layer_content = [
    "block5_conv4" #last layer
]
#alpha
alpha = 1e-8

#total cost:
def cost_content(img_orig, stylated_img):

    img_orig_activation = feature_extraction(img_orig)
    stylated_img_activation = feature_extraction(stylated_img)

    img_orig_content = img_orig_activation[layer_content[0]]
    stylated_img_content = stylated_img_activation[layer_content[0]]

    J_content = content_loss(img_orig_content, stylated_img_content) 

    return J_content


def cost_style(img_style, stylated_img):

    img_style_activation = feature_extraction(img_style)
    stylated_img_activation = feature_extraction(stylated_img)

    J_style = 0
    for layer in layers_style_cost:

        img_style_layer = img_style_activation[layer]
        stylated_img_layer = stylated_img_activation[layer]

        #reshaping tensors for gram matrix
        a_size = tf.shape(img_style_layer)   #activation size
        img_style_layer = tf.reshape(img_style_layer, (a_size[1], a_size[2], a_size[3]))
        stylated_img_layer = tf.reshape(stylated_img_layer, (a_size[1], a_size[2], a_size[3]))

        J_style += style_loss(img_style_layer, stylated_img_layer)
    return J_style

def total_cost(img_orig, img_style, stylated_img):
    return alpha*cost_content(img_orig, stylated_img) + beta*cost_style(img_style, stylated_img)

#checking costs
#img_orig = process(tf.convert_to_tensor(img_orig))
#img_style = process(tf.convert_to_tensor(img_style))
#cost_content(img_orig, img_style)
#cost_style(img_orig, img_style)

#NOTE: 3 costs here are calculated separately as i wanted to debug them. will combine afterwards.

#TODO: calculate total cost - content w/ feature extr, style - sum jstyle for layers in it, sum them; random initialize img of h x w x 3, then minimize cost - with gradient, or smth, after n iters, plt.imshow im

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#optimizing

@tf.function
def compute_grads_loss(img_orig, img_style, stylated_img):
    with tf.GradientTape() as tape:
        cost = total_cost(img_orig, img_style, stylated_img)   #defining relation between x and y to find dy/dx: y here - loss, x is generated img.
    gradients = tape.gradient(cost, stylated_img)               #gradient of cost w.r.t stylated_img.
    return cost, gradients


#initializing stylated img as orig_img, and storing as tf Variable.
stylated_img = tf.Variable(process(img_orig))

#processing imgs and converting into tensors - for feature_extraction
img_orig = tf.convert_to_tensor(process(img_orig))
img_style = tf.convert_to_tensor(process(img_style))

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01) #defining optimiser - SGD, rate 0.1

iterations = 100                                
for i in range(1, iterations+1):                                              #do # iterations of gd                               
    cost, gradients = compute_grads_loss(img_orig, img_style, stylated_img)
    optimizer.apply_gradients([(gradients, stylated_img)])                        #change stylated_img's values (h x w x 3) in accordance with gradients
    print(f'iteration {i}: cost: {cost}')

print(f"Img generated with {iterations} iterations. Cost is {cost}")
generated_name = f"image_{iterations}.png"

stylated_img = reprocess(stylated_img.numpy())
keras.utils.save_img(generated_name, stylated_img)