import tensorflow as tf
import numpy as np
from PIL import Image
import os
import glob
import random

#Path to the folder that we want to save the logs for tensorboard :
logs_path = "./logs/embedding/"
SAVE_PATH = './save'
EPOCHS = 10
#LEARNING_RATE = 0.000000001
#dropout_rate=0.2
MODEL_NAME = 'model'
#batch_size = 4
#num_batch = 3
#LEARNING_RATE = tf.train.exponential_decay( 0.0001, batch_size,  0.95 ,decay_rate= 0.96 , staircase=True)  #adaptative learning rate
LEARNING_RATE = 0.0001

#CALLING THE IMAGES##########################################################################################"

# load stack of images :
# img_raw = glob.glob('./img/input/*.tif')
# img_lab = glob.glob('./img/output/*.tif')

# im = Image.open(filename)
# im_arr = np.array(im)[:200, :200]
# def f1(i):
X = []
for filename in glob.glob('./img/train/input/*'):  # /{}...format(i)
    im = Image.open(filename)
    X.append(np.array(im)) #[:200, :200])  # turn to array and #resize images to (200,200) ; append: send the value to the X list above
    # X = np.array(X_)
# print new shape
# print(X.shape)
tf.shape(X)

Y = []
for filename in glob.glob('./img/train/output/*'):
    im = Image.open(filename)
    Y.append(np.array(im)) #[:200, :200])
    # Y = np.array(Y_)

    # print(Y.shape)

# shuffling arrays keeping the same order :
# indices = np.arange(X.shape[0])
# np.random.shuffle(indices)
# X = X[indices]
# Y = Y[indices]


# print('input')
# print(X)
# print('output')
# print(Y)


#######################################################################################
#     #test files :
# for filename in glob.glob('./img/test/input/*.tif'):   #/{}...format(i)
#         im = Image.open(filename)
#         X_ = np.array(im)[:200, :200]                   # turn to array and #resize images to (200,200)

#     #print new shape
#         print(X_.shape)


# for filename in glob.glob('./img/test/output/*.tif'):
#         im = Image.open(filename)
#         Y_ = np.array(im)[:200, :200]

#         print(Y_.shape)
###########################################################################################

# reshape array :
X_train = np.reshape(X, newshape=(-1, 72, 72, 1))  # [: , 200:300, 200:300, :] #reshape image to have a 4 dimentional tensor and to fit the placeholder
Y_train = np.reshape(Y, newshape=(-1, 72, 72, 1))  # [: , 200:300, 200:300, :]
# X_test = np.reshape(X_ , newshape=(-1, 200, 200, 1))     #[: , 200:300, 200:300, :] #reshape image to have a 4 dimentional tensor and to fit the placeholder
# Y_test = np.reshape(Y_ , newshape=(-1, 200, 200, 1))     #[: , 200:300, 200:300, :]


# define the graph
# tf.reset_default_graph()

# Create graph and Define placeholder for input :
with tf.variable_scope('Input'):
    x_ph = tf.placeholder("float32", [None, 72, 72, 1])
tf.summary.image('input_img', tf.reshape(X_train, (-1, 72, 72, 1)), max_outputs=5)
# graph , and define placeholder for output :
with tf.variable_scope('Output'):
    y_ph = tf.placeholder("float32", [None, 72, 72, 1])
    tf.summary.image('output_img', tf.reshape(X_train, (-1, 72, 72, 1)), max_outputs=5)

#ENCODER ###############################################################################################################
# Input Layer
# input_layer = slice # convrt input feature maps to this shape [-1,28,28,1]

# Convolutional Layer #1
conv1 = tf.layers.conv2d(inputs=x_ph, filters=64, kernel_size=[3, 3], strides=(1, 1), padding="same",
                         activation=tf.nn.relu)
conv1bis = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[3, 3], strides=(1, 1), padding="same",
                            activation=tf.nn.relu)

# Pooling Layer #1
pool1 = tf.layers.max_pooling2d(inputs=conv1bis, pool_size=[2, 2], strides=2)

# Convolutional Layer #2
conv2 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
conv2bis = tf.layers.conv2d(inputs=conv2, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)

# Pooling Layer #2
pool2 = tf.layers.max_pooling2d(inputs=conv2bis, pool_size=[2, 2], strides=2)

# Convolutional Layer #3
conv3 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
conv3bis = tf.layers.conv2d(inputs=conv3, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
conv3bisbis = tf.layers.conv2d(inputs=conv3bis, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)

# Pooling Layer #3
pool3 = tf.layers.max_pooling2d(inputs=conv3bisbis, pool_size=[2, 2], strides=2)

# Convolutional Layer #4
conv4 = tf.layers.conv2d(inputs=pool3, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
conv4bis = tf.layers.conv2d(inputs=conv4, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
conv4bisbis = tf.layers.conv2d(inputs=conv4bis, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)

# Pooling Layer #4
pool4 = tf.layers.max_pooling2d(inputs=conv4bisbis, pool_size=[2, 2], strides=2)

# Convolutional Layer #5
conv5 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
conv5bis = tf.layers.conv2d(inputs=conv5, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
conv5bisbis = tf.layers.conv2d(inputs=conv5bis, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)

# Pooling Layer #5
pool5 = tf.layers.max_pooling2d(inputs=conv5bisbis, pool_size=[2, 2], strides=2)

#DECODER################################################################################################################
# upsampling layer 1 :
#upsample1 = tf.resized_image(pool5, size=(72, 72), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) #note: if you use resize, it will distore your image
upsample1 = tf.reshape(pool5, shape=(72, 72))
# Deconvolutional layer 1 :
deconv1 = tf.layers.conv2d_transpose(inputs=upsample1, filters=512, kernel_size=(3, 3), padding='same',
                                     activation=tf.nn.relu)
deconv1bis = tf.layers.conv2d_transpose(inputs=deconv1, filters=512, kernel_size=(3, 3), padding='same',
                                        activation=tf.nn.relu)
deconv1bisbis = tf.layers.conv2d_transpose(inputs=deconv1bis, filters=512, kernel_size=(3, 3), padding='same',
                                           activation=tf.nn.relu)

# upsampling layer 2 :
upsample2 = tf.image.resize_images(deconv1bisbis, size=(72, 72), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

# Deconvolutional layer 2 :
deconv2 = tf.layers.conv2d_transpose(inputs=upsample2, filters=512, kernel_size=(3, 3), padding='same',
                                     activation=tf.nn.relu)
deconv2bis = tf.layers.conv2d_transpose(inputs=deconv2, filters=512, kernel_size=(3, 3), padding='same',
                                        activation=tf.nn.relu)
deconv2bisbis = tf.layers.conv2d_transpose(inputs=deconv2bis, filters=512, kernel_size=(3, 3), padding='same',
                                           activation=tf.nn.relu)

# upsampling layer 3 :
upsample3 = tf.image.resize_images(deconv2bisbis, size=(72, 72), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

# Deconvolutional layer 3 :
deconv3 = tf.layers.conv2d_transpose(inputs=upsample3, filters=256, kernel_size=(3, 3), padding='same',
                                     activation=tf.nn.relu)
deconv3bis = tf.layers.conv2d_transpose(inputs=deconv3, filters=256, kernel_size=(3, 3), padding='same',
                                        activation=tf.nn.relu)
deconv3bisbis = tf.layers.conv2d_transpose(inputs=deconv3bis, filters=512, kernel_size=(3, 3), padding='same',
                                           activation=tf.nn.relu)

# upsampling layer 4 :
upsample4 = tf.image.resize_images(deconv3bisbis, size=(72, 72), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

# Deconvolutional layer 4 :
deconv4 = tf.layers.conv2d_transpose(inputs=upsample4, filters=128, kernel_size=(3, 3), padding='same',
                                     activation=tf.nn.relu)
deconv4bis = tf.layers.conv2d_transpose(inputs=deconv4, filters=128, kernel_size=(3, 3), padding='same',
                                        activation=tf.nn.relu)

# upsampling layer 5 :
upsample5 = tf.image.resize_images(deconv4bis, size=(72, 72), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

# Deconvolutional layer 5 :
deconv5 = tf.layers.conv2d_transpose(inputs=upsample5, filters=64, kernel_size=(3, 3), padding='same',
                                     activation=tf.nn.relu)
deconv5bis = tf.layers.conv2d_transpose(inputs=deconv5, filters=64, kernel_size=(3, 3), padding='same',
                                        activation=tf.nn.relu)

# adding DropOut :
# dropout = tf.layers.dropout(deconv5bis, rate=0.5 )
# Logits Layer
logits = tf.layers.dense(inputs=deconv5bis, units=1, activation=tf.nn.relu)

# Softmax function :
# loss = tf.losses.sparse_softmax_cross_entropy(logits=logits)

# define the losse function , optimizer and accuracy , with grpahs :
with tf.variable_scope('train'):
    with tf.variable_scope('LOSS'):
        # loss funtion with graph :
        loss = tf.reduce_mean(tf.squared_difference(y_ph, logits), name='loss')  # defining the loss
        tf.summary.scalar('loss', loss)

    # Define Optimizer :
    with tf.variable_scope('Optimizer'):
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)  # definig the training

    # Define accuracy :
    with tf.variable_scope("Accuracy"):
        # correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y_train, 1), name='correct_pred')  #note
        # accuracy = tf.reduce_mean(tf.cast(tf.equal(logits, y_ph, name='correct_pred'), tf.int8)) #note: if you do reduce_mean, you will have one value, not an image
        accuracy = tf.losses.mean_square_error(labels=y_ph, predictions=logits)
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')  #note
        tf.summary.scalar('accuracy', accuracy)

# Initializing the variables :
init = tf.global_variables_initializer()
merged = tf.summary.merge_all()

###########################
checkpoint = tf.train.latest_checkpoint(SAVE_PATH)

#TRAINING PROCESS WITH SESSion############################################################################################
# sess = tf.InteractiveSession()  # Using InteractiveSession instead of Session to test network in separate cell . # note
with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
    sess.run(init)
    train_writer = tf.summary.FileWriter(logs_path, sess.graph)
    for epoch in range(EPOCHS):
        #todo: shuffle before epoch
        for i in range(num_batch):  #todo: please define your number of batch before
            #X_train, Y_train = f1(epoch)
            #Calculate and display the loss and accuracy
            summary, acc, _, curr_loss = sess.run([merged, accuracy, optimizer, loss], feed_dict={x_ph: X_train[i*batch_size: (i+1)*batch_size], y_ph: Y_train[i*batch_size: (i+1)*batch_size]})   #[i*batch_size: (i+1)*batch_size]......[i*batch_size: (i+1)*batch_size]
            train_writer.add_summary(summary, epoch)
            print('EPOCH = {}, STEO = {}, LOSS = {}, ACCURACY = {}'.format(epoch, i, curr_loss, acc))  #LOSS = {:0.4f}  # the underscore in the beginning is for ignoringg train_step
            #saver = tf.train.saver()
            #path = saver.save(sess, SAVE_PATH + '/' + MODEL_NAME + '.ckpt')
            #print("saved at {}".format(path))
