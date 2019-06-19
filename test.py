import tensorflow as tf
import numpy as np
from PIL import Image
import os
import glob
from random import shuffle



#Path to the folder where to save the logs for tensorboard :
logs_path = "./logs/embedding/"
SAVE_PATH = './save'
EPOCHS = 5
#LEARNING_RATE = 0.000000001
#dropout_rate=0.2
MODEL_NAME = 'model'
batch_size = 1
patch_size = 2
num_batch = 3
#LEARNING_RATE = tf.train.exponential_decay(0.00001, batch_size, 0.95, decay_rate=0.96, staircase=True)  #adaptative learning rate
LEARNING_RATE = 0.0001


# X = []
# for filename in glob.glob('./img/train/input/*.tif'):  # /{}...format(i)
#     im = Image.open(filename)
#     X.append(np.array(im)[:200, :200])  # turn to array and #resize images to (200,200)


# Y = []
# for filename in glob.glob('./img/train/output/*.tif'):
#     im = Image.open(filename)
#     Y.append(np.array(im)[:200, :200])


# # reshape array :
# X_train = np.reshape(X, newshape=(-1, 200, 200, 1))  # [: , 200:300, 200:300, :] #reshape image to have a 4 dimentional tensor and to fit the placeholder
# Y_train = np.reshape(Y, newshape=(-1, 200, 200, 1))  # [: , 200:300, 200:300, :]


# Create graph and Define placeholder for input :
with tf.variable_scope('Input'):
    x_ph = tf.placeholder("float32", [None, 200, 200, 1])
    #tf.summary.image("x_ph" , x_ph)
    #tf.summary.image('input_img',tf.reshape(x, (-1, 200, 200, 1)), max_outputs=5)
# # graph , and define placeholder for output :
with tf.variable_scope('Output'):
    y_ph = tf.placeholder("float32", [None, 200, 200, 1])
    #tf.summary.image("y_ph", y_ph)
   #tf.summary.image('output_img',tf.reshape(x_ph, (-1, 200, 200, 1)),  max_outputs=5)

class MBGDHelper:
    '''Mini Batch Grandient Descen helper'''
    def __init__(self, batch_size, patch_size):
        self.i = 0
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.inpath = './img/train/input/'
        self.outpath = './img/train/output/'
        self.indir = os.listdir(self.inpath)
        self.outdir = os.listdir(self.outpath)
        self.epoch_len = self._epoch_len()
        self.order = np.arange(self.epoch_len) #data has been pre-shuffle
        self.onoff = 0

    def next_batch(self):
        try:
            X = np.array(Image.open(self.inpath + self.indir[self.order[0]], 'r'))[:200, :200]
            y = np.array(Image.open(self.outpath + self.outdir[self.order[0]], 'r'))[:200, :200]
            for j in range(self.i + 1, self.i + self.batch_size):
                X = np.vstack((X, np.array(Image.open(self.inpath + self.indir[self.order[j]]))[:200, :200]))
                y = np.vstack((y, np.array(Image.open(self.outpath + self.outdir[self.order[j]]))[:200, :200]))
            self.i += self.batch_size
            return X.reshape((-1, self.patch_size, self.patch_size, 1)), \
                   y.reshape((-1, self.patch_size, self.patch_size, 1))

            tf.summary.image('input_img', tf.reshape(X, (-1, 200, 200, 1)), max_outputs=5)
            tf.summary.image('output_img', tf.reshape(y, (-1, 200, 200, 1)), max_outputs=5)
#             except:
#                 print('\n***Load last batch')
#                 pass
#                 with h5py.File('./proc/{}.h5'.format(self.patch_size), 'r') as f:
#                     modulo = f['X'].shape % self.batch_size
#                     X = f['X'][self.order[-modulo:, ]].reshape(modulo, self.patch_size, self.patch_size, 1)
#                     y = f['y'][self.order[-modulo:, ]].reshape(modulo, self.patch_size, self.patch_size, 1)
#                 self.i += 1
#                 return X, y
        except Exception as e:
            print(e)
            print('\n***epoch finished')
            self.onoff = 1
            pass

    def _epoch_len(self):
        dirs = os.listdir(self.inpath)
        return len(dirs)

    def get_epoch_len(self):
        return self.epoch_len

    def shuffle(self):
        shuffle(self.order)
        print('shuffled datas')



#The Model
conv1 = tf.layers.conv2d(inputs=x_ph, filters=64, kernel_size=[3, 3], strides=(1, 1), padding="same",
                         activation=tf.nn.relu)
conv1bis = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[3, 3], strides=(1, 1), padding="same",
                            activation=tf.nn.relu)

# Pooling Layer #1
pool1 = tf.layers.max_pooling2d(inputs=conv1bis, pool_size=[2, 2], strides=(1, 1))

# Convolutional Layer #2
conv2 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=[3, 3],strides=(1, 1), padding="same", activation=tf.nn.relu)
conv2bis = tf.layers.conv2d(inputs=conv2, filters=128, kernel_size=[3, 3],strides=(1, 1), padding="same", activation=tf.nn.relu)

# Pooling Layer #2
pool2 = tf.layers.max_pooling2d(inputs=conv2bis, pool_size=[2, 2], strides=(1, 1))

# Convolutional Layer #3
conv3 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=[3, 3], strides=(1, 1), padding="same", activation=tf.nn.relu)
conv3bis = tf.layers.conv2d(inputs=conv3, filters=256, kernel_size=[3, 3],strides=(1, 1),  padding="same", activation=tf.nn.relu)
conv3bisbis = tf.layers.conv2d(inputs=conv3bis, filters=256, kernel_size=[3, 3],strides=(1, 1),  padding="same", activation=tf.nn.relu)

# Pooling Layer #3
pool3 = tf.layers.max_pooling2d(inputs=conv3bisbis, pool_size=[2, 2], strides=(1, 1))

# Convolutional Layer #4
conv4 = tf.layers.conv2d(inputs=pool3, filters=512, kernel_size=[3, 3],strides=(1, 1),  padding="same", activation=tf.nn.relu)
conv4bis = tf.layers.conv2d(inputs=conv4, filters=512, kernel_size=[3, 3],strides=(1, 1), padding="same", activation=tf.nn.relu)
conv4bisbis = tf.layers.conv2d(inputs=conv4bis, filters=512, kernel_size=[3, 3], strides=(1, 1), padding="same", activation=tf.nn.relu)

# Pooling Layer #4
pool4 = tf.layers.max_pooling2d(inputs=conv4bisbis, pool_size=[2, 2], strides=(1, 1) )

# Convolutional Layer #5
conv5 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=[3, 3],strides=(1, 1),  padding="same", activation=tf.nn.relu)
conv5bis = tf.layers.conv2d(inputs=conv5, filters=512, kernel_size=[3, 3],strides=(1, 1),  padding="same", activation=tf.nn.relu)
conv5bisbis = tf.layers.conv2d(inputs=conv5bis, filters=512, kernel_size=[3, 3],strides=(1, 1),  padding="same", activation=tf.nn.relu)

# Pooling Layer #5
pool5 = tf.layers.max_pooling2d(inputs=conv5bisbis, pool_size=[2, 2], strides=(1, 1))

#DECODER################################################################################################################
# upsampling layer 1 :
upsample1 = tf.image.resize_images(pool5, size=(200, 200), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

# Deconvolutional layer 1 :
deconv1 = tf.layers.conv2d_transpose(inputs=upsample1, filters=512, kernel_size=(3, 3),strides=(1, 1),  padding='same',
                                     activation=tf.nn.relu)
deconv1bis = tf.layers.conv2d_transpose(inputs=deconv1, filters=512, kernel_size=(3, 3),strides=(1, 1),  padding='same',
                                        activation=tf.nn.relu)
deconv1bisbis = tf.layers.conv2d_transpose(inputs=deconv1bis, filters=512, kernel_size=(3, 3),strides=(1, 1),  padding='same',
                                           activation=tf.nn.relu)

# upsampling layer 2 :
upsample2 = tf.image.resize_images(deconv1bisbis, size=(200, 200),  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

# Deconvolutional layer 2 :
deconv2 = tf.layers.conv2d_transpose(inputs=upsample2, filters=512,strides=(1, 1),  kernel_size=(3, 3), padding='same',
                                     activation=tf.nn.relu)
deconv2bis = tf.layers.conv2d_transpose(inputs=deconv2, filters=512,strides=(1, 1),  kernel_size=(3, 3), padding='same',
                                        activation=tf.nn.relu)
deconv2bisbis = tf.layers.conv2d_transpose(inputs=deconv2bis, filters=512, strides=(1, 1), kernel_size=(3, 3), padding='same',
                                           activation=tf.nn.relu)

# upsampling layer 3 :
upsample3 = tf.image.resize_images(deconv2bisbis, size=(200, 200),  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

# Deconvolutional layer 3 :
deconv3 = tf.layers.conv2d_transpose(inputs=upsample3, filters=256,strides=(1, 1),  kernel_size=(3, 3), padding='same',
                                     activation=tf.nn.relu)
deconv3bis = tf.layers.conv2d_transpose(inputs=deconv3, filters=256,strides=(1, 1),  kernel_size=(3, 3), padding='same',
                                        activation=tf.nn.relu)
deconv3bisbis = tf.layers.conv2d_transpose(inputs=deconv3bis, filters=512,strides=(1, 1), kernel_size=(3, 3), padding='same',
                                           activation=tf.nn.relu)

# upsampling layer 4 :
upsample4 = tf.image.resize_images(deconv3bisbis, size=(200, 200),  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

# Deconvolutional layer 4 :
deconv4 = tf.layers.conv2d_transpose(inputs=upsample4, filters=128,strides=(1, 1),  kernel_size=(3, 3), padding='same',
                                     activation=tf.nn.relu)
deconv4bis = tf.layers.conv2d_transpose(inputs=deconv4, filters=128,strides=(1, 1),  kernel_size=(3, 3), padding='same',
                                        activation=tf.nn.relu)

# upsampling layer 5 :
upsample5 = tf.image.resize_images(deconv4bis, size=(200, 200),  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

# Deconvolutional layer 5 :
deconv5 = tf.layers.conv2d_transpose(inputs=upsample5, filters=64,strides=(1, 1),  kernel_size=(3, 3), padding='same',
                                     activation=tf.nn.relu)
deconv5bis = tf.layers.conv2d_transpose(inputs=deconv5, filters=64,strides=(1, 1),  kernel_size=(3, 3), padding='same',
                                        activation=tf.nn.relu)

# adding DropOut :
# dropout = tf.layers.dropout(deconv5bis, rate=0.5 )
# Logits Layer
logits = tf.layers.dense(inputs=deconv5bis, units=1, activation=tf.nn.relu)

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
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(logits, tf.int32), tf.cast(y_ph, tf.int32), name='correct_pred'), tf.int32))
        #accuracy = tf.nn.l2_loss(logits - y_ph)    # wrong !! that is a loss function

        tf.summary.scalar('accuracy', accuracy)

# Initializing the variables :
init = tf.global_variables_initializer()
merged = tf.summary.merge_all()

###########################
checkpoint = tf.train.latest_checkpoint(SAVE_PATH)
batch_helper = MBGDHelper(batch_size=batch_size, patch_size=200)
epoch_len = batch_helper.get_epoch_len()

with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
    sess.run(init)
    train_writer = tf.summary.FileWriter(logs_path, sess.graph)
    for epoch in range(EPOCHS):
        # shuffle
        batch_helper.shuffle( )
        for i in range(epoch_len // batch_size):  #numbatch * batchsize = total number of img
            batch = batch_helper.next_batch()
            #X_train, Y_train = f1(epoch)
    #Calculate and display the loss and accuracy
            print(batch[0].shape)
            summary, acc, _, curr_loss = sess.run([merged, accuracy, optimizer, loss], feed_dict={x_ph: batch[0], y_ph: batch[1]})   #[i*batch_size: (i+1)*batch_size]......[i*batch_size: (i+1)*batch_size]
            #acc, _, curr_loss = sess.run([accuracy, optimizer, loss], feed_dict={x_ph: batch[0], y_ph: batch[1]})   #[i*batch_size: (i+1)*batch_size]......[i*batch_size: (i+1)*batch_size]
            # train_writer.add_summary(summary, i)
            print('EPOCH = {}, STEP = {}, LOSS = {}, ACCURACY = {}'.format(epoch, i, curr_loss, acc))  #LOSS = {:0.4f}  # the underscore in the beginning is for ignoringg train_step
               #saver = tf.train.saver()
               #path = saver.save(sess, SAVE_PATH + '/' + MODEL_NAME + '.ckpt')
               #print("saved at {}".format(path))