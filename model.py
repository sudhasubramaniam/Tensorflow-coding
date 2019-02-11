
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import dataset
import random
import cv2
    
# Convolutional Layer 1.
filter_size1 = 5 
num_filters1 = 64

# Convolutional Layer 2.
filter_size2 = 5
num_filters2 = 64

# Convolutional Layer 3.
filter_size3 = 5
num_filters3 = 128

# Convolutional Layer 4.
filter_size4 = 5
num_filters4 = 128

# Convolutional Layer 5.
filter_size5 = 5
num_filters5 = 256

# Convolutional Layer 6.
filter_size6 = 5
num_filters6 = 256
 
# Convolutional Layer 7.
filter_size7 = 5
num_filters7 = 256

# Convolutional Layer 8.
filter_size8 = 5
num_filters8 = 512

# Convolutional Layer 9.
filter_size9 = 5
num_filters9 = 512

# Convolutional Layer 10.
filter_size10 = 5
num_filters10 = 512   

# Fully-connected layer.
fc_size1 = 2048             # Number of neurons in fully-connected layer.

# Fully- connected layer 2
fc_size2 = 2048

# Number of color channels for the images: 1 channel for gray-scale.
num_channels = 3

# image dimensions (only squares for now)
img_size = 75
img_size1=75

# Size of image when flattened to a single dimension
img_size_flat = img_size1 * img_size * num_channels

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size1, img_size)

# class info

classes = ['NN', 'ROI']
num_classes = len(classes)

# batch size
batch_size = 200

# validation split
validation_size = .2
epsilon=1e-2

# how long to wait after validation loss stops improving before terminating training
early_stopping = None  # use None if you don't want to implement early stoping



plt_k=[]
plt_vacc=[]
plt_epoch=[]
plt_loss=[]
plt_acc=[]
fpr1=[]
tpr1=[]
train_path='training data'
#test_path='testing data'


data = dataset.read_train_sets(train_path, img_size1,img_size, classes, validation_size=validation_size)
#test_images, test_ids = dataset.read_test_set(test_path, img_size1,img_size,classes)

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
#print("- Test-set:\t\t{}".format(len(test_images)))
print("- Validation-set:\t{}".format(len(data.valid.labels)))



def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05),name='Weights')

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]), name='Biases')



def new_conv_layer(input, num_input_channels,filter_size,num_filters,name,use_pooling=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
    	             filter=weights,
    	             strides=[1, 1, 1, 1],
    	             padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
    # Batch normaliza																												tion
    batch_mean,  batch_var= tf.nn.moments(layer,[0])
    scale = tf.Variable(tf.ones([num_filters]))
    beta = tf.Variable(tf.zeros([num_filters]))
    BN = tf.nn.batch_normalization(layer,batch_mean,batch_var,beta,scale,epsilon)

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.elu(BN)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights

    
																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																										
def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features


def new_fc_layer(input,          # The previous layer.
             num_inputs,     # Num. inputs from prev. layer.
             num_outputs,    # Num. outputs.
	     name,
             use_elu=True,
             drop_out=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_elu:
        layer = tf.nn.elu(layer)

    return layer
    if drop_out:
        layer = tf.layers.dropout(layer, rate=0.8)
    return layer

session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size1, img_size, num_channels])


y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

with tf.name_scope('Conv'):

 layer_conv1, weights_conv1 = \
 new_conv_layer(input=x_image,
               num_input_channels=num_channels,
               filter_size=filter_size1,
               num_filters=num_filters1,
               use_pooling=False,
               name='Conv1')
#cv2.imwrite("/home/ksr-ece/Downloads/EMBC/RECENT CODING/coding new/features/layer1.jpg",layer_conv1)
#cv2.imwrite("/home/ksr-ece/Downloads/EMBC/RECENT CODING/coding new/features/weight1.jpg",weights_conv1)
#print("now layer2 input")
#print(layer_conv1.get_shape()) 
    
with tf.name_scope('Conv'):
 layer_conv2, weights_conv2 = \
 new_conv_layer(input=layer_conv1,
               num_input_channels=num_filters1,
               filter_size=filter_size2,
               num_filters=num_filters2,
               use_pooling=True,name='Conv2')
#print("now layer3 input")
#print(layer_conv2.get_shape())     

with tf.name_scope('Conv'):               
 layer_conv3, weights_conv3 = \
 new_conv_layer(input=layer_conv2,
               num_input_channels=num_filters2,
               filter_size=filter_size3,
               num_filters=num_filters3,
               use_pooling=False, name='Conv3')
#print("now layer flatten input")
#print(layer_conv3.get_shape()) 

with tf.name_scope('Conv'):
 layer_conv4, weights_conv4 = \
 new_conv_layer(input=layer_conv3,
               num_input_channels=num_filters3,
               filter_size=filter_size4,
               num_filters=num_filters4,
               use_pooling=True,name='Conv4')

with tf.name_scope('Conv'):
 layer_conv5, weights_conv5 = \
 new_conv_layer(input=layer_conv4,
               num_input_channels=num_filters4,
               filter_size=filter_size5,
               num_filters=num_filters5,
               use_pooling=False,name='Conv5')

with tf.name_scope('Conv'):
 layer_conv6, weights_conv6 = \
 new_conv_layer(input=layer_conv5,
               num_input_channels=num_filters5,
               filter_size=filter_size6,
               num_filters=num_filters6,
               use_pooling=False,name='Conv6')

with tf.name_scope('Conv'):
 layer_conv7, weights_conv7 = \
 new_conv_layer(input=layer_conv6,
               num_input_channels=num_filters6,
               filter_size=filter_size7,
               num_filters=num_filters7,
               use_pooling=True,name='Conv7')

with tf.name_scope('Conv'):
 layer_conv8, weights_conv8 = \
 new_conv_layer(input=layer_conv7,
               num_input_channels=num_filters7,
               filter_size=filter_size8,
               num_filters=num_filters8,
               use_pooling=False,name='Conv8')

with tf.name_scope('Conv'):
 layer_conv9, weights_conv9 = \
 new_conv_layer(input=layer_conv8,
               num_input_channels=num_filters8,
               filter_size=filter_size9,
               num_filters=num_filters9,
               use_pooling=False,name='Conv9')

with tf.name_scope('Conv'):
 layer_conv10, weights_conv10 = \
 new_conv_layer(input=layer_conv9,
               num_input_channels=num_filters9,
               filter_size=filter_size10,
               num_filters=num_filters10,
               use_pooling=True,name='Conv10')    
          
with tf.name_scope('FC1'):
 layer_flat, num_features = flatten_layer(layer_conv10)

 layer_fc1 = new_fc_layer(input=layer_flat,
                     num_inputs=num_features,
                     num_outputs=fc_size1,
                     use_elu=True,
                     drop_out=True,name='FC1')

with tf.name_scope('FC2'):
 layer_fc2 = new_fc_layer(input=layer_fc1,
                     num_inputs=fc_size1,
                     num_outputs=fc_size2,
                     use_elu=True,
                     drop_out=True,name='FC2')

with tf.name_scope('FC3'):
 layer_fc3 = new_fc_layer(input=layer_fc2,
                     num_inputs=fc_size1,
                     num_outputs=num_classes,
                     use_elu=False,name='FC3')

with tf.name_scope('Model'):
 y_pred = tf.nn.softmax(layer_fc3,name='y_pred')
 y_pred_cls = tf.argmax(y_pred, dimension=1)

with tf.name_scope('Loss'):
 with tf.name_scope('Cross_entropy'):
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc3,
                                                    labels=y_true)
 with tf.name_scope('Cost'):
  cost = tf.reduce_mean(cross_entropy)

with tf.name_scope('Adam'):
 optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

with tf.name_scope('Acc'):
 correct_prediction = tf.equal(y_pred_cls, y_true_cls)
 accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='Accuracy')

init = tf.global_variables_initializer()

#session.run(tf.global_variables_initializer()) # for newer versions
#session.run(tf.initialize_all_variables()) # for older versions
train_batch_size = batch_size

### FOR TENSORBOARD TUTORIAL ONLY
#writer= tf.summary.FileWriter('/tmp/tensorboard_tut')
#writer.add_graph(session.graph)

tf.summary.scalar("loss", cost)
tf.summary.scalar("accuracy", accuracy)
merged_summary_op = tf.summary.merge_all()

total_iterations = 0
train_x_all = data.train.images
train_y_all = data.train.labels
test_x = data.valid.images
test_y = data.valid.labels

plt_acc1=[]
plt_loss1=[]
plt_epoch1=[]


def run_train(session, train_x, train_y,cl):
 with tf.name_scope('Train'):
  print ("\nStart training")
  session.run(init) 
  total_batch = int(train_x.shape[0] / batch_size)
  plt_acc=[]
  plt_loss=[]
  plt_epoch=[]

  def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations
    j=0
    best_val_loss = float("inf")

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        batch_x = train_x[j*batch_size:(j+1)*batch_size]
        batch_y = train_y[j*batch_size:(j+1)*batch_size]
        batch_x=batch_x.reshape(batch_size, img_size_flat)
        _, c, summary = session.run([optimizer, cost,merged_summary_op], feed_dict={x: batch_x, y_true: batch_y})
        saver = tf.train.Saver()
        saver.save(session, 'my_test_model_VGG_10fold_BN_64(5x5)') 
        j=j+1
        if j == total_batch :
            j = 0

        # Print status at end of each epoch (defined as full pass through training dataset).
        if i % int(total_batch) == 0: 
            #print(i)
            epoch = int(i / int(total_batch))
            #print(total_iterations + num_iterations, i) 
            train_all=train_x.reshape(len(train_y), img_size_flat)
            acc,summary = session.run([accuracy, merged_summary_op],feed_dict={x: train_all, y_true: train_y})
            #msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, cost: {3:.3f}"
            #print(msg.format(epoch + 1, acc, cost))
            print ("Epoch #%d Accuracy=%f cost=%f" % (epoch, acc, c))
            plt_epoch.append(epoch)
            plt_loss.append(c)
            plt_acc.append(acc)
            summary_writer.add_summary(summary, epoch * total_batch + i)
    
	    
    #total_iterations += num_iterations
    total_iterations =0
    
  optimize(num_iterations=70)
  #total_iterations = 0
  plt_acc1.append(plt_acc)
  plt_loss1.append(plt_loss)
 
  plt_epoch1.append(plt_epoch)

with tf.name_scope('Validate'):
 def cross_validate(session, split_size=10):
   results = []
   k=0
   kf = KFold(n_splits=split_size)
   cl=0
   for train_idx, val_idx in kf.split(train_x_all, train_y_all):
    train_x = train_x_all[train_idx]
    train_y = train_y_all[train_idx]
    val_x = train_x_all[val_idx]
    val_y = train_y_all[val_idx]
    run_train(session, train_x, train_y,cl)
    cl=cl+1
    length = len(val_y)
    val_x = val_x.reshape(length, img_size_flat)
    accv=session.run(accuracy, feed_dict={x: val_x, y_true: val_y})
    results.append(accv)
    plt_k.append(k)
    k=k+1
   plt.style.use("ggplot")
   
   for p in range(0,cl):

    plt.figure(1)
    #palette = plt.get_cmap('Set1')
    clr1=['green','gray','maroon','olive','purple','teal','navy','blue','peru','red']
    x2=plt_epoch1[p]
    y2=plt_acc1[p]
    
    plt.plot(x2,y2,color=clr1[p])
    #plt.hold(True)
    plt.xlim([0,len(x2)])
    plt.ylim([0.0,1.2])
    plt.rcParams['font.size']=10
    plt.title('Training Accuracy curve of Intima media segmentation')
    plt.xlabel('EPOCH')
    plt.ylabel('ACCURACY')
    plt.legend(['1-Fold','2-Fold','3-Fold','4-Fold','5-Fold','6-Fold','7-Fold','8-Fold','9-Fold','10-Fold'])
    plt.figure(2)
    #palette = plt.get_cmap('Set1')
    clr2=['green','gray','maroon','olive','purple','teal','navy','blue','peru','red']

    u=plt_epoch1[p]
    v=plt_loss1[p]
    plt.plot(u,v,color=clr2[p])
    #plt.hold(True)
    plt.xlim([0,len(u)])
    plt.ylim([0.0,50])
    plt.rcParams['font.size']=10
    plt.title('Training Loss curve of Intima media segmentation')
    plt.xlabel('EPOCH')
    plt.ylabel('LOSS')
    plt.legend(['1-Fold','2-Fold','3-Fold','4-Fold','5-Fold','6-Fold','7-Fold','8-Fold','9-Fold','10-Fold'])
   plt.show()
   plt.grid(True)

   '''plt.figure(3)
   plt.plot(k,results)
   plt.xlim([0,k])
   plt.ylim([0.0,1.2])
   plt.rcParams['font.size']=10
   plt.title('Validation Accuracy curve of Intima media segmentation')
   plt.xlabel('KFOLD')
   plt.ylabel('ACCURACY') '''   
   return results

with tf.Session() as session:
  summary_writer = tf.summary.FileWriter('/home/ksr-ece/Downloads/EMBC/RECENT CODING', graph=tf.get_default_graph())
  result = cross_validate(session)
  print ("Cross-validation result: %s" % result)
  length = len(test_y)
  test_x = test_x.reshape(length, img_size_flat)
  print ("Test accuracy: %f" % session.run(accuracy,feed_dict={x: test_x, y_true: test_y}))
  






