import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data # hand-written reco

mnist = input_data.read_data_sets(train_dir='MNIST_data', one_hot=True)

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_var(shape):
    # Outputs random values from a truncated normal distribution
    init = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial_value=init)

def bias_var(shape):
    init = tf.constant(value=0.1, shape=shape)
    return tf.Variable(init)

# CONV layer
def conv2d(x, W):
    return tf.nn.conv2d(input=x, filter=W, strides=[1,1,1,1], padding='SAME')

# POOL layer
def max_poo_2x2(x):
    # [1, height, width, 1], [1, stride,stride, 1]
    return tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1,2,2,1],padding="SAME")

# image processing
xs = tf.placeholder(dtype=tf.float32,shape=[None, 784]) # 784 column, row undefined
ys = tf.placeholder(dtype=tf.float32, shape=[None, 10])
# -1 ignore number of examples dimension, size  = 28x28, channel = 1 b&w
x_image = tf.reshape(tensor=xs, shape=[-1, 28, 28, 1])

# drop out
keep_prob = tf.placeholder(tf.float32)

# define CONV layer 1
w_conv1 = weight_var([5, 5, 1, 32]) # filter, 1 channel, input is 1
b_conv1 = bias_var([32])
# activation
h_conv1 = tf.nn.relu(features=conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_poo_2x2(h_conv1)

# define CONV layer 2
w_conv2 = weight_var([5, 5, 32, 64]) # filter, 32 feature maps, input is 32
b_conv2 = bias_var([64])
# activation
h_conv2 = tf.nn.relu(features=conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_poo_2x2(h_conv2)


#FC layer
# convert 3d hpool2 to 1d array
h_pool2_flat = tf.reshape(tensor=h_pool2, shape=[-1, 7*7*64])
w_fc1 = weight_var(shape=[7*7*64, 1024])
b_fc1 = bias_var([1024])
h_fc1 = tf.nn.relu(tf.matmul(a=h_pool2_flat, b=w_fc1) + b_fc1)
# keepprob = A scalar Tensor with the same type as x. The probability that each element is kept.
h_fc1_drop = tf.nn.dropout(x=h_fc1, keep_prob=keep_prob)

w_fc2 = weight_var(shape=[1024, 10])
b_fc2 = bias_var(shape=[10])
# softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis)
prediction = tf.nn.softmax(logits=tf.matmul(a=h_fc1_drop, b=w_fc2) + b_fc2)

# cost function
cross_entropy = tf.reduce_mean(
    input_tensor=-tf.reduce_sum(input_tensor=ys * tf.log(x=prediction), reduction_indices=[1]))

# define training algorithm and initalize session
train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss=cross_entropy)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# save training progress
saver = tf.train.Saver()

def train_loop():
    for i in range(501):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
        if i % 50 == 0:
            save_path = saver.save(sess=sess, save_path="./saved_cnn_mnist.ckpt")
            print("saving to %s" % save_path)
            print(compute_accuracy(
                mnist.test.images[:1000], mnist.test.labels[:1000]))


if __name__ == '__main__':
    saver.restore(sess=sess, save_path="./saved_cnn_mnist.ckpt")
    print("loading training model: ")
    print(compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000]))