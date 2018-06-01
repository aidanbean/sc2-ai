# add layer in tensorflow

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs, in_size, out_size, activation_func = None):
    weights = tf.Variable(initial_value=tf.random_normal([in_size, out_size]))
    bias = tf.Variable(initial_value=tf.zeros([1, out_size]) + 0.1) # [0.1 ...., 0.1] size = outsize
    wx_plus_b = tf.matmul(a=inputs, b=weights) + bias

    if activation_func is None:
        return wx_plus_b
    else:
        return activation_func(wx_plus_b)

x_data = np.linspace(start=-1,stop=1,num=300,dtype=np.float32)[:, np.newaxis] # convert to 300 rows
noise = np.random.normal(loc=0,scale=0.05,size=x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(dtype=tf.float32, shape=[None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# hidden layer
l1 = add_layer(inputs=xs, in_size=1, out_size=10, activation_func=tf.nn.relu)
# output layer
prediction = add_layer(l1, in_size=10, out_size=1, activation_func=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1])) # 1/ n (sum (difference^ 2))
# reduce indices = The old (deprecated) name for axis, axis = axis for summation

# define training algorithm
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

# initalization for all variable for tensorflow
init = tf.global_variables_initializer()

# start session and
sess = tf.Session()
sess.run(init)


# visualization
fig = plt.figure()
ax= fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion() # interaction mode
plt.show()

if __name__ == '__main__':
    for i in range(1000):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            pred = sess.run(prediction, feed_dict={xs: x_data, ys: y_data})
            lines = ax.plot(x_data, pred, 'r-', lw=5)
            plt.pause(0.1)


