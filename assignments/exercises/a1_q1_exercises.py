"""
Simple TensorFlow exercises
You should thoroughly test your code
"""

import tensorflow as tf

sess = tf.InteractiveSession()
###############################################################################
# 1a: Create two random 0-d tensors x and y of any distribution.
# Create a TensorFlow object that returns x + y if x > y, and x - y otherwise.
# Hint: look up tf.cond()
# I do the first problem for you
###############################################################################

x = tf.random_uniform([])  # Empty array as shape
# creates a scalar.
y = tf.random_uniform([])
out = tf.cond(tf.less(x, y), lambda: tf.add(x, y), lambda: tf.subtract(x, y),
              name='cond')
print(out.eval())

###############################################################################
# 1b: Create two 0-d tensors x and y randomly selected from -1 and 1.
# Return x + y if x < y, x - y if x > y, 0 otherwise.
# Hint: Look up tf.case().
###############################################################################

x2 = tf.random_uniform(shape=[], minval=-1, maxval=1)
y2 = tf.random_uniform(shape=[], minval=-1, maxval=1)
out2 = tf.case(pred_fn_pairs={tf.less(x, y): lambda: tf.add(x, y),
                              tf.greater(x, y): lambda: tf.subtract(x, y)},
               default=lambda: tf.constant(0.0), exclusive=True,
               name='case')
print(out2.eval())

###############################################################################
# 1c: Create the tensor x of the value [[0, -2, -1], [0, 1, 2]]
# and y as a tensor of zeros with the same shape as x.
# Return a boolean tensor that yields Trues if x equals y element-wise.
# Hint: Look up tf.equal().
###############################################################################

x3 = tf.constant(value=[[0, -2, -1], [0, 1, 2]])
y3 = tf.zeros_like(x3)
out3 = tf.equal(x3, y3, name='equal')

print(out3.eval())

###############################################################################
# 1d: Create the tensor x of value
# [29.05088806,  27.61298943,  31.19073486,  29.35532951,
#  30.97266006,  26.67541885,  38.08450317,  20.74983215,
#  34.94445419,  34.45999146,  29.06485367,  36.01657104,
#  27.88236427,  20.56035233,  30.20379066,  29.51215172,
#  33.71149445,  28.59134293,  36.05556488,  28.66994858].
# Get the indices of elements in x whose values are greater than 30.
# Hint: Use tf.where().
# Then extract elements whose values are greater than 30.
# Hint: Use tf.gather().
###############################################################################

x4 = tf.constant(value=[29.05088806, 27.61298943, 31.19073486, 29.35532951,
                        30.97266006, 26.67541885, 38.08450317, 20.74983215,
                        34.94445419, 34.45999146, 29.06485367, 36.01657104,
                        27.88236427, 20.56035233, 30.20379066, 29.51215172,
                        33.71149445, 28.59134293, 36.05556488, 28.66994858]
                 )
pos = tf.where(tf.greater(x4, tf.constant(30.0)))

# tf.gather is a tensor transformation, not a controlflow like cond, case and
# where
vals = tf.gather(params=x4, indices=pos, name='where_gather')

print(vals.eval())

###############################################################################
# 1e: Create a diagnoal 2-d tensor of size 6 x 6 with the diagonal values of 1,
# 2, ..., 6
# Hint: Use tf.range() and tf.diag().
###############################################################################

diag_vals = tf.range(start=1, limit=7)
diag = tf.diag(diagonal=diag_vals, name='diag')

print(diag.eval())


###############################################################################
# 1f: Create a random 2-d tensor of size 10 x 10 from any distribution.
# Calculate its determinant.
# Hint: Look at tf.matrix_determinant().
###############################################################################

rand = tf.random_normal(shape=[10, 10])
det = tf.matrix_determinant(input=rand, name='det')

print(rand.eval(), '\n',
      det.eval())
###############################################################################
# 1g: Create tensor x with value [5, 2, 3, 5, 10, 6, 2, 3, 4, 2, 1, 1, 0, 9].
# Return the unique elements in x
# Hint: use tf.unique(). Keep in mind that tf.unique() returns a tuple.
###############################################################################

x5 = tf.constant(value=[5, 2, 3, 5, 10, 6, 2, 3, 4, 2, 1, 1, 0, 9])
uni, idx = tf.unique(x5, name='unique')

print(x5.eval(), '\n',
      uni.eval())
###############################################################################
# 1h: Create two tensors x and y of shape 300 from any normal distribution,
# as long as they are from the same distribution.
# Use tf.less() and tf.select() to return:
# - The mean squared error of (x - y) if the average of all elements in (x - y)
#   is negative, or
# - The sum of absolute value of all elements in the tensor (x - y) otherwise.
# Hint: see the Huber loss function in the lecture slides 3.
###############################################################################

x6 = tf.random_normal(shape=[300])
y6 = tf.random_normal(shape=[300])
diff = tf.subtract(x6, y6)
avg = tf.reduce_mean(diff, axis=0)
result = tf.cond(pred=tf.less(avg, 0),
                 fn1=lambda: 0.5 * tf.reduce_sum(diff**2),
                 fn2=lambda: tf.reduce_sum(tf.abs(diff)),
                 name='mse_abs')

print('diff: ', diff.eval(), '\n',
      'avg:', avg.eval(), '\n',
      'result: ', result.eval())

writer = tf.summary.FileWriter(logdir='./graphs',
                               graph=sess.graph)
# print(tf.get_default_graph().as_graph_def())
sess.close()
