import numpy as np
import pandas as pd        
import tensorflow as tf

df_train = pd.read_csv('msd.csv')
df_test = pd.read_csv('test.csv')

y_train = df_train['genre'].as_matrix()
df_train = df_train.drop('genre', axis=1)
df_train = df_train.drop('track_id', axis=1)
df_train = df_train.drop('artist_name', axis=1)
df_train = df_train.drop('title', axis=1)

y_test = df_test['genre'].as_matrix()
df_test = df_test.drop('genre', axis=1)
df_test = df_test.drop('track_id', axis=1)
df_test = df_test.drop('artist_name', axis=1)
df_test = df_test.drop('title', axis=1)

features=['loudness', 'tempo', 'time_signature', 'key', 'mode', 'duration', 'avg_timbre1', 'avg_timbre2', 'avg_timbre3', 'avg_timbre4', 'avg_timbre5', 'avg_timbre6', 'avg_timbre7', 'avg_timbre8', 'avg_timbre9', 'avg_timbre10', 'avg_timbre11', 'avg_timbre12', 'var_timbre1', 'var_timbre2', 'var_timbre3', 'var_timbre4', 'var_timbre5', 'var_timbre6', 'var_timbre7', 'var_timbre8', 'var_timbre9', 'var_timbre10', 'var_timbre11', 'var_timbre12', 'max_segment_timbre1', 'max_segment_timbre2', 'max_segment_timbre3', 'max_segment_timbre4', 'max_segment_timbre5', 'max_segment_timbre6', 'max_segment_timbre7', 'max_segment_timbre8', 'max_segment_timbre9', 'max_segment_timbre10', 'max_segment_timbre11', 'max_segment_timbre12']

# Subtracting mean from and dividing by standard deviation
for s in features:
	mean_f = df_train[s].mean()
	std_f = df_train[s].std()
	df_train[s] = (df_train[s] - mean_f) / std_f

for s in features:
	mean_f = df_test[s].mean()
	std_f = df_test[s].std()
	df_test[s] = (df_test[s] - mean_f) / std_f

X_train = df_train.as_matrix()
X_test = df_test.as_matrix()

labels_train = (np.arange(9) == y_train[:,None])
labels_test = (np.arange(9) == y_test[:,None])

inputs = tf.placeholder(tf.float32, shape=(None, X_train.shape[1]), name='inputs')
label = tf.placeholder(tf.float32, shape=(None, 9), name='genre')

# Hidden layer 1
hid1_size = 256
w1 = tf.Variable(tf.random_normal([hid1_size, X_train.shape[1]], stddev=0.01), name='w1')
b1 = tf.Variable(tf.constant(0.1, shape=(hid1_size, 1)), name='b1')
y1 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(w1, tf.transpose(inputs)), b1)), keep_prob=1) #keep_prob=1 for part A as no dropout

# Hidden layer 2
hid2_size = 256
w2 = tf.Variable(tf.random_normal([hid2_size, hid1_size], stddev=0.01), name='w2')
b2 = tf.Variable(tf.constant(0.1, shape=(hid2_size, 1)), name='b2')
y2 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(w2, y1), b2)), keep_prob=1) #keep_prob=1 for part B as no dropout

# Output layer
wo = tf.Variable(tf.random_normal([9, hid2_size], stddev=0.01), name='wo')
bo = tf.Variable(tf.random_normal([9, 1]), name='bo')
yo = tf.transpose(tf.add(tf.matmul(wo, y2), bo))

# Cross Entropy Loss function and Adam optimizer
lr = tf.placeholder(tf.float32, shape=(), name='learning_rate')
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=yo, labels=label))
#optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)

# Prediction
pred = tf.nn.softmax(yo)
pred_label = tf.argmax(pred, 1)
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Referred Google for all this on how to use the Tenserflow library
# Next 6 lines are not my code
# Create operation which will initialize all variables
init = tf.global_variables_initializer()

# Configure GPU not to use all memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Start a new tensorflow session and initialize variables
sess = tf.InteractiveSession(config=config)
sess.run(init)

#Number of iterations
for epoch in range(100):

        avg_cost = 0.0
        batch_size=100
        
        total_batch = int(df_train.shape[0]/batch_size)
        for i in range(total_batch):
           
            _, c = sess.run([optimizer, loss], feed_dict={lr:0.001, inputs: X_train[i, None], label: labels_train[i, None]})
            avg_cost += c/total_batch   

acc_train = accuracy.eval(feed_dict={inputs: X_train, label: labels_train})
print("Train accuracy: {:3.2f}%".format(acc_train*100.0))

acc_test = accuracy.eval(feed_dict={inputs: X_test, label: labels_test})
print("Test accuracy:  {:3.2f}%".format(acc_test*100.0))










