''' 
Movie Recommendation System 
Data: Movie Lens data: 1 Million data with 1,000,209 anonymous ratings 
Download: http://files.grouplens.org/datasets/movielens/ml-1m.zip 
Columns: user ID, movie ID, rating, and timestamp
codes mainly used from : https://github.com/karthikmswamy/RecSys/blob/master/Test_RecSys.ipynb
'''
import time
from collections import deque #Perform append, pop, popleft, appendleft methods on deque 
from six import next #Python 2, 3 compatibility 
import csv
import preprocess
import tensorflow as tf
import numpy as np

np.random.seed(77)

#number of users in the dataset
usr_num = 6040 
#number of movies in the dataset
movie_num = 3952

batch_size = 1000
dims = 15 #dimension of data
max_epochs = 15

place_device = "/cpu:0"

def get_data():
	df = preprocess.read_file("ratings.dat", sep="::")
	rows = len(df)
	#Integer location based indexing for selection by position
	df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)

	#Separate data: 90% Train, 10% Test -- think about validation set later
	split_index = int(rows * 0.9)
	df_train = df[0:split_index]
	df_test = df[split_index:].reset_index(drop=True)

	return df_train, df_test

def clip(x):
	#Given an interval, values outside the interval are clipped to the interval edges. 
	#For example, if an interval of [0, 1] is specified, values smaller than 0 become 0, and values larger than 1 become 1.
	return np.clip(x, 1.0, 5.0) 

def model(user_batch, item_batch, user_num, item_num, dim=5, device="/cpu:0"):
	with tf.device("/cpu:0"):
		bias_global = tf.get_variable("bias_global", shape=[])

		w_bias_user = tf.get_variable("embd_bias_user", shape=[user_num])
		w_bias_item = tf.get_variable("embd_bias_item", shape=[item_num])
		# bias embedding lookup: look up ids in a list of embedding tensors 
		bias_user = tf.nn.embedding_lookup(w_bias_user, user_batch, name="bias_user")
		bias_item = tf.nn.embedding_lookup(w_bias_item, item_batch, name="bias_item")

		w_user = tf.get_variable("embd_user", shape=[user_num, dim], initializer = tf.truncated_normal_initializer(stddev=0.02))
		w_item = tf.get_variable("embd_item", shape=[item_num, dim], initializer = tf.truncated_normal_initializer(stddev=0.02))
		#weight embeddings lookup 
		embd_user = tf.nn.embedding_lookup(w_user, user_batch, name="embedding_user")
		embd_item = tf.nn.embedding_lookup(w_item, item_batch, name="embedding_item")

	with tf.device(device):
		#Reduce sum: compute sum of elements across dimensions of a tensor 
		infer = tf.reduce_sum(tf.multiply(embd_user, embd_item), 1)
		infer = tf.add(inter, bias_global)
		infer = tf.add(inter, bias_user)
		infer = tf.add(infer, bias_item, name="svd_inference")

		#Use L2 Loss
		regularizer = tf.add(tf.nn.l2_loss(embd_user), tf.nn.l2_loss(embd_item), name="svd_regularizer")
	return infer, regularizer

def loss(infer, regularizer, rate_batch, learning_rate = 0.1, reg=0.1, device = "/cpu:0"):
	with tf.device(device):
		#L2 loss to compute penalty
		cost_l2 = tf.nn.l2_loss(tf.subtract(infer, rate_batch))
		penalty = tf.constant(reg, dtype=tf.float32, shape=[], name="l2")
		cost = tf.add(cost_l2, tf.multiply(regularizer, penalty))
		train = tf.train.FtrlOptimizer(learning_rate).minimize(cost)
	return cost, train

df_train, df_test = get_data()

samples_per_batch = len(df_train) // batch_size
print("Number of train samples: %d \nNumber of test samples: %d \nSamples per batch: %d" % (len(df_train), len(df_test), samples_per_batch))

#Use shuffle iterator to generate random batches for training
iter_train = preprocess.ShuffleIterator([df_train["user"], df_train["item"], df_train["rate"]], batch_size=batch_size)
#Generate one-epoch batches for testing
iter_test = preprocess.OneEpochIterator([df_test["user"], df_test["item"], df_test["rate"]], batch_size=-1)

user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
rate_batch = tf.placeholder(tf.float32, shape=[None])

infer, regularizer = model(user_batch, item_batch, user_num=usr_num, item_num=movie_num, dim=dims, device=place_device)
_, train_op = loss(infer, regularizer, rate_batch, learning_rate = 0.10, reg = 0.05, device = place_device)

saver = tf.train.Saver()
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init_op)
	print("%s\t%s\t%s\t%s" % ("Epoch", "Train Error", "Val Error", "Elapsed Time"))
	errors = deque(maxlen=samples_per_batch)
	start = time.time()
	for i in range(max_epochs * samples_per_batch):
		users, items, rates = next(iter_train)
		_, pred_batch = sess.run([train_op, infer], feed_dict={user_batch: users, item_batch: items, rate_batch: rates})
		pred_batch = clip(pred_batch)
		errors.append(np.power(pred_batch - rates, 2))
		if i % samples_per_batch == 0:
			train_err = np.sqrt(np.mean(errors))
			test_err2 = np.array([])
			for users, items, rates in iter_test:
				pred_batch = sess.run(infer, feed_dict={user_batch:users, item_batch:items})
				pred_batch = clip(pred_batch)
				test_err2 = np.append(test_err2, np.power(pred_batch - rates, 2))
			end = time.time()

			print("%02d\t%.3f\t\t%.3f\t\t%.3f secs" %(i // samples_per_batch, train_err, np.sqrt(np.mean(test_err2)), end - start))
			start = end

	saver.save(sess, './save/model')


#REAL Inference! 
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
	#sess.run(init_op) 
	new_saver = tf.train.import_meta_graph('./save/model.meta')
	new_saver.restore(sess, tf.train.latest_checkpoint('./save/'))
	test_err2 = np.array([])
	for users, items, rates in iter_test:
		pred_batch = sess.run(infer, feed_dict={user_batch:users, item_batch:items})
		pred_batch = clip(pred_batch)

		print("Pred\tActual")
		for a in range(10):
			print("%.3f\t%.3f" %(pred_batch[a], rate[a]))
		test_err2 = np.append(test_err2, np.power(pred_batch - rates, 2))
		print(np.sqrt(np.mean(test_err2)))

























