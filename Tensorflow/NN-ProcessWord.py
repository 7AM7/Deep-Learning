import nltk
import tensorflow as tf
from nltk.tokenize import word_tokenize
import numpy as np
import random
import pickle
from collections import Counter
from nltk.stem import WordNetLemmatizer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

lemmatizer = WordNetLemmatizer()
hm_lines = 100000
POS_FILE_NAME = "data/pos.txt"
NEG_FILE_NAME = "data/neg.txt"
OUPUT_FILE_NAME = "data/output.txt"

def create_lexicon(pos,neg):
	lexicon = []
	for fi in [pos, neg] :
		with open(fi,'r') as f:
			contents = f.readlines()
			for l in contents[:hm_lines]:
				all_words = word_tokenize(l.lower())
				lexicon += list(all_words)

	lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
	w_counts = Counter(lexicon)
	l2 = []
	for w in w_counts:
		if 1000 > w_counts[w] > 50:
			l2.append(w_counts[w])
	print("lexicon length: ",len(l2))
	with open('data/lexicon.pickle','wb') as f:
		pickle.dump(list(l2),f)
	return l2

'''
classification = [1 0] pos data		[0 1] neg data
'''
def create_feature_pos_and_neg_file(sample,lexicon,classification):
	featureset = []
	with open(sample,'r') as f:
		contents = f.readlines()
		for l in contents[:hm_lines]:
			current_words = word_tokenize(l.lower())
			current_words = [lemmatizer.lemmatize(i) for i in current_words]
			features = np.zeros(len(lexicon))
			for word in current_words:
				if word.lower() in lexicon:
					index_value = lexicon.index(word.lower())
					features[index_value] += 1

			features = list(features)
			featureset.append([features,classification])

	return featureset

def create_feature_sets_and_labels(pos, neg, test_size = 0.1):
	lexicon = []
	if not os.path.exists('lexicon.pickle'):
		lexicon = create_lexicon(pos, neg)
	else:	
		with open('lexicon.pickle','rb') as f:
			lexicon = pickle.load(f)
	if len(lexicon) <= 0:
		lexicon = create_lexicon(pos, neg)

	features = []
	features += create_feature_pos_and_neg_file(POS_FILE_NAME, lexicon,[1,0])
	features += create_feature_pos_and_neg_file(NEG_FILE_NAME, lexicon,[0,1])
	random.shuffle(features)
	features = np.array(features)

	testing_size = int(test_size*len(features))

	train_x = list(features[:,0][:-testing_size])
	train_y = list(features[:,1][:-testing_size])
	test_x = list(features[:,0][-testing_size:])
	test_y = list(features[:,1][-testing_size:])	

	with open(OUPUT_FILE_NAME,'wb') as f:
		pickle.dump([train_x,train_y,test_x,test_y], f)

	return train_x,train_y,test_x,test_y


##########################################################  DEEP NUERAL #####################################
n_nodes_hl1 = 100
n_nodes_hl2 = 100
#n_nodes_hl3 = 500

n_classes = 2
batch_size = 50
#total_batches = int(1600000/batch_size)
epochs = 15
current_epoch = tf.Variable(1)
x = tf.placeholder('float')
y = tf.placeholder('float')

if os.path.exists(OUPUT_FILE_NAME):
	with open(OUPUT_FILE_NAME,'rb') as f:
		train_x,train_y,test_x,test_y = pickle.load(f)
else:
    train_x,train_y,test_x,test_y = create_feature_sets_and_labels(POS_FILE_NAME,NEG_FILE_NAME)

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    # hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
    #                   'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes]))}

    # (input data * weights) + biases

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    # l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    # l3 = tf.nn.relu(l3)

    output = tf.matmul(l2, output_layer['weights']) + output_layer['biases']

    return output

saver = tf.train.Saver(tf.global_variables())
tf_log = 'data/tf.log'

def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
	optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		try:
			epoch = int(open(tf_log,'r').read().split('\n')[-2])+1
			print('STARTING:',epoch)
		except:
			epoch = 1
		
	#	for epoch in range(epochs):
		while epoch <= epochs:
			epoch_loss = 1
			i = 0
			batches_run = 0
			while i < len(train_x):
				start = i
				end = i + batch_size
				batch_x = np.array(train_x[start:end])
				batch_y = np.array(train_y[start:end])
				if len(batch_x) >= batch_size:
					_, c = sess.run([optimizer, cost], feed_dict={x: batch_x,  # c = cost
				                                              y: batch_y})
					batches_run +=1
					print('Batch run:',batches_run,'/',len(train_x/batch_size),'| Epoch:',epoch,'| Batch Loss:',c,)
				
				epoch_loss += c
				i += batch_size
			saver.save(sess, "data/model.ckpt")	
			print('Epoch', epoch, 'completed out of',epochs,'loss:',epoch_loss)
			with open(tf_log,'a') as f:
				f.write(str(epoch)+'\n') 
			epoch += 1

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

		print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))

def use_neural_network(input_data):
	prediction = neural_network_model(x)
	with open('data/lexicon.pickle','rb') as f:
		lexicon = pickle.load(f)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())		
		#saver = tf.train.Saver()
		try:
			saver.restore(sess,"data/model.ckpt")
			#print('Restored!')
		except Exception as e:
			print(str(e))
		
		current_words = word_tokenize(input_data.lower())
		current_words = [lemmatizer.lemmatize(i) for i in current_words]
		features = np.zeros(len(lexicon))

		for word in current_words:
		  	if word.lower() in lexicon:
			    index_value = lexicon.index(word.lower())
			    features[index_value] += 1

		features = np.array(list(features))
        # pos: [1,0] , argmax: 0
        # neg: [0,1] , argmax: 1
		result = (sess.run(tf.argmax(prediction.eval(feed_dict={x:[features]}),1)))
		if result[0] == 0:
			print('Positive:',input_data)
		elif result[0] == 1:
			print('Negative:',input_data)	    

train_neural_network(x)
use_neural_network("He's an idiot and a jerk.")
use_neural_network("This was the best store i've ever seen.")