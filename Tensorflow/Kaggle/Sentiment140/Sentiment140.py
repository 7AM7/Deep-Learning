import nltk
import tensorflow as tf
from nltk.tokenize import word_tokenize
import numpy as np
import random
import pickle
from collections import Counter
from nltk.stem import WordNetLemmatizer
import os, time
import pandas as pd
import subprocess
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
lemmatizer = WordNetLemmatizer()

'''
polarity 0 = negative. 2 = neutral. 4 = positive.
id
date
query
user
tweet
'''

def init_process(fin,fout):
    outfile = open(fout,'a')
    with open(fin, buffering=200000, encoding='latin-1') as f:
        try:
            for line in f:
                line = line.replace('"','')
                initial_polarity = line.split(',')[0]
                if initial_polarity == '0':
                    initial_polarity = [1,0]  ## for positve
                elif initial_polarity == '4': ## for negitve
                    initial_polarity = [0,1]

                tweet = line.split(',')[-1]
                outline = str(initial_polarity)+':::'+tweet
                outfile.write(outline)
        except Exception as e:
            print(str(e))
    outfile.close()
#init_process('Kaggle/Sentiment140/training.1600000.processed.noemoticon.csv','Kaggle/Sentiment140/train_set.csv')
#init_process('Kaggle/Sentiment140/testdata.manual.2009.06.14.csv','Kaggle/Sentiment140/test_set.csv')

def create_lexicon(fin):
    lexicon = []
    with open(fin, 'r', buffering=100000, encoding='latin-1') as f:
        try:
            counter = 1
            content = ''
            for line in f:
                counter += 1
                if (counter/2500.0).is_integer():
                    tweet = line.split(':::')[1]
                    content += ' '+tweet
                    words = word_tokenize(content)
                    words = [lemmatizer.lemmatize(i) for i in words]
                    lexicon = list(set(lexicon + words))
               
            print(counter, len(lexicon))
        except Exception as e:
            print(str(e))

    with open('Kaggle/Sentiment140/lexicon.pickle','wb') as f:
        pickle.dump(lexicon,f)

#create_lexicon('Kaggle/Sentiment140/train_set.csv')

def convert_to_vec(fin, fout, lexicon_pickle):
    with open(lexicon_pickle,'rb') as f:
        lexicon = pickle.load(f)
    outfile = open(fout,'a')
    with open(fin, buffering=20000, encoding='latin-1') as f:
	    counter = 0
	    for line in f:
	    	counter +=1
	    	label = line.split(':::')[0]
	    	tweet = line.split(':::')[1]

	    	current_words = word_tokenize(tweet.lower())
	    	current_words = [lemmatizer.lemmatize(i) for i in current_words]
	    	features = np.zeros(len(lexicon))

	    	for word in current_words:
	    		if word.lower() in lexicon:
	    			index_value = lexicon.index(word.lower())
	    			features[index_value] += 1

	    	features = list(features)
	    	outline = str(features)+'::'+str(label)+'\n'
	    	outfile.write(outline)
	    print(counter)

#convert_to_vec('Kaggle/Sentiment140/test_set.csv','Kaggle/Sentiment140/processed-test-set.csv','Kaggle/Sentiment140/lexicon.pickle')

def shuffle_data(fin):
    df = pd.read_csv(fin, error_bad_lines=False)
    df = df.iloc[np.random.permutation(len(df))]
    print(df.head())
    df.to_csv('Kaggle/Sentiment140/train_set_shuffled.csv', index=False)
	
#shuffle_data('Kaggle/Sentiment140/train_set.csv')
def create_test_data_pickle(fin):
    feature_sets = []
    labels = []
    counter = 0
    with open(fin, buffering=20000) as f:
        for line in f:
            try:
                features = list(eval(line.split('::')[0]))
                label = list(eval(line.split('::')[1]))
                feature_sets.append(features)
                labels.append(label)
                counter += 1
            except:
            	pass
    print('Tested',counter,'samples.')
    feature_sets = np.array(feature_sets)
    labels = np.array(labels)
    return feature_sets, labels



##########################################################  DEEP NUERAL #####################################
n_nodes_hl1 = 500
n_nodes_hl2 = 500

n_classes = 2

batch_size = 32
total_batches = int(1600000/batch_size)
hm_epochs = 8
current_epoch = tf.Variable(1)
x = tf.placeholder('float')
y = tf.placeholder('float')


def neural_network_model(data):

    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([2638, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)
    output = tf.matmul(l2, output_layer['weights']) + output_layer['biases']

    return output

saver = tf.train.Saver(tf.global_variables())
tf_log = 'Kaggle/Sentiment140/tf.log'

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

        while epoch <= hm_epochs:
            if epoch != 1:
                saver.restore(sess,"Kaggle/Sentiment140/model.ckpt")
            epoch_loss = 1
            with open('Kaggle/Sentiment140/lexicon.pickle','rb') as f:
                lexicon = pickle.load(f)
            with open('Kaggle/Sentiment140/train_set_shuffled.csv', buffering=20000, encoding='latin-1') as f:
                batch_x = []
                batch_y = []
                batches_run = 0
                for line in f:
                    label = line.split(':::')[0]
                    tweet = line.split(':::')[1]
                    current_words = word_tokenize(tweet.lower())
                    current_words = [lemmatizer.lemmatize(i) for i in current_words]

                    features = np.zeros(len(lexicon))

                    for word in current_words:
                        if word.lower() in lexicon:
                            index_value = lexicon.index(word.lower())
                            # OR DO +=1, test both
                            features[index_value] += 1
                    line_x = list(features)
                    line_y = eval(label)
                    batch_x.append(line_x)
                    batch_y.append(line_y)
                    if len(batch_x) >= batch_size:
                        _, c = sess.run([optimizer, cost], feed_dict={x: np.array(batch_x),
                                                                  y: np.array(batch_y)})
                        epoch_loss += c
                        batch_x = []
                        batch_y = []
                        batches_run +=1
                        print('Batch run:',batches_run,'/',total_batches,'| Epoch:',epoch,'| Batch Loss:',c,)

            saver.save(sess, "model.ckpt")
            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
            with open(tf_log,'a') as f:
                f.write(str(epoch)+'\n') 
            epoch +=1

#train_neural_network(x)

def test_neural_network():
    prediction = neural_network_model(x)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            try:
                saver.restore(sess,"Kaggle/Sentiment140/model.ckpt")
            except Exception as e:
                print(str(e))

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        feature_sets, labels = create_test_data_pickle('Kaggle/Sentiment140/processed-test-set.csv')
        test_x = np.array(feature_sets)
        test_y = np.array(labels)
        print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))




def use_neural_network(input_data):
    prediction = neural_network_model(x)
    with open('Kaggle/Sentiment140/lexicon.pickle','rb') as f:
        lexicon = pickle.load(f)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,"Kaggle/Sentiment140/model.ckpt")
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
            subprocess.call(["say","Positive"])
            time.sleep(0.2)
            
            subprocess.call(["say",input_data])
            print('Positive:'+input_data )
            ##time.sleep(0.2)
        elif result[0] == 1:
            subprocess.call(["say","Negative"])
            time.sleep(0.2)
            subprocess.call(["say",input_data])
            print('Negative:'+input_data )

for i in range(2):
    use_neural_network("but they want to destroy my house")
    use_neural_network("have a fantastic Day")
    test_neural_network()
#use_neural_network("his is so nice! i love it! thanks!! thank you so much! ")

