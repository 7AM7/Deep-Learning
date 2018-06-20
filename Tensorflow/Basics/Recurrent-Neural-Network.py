import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


'''
How to Neural Network work:
> = then
First Step:

input > weight > hidden layer 1 (activation function) > 
weights > hidden Layer 2 (activation function) >
weights > output layer

Second Step:

Compare output to intedned output(the output we want or we know) > 
cost function (cross entropy)
optimization function (optimizer) > minimize cost (AdamOptimizer ... SGD , AdaGrad)

Third Step:

backpropagation
feed forward + backpop = epoch

'''

## In This Example I will check the number  between 0 to 9 from image 28 x 28 paixels
## So we Have 10 classes , 0 - 9
'''
0 = [1,0,0,0,0,0,0,0,0,0]       
1 = [0,1,0,0,0,0,0,0,0,0]      
2 = [0,0,1,0,0,0,0,0,0,0]       
3 = [0,0,0,1,0,0,0,0,0,0]       
4 = [0,0,0,0,1,0,0,0,0,0]       
5 = [0,0,0,0,0,1,0,0,0,0]       
6 = [0,0,0,0,0,0,1,0,0,0]
7 = [0,0,0,0,0,0,0,1,0,0]
8 = [0,0,0,0,0,0,0,0,1,0]
9 = [0,0,0,0,0,0,0,0,0,1]


So the Right number will be ON When ON = 1  
Not Right number will be OFF when OFF = 0
'''

mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

hm_epochs = 10
n_classes = 10
batch_size = 128
chunk_size = 28
n_chunks = 28
rnn_size = 128


# height x width 
# 28 x 28 = 784
x = tf.placeholder('float',[None,n_chunks,chunk_size])
y = tf.placeholder('float')

# First Step:
def recurrent_neural_network(x):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
             'biases':tf.Variable(tf.random_normal([n_classes]))}
    '''
    [[
        [ 1.  1.  1.], ## ones
        [ 1.  1.  1.]
    ]]

    [
        [[ 1.  1.  1.]],                #transpose 
        [[ 1.  1.  1.]]
    ]
    '''
    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(rnn_size,state_is_tuple=True)
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    ## last output * layer weights + biases
    output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']

    return output

# Second Step , Third Step:
def train_neural_network(x):
    prediction = recurrent_neural_network(x) 
    print(prediction)
    ## cost function
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    ## optimize cost
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    ## Third Step:

    # cycles feed forward + backpop
    epochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                epoch_x = epoch_x.reshape((batch_size,n_chunks,chunk_size))

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch+1, 'completed out of',epochs,'loss:',epoch_loss)

        # compare prediction with label
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images.reshape((-1, n_chunks, chunk_size)), y:mnist.test.labels}))


train_neural_network(x)