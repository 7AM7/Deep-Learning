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

n_nodes_hl1 = 500   # number of nodes in hidden layer 1
n_nodes_hl2 = 500   # number of nodes in hidden layer 2
n_nodes_hl3 = 500   # number of nodes in hidden layer 3

n_classes = 10      # number of classes we have 0 - 9
batch_size = 100    # feed the NN 100 features

# height x width 
# 28 x 28 = 784
x = tf.placeholder('float',[None,784])
y = tf.placeholder('float')

# First Step:
def neural_network_model(data):
    '''
        The bias is a value that is added to our sums, before being passed through the activation function, n
        ot to be confused with a bias node, which is just a node that is always on. 
        The purpose of the bias here is mainly to handle for scenarios where all neurons fired a 0 into the layer. 
        A bias makes it possible that a neuron still fires out of that layer. A bias is as unique as the weights, 
        and will need to be optimized too.
    '''
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes]))}

    # (input data * weights) + biases

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    # activation function
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    # activation function
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    # activation function
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output

# Second Step , Third Step:
def train_neural_network(x):
    prediction = neural_network_model(x) 
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
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch+1, 'completed out of',epochs,'loss:',epoch_loss)

        # compare prediction with label
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


train_neural_network(x)