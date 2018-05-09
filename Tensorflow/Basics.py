
import tensorflow as tf


'''
Build The data 
'''
x1 = tf.constant(5)
x2 = tf.constant(6)
reuslt = tf.multiply(x1,x2)
print(reuslt)

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])
product = tf.matmul(matrix1, matrix2) 
'''
Run The data in Session
'''
with tf.Session() as sess:
    output = sess.run(reuslt)
    print(output)
    # multiply two matrix
    output = sess.run(product)
    print(output)