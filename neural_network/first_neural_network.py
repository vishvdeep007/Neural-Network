import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#path for the graph
logs_path = "tensorflow_graph"

#number of neurons of each layer
n_node_hl1 = 500
n_node_hl2 = 500
n_node_hl3 = 500

n_classes = 10 # numbers of digit
batch_size = 100 

#input for Calculation graph
x = tf.placeholder('float', [None, 784]) #shaping input data 
y = tf.placeholder('float')

def neural_network_model(data):

    #Dictionary for the layers
    hidden_1_layer = {'weight':tf.Variable(tf.random_normal([784, n_node_hl1])),
                    'baises':tf.Variable(tf.random_normal([n_node_hl1]))}

    hidden_2_layer = {'weight':tf.Variable(tf.random_normal(n_node_hl1, n_node_hl2)),
                    'baises':tf.Variable(tf.random_normal([n_node_hl2]))}

    hidden_3_layer = {'weight':tf.Variable(tf.random_normal(n_node_hl2, n_node_hl3)),
                    'baises':tf.Variable(tf.random_normal([n_node_hl3]))}

    output_layer = {'weight':tf.Variable(tf.random_normal(n_node_hl3, n_classes)),
                    'baises':tf.Variable(tf.random_normal([n_classes]))}

    #(input_Data * weight) + biases

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weight']), tf.hidden_1_layer['baises'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weight']), tf.hidden_2_layer['baises'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weight']), tf.hidden_3_layer['baises'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weight']) + output_layer['baises']

    return output

def train_neural_network(x):

    #prediction of the model
    prediction  = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))

    #learning rate = 0.01
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    #cycle feed forward + backprop = epoch
    hm_epoch = 50 

    with tf.Sesson() as sess:
        sess.run(tf.global_variable_initializer())
        #training model with training data
        for epoch  in range(hm_epoch):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, x - sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', hm_epoch, 'loss:', epoch_loss) 
        writer = tf.summary.FileWriter(logs_path,graph=tf.get_default_graph())  

        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.text.labels}))


train_neural_network(x)