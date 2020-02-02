import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from interpret import calc_precision, calc_f1_score, calc_recall, avg_accuracy, calc_confusion_matrix
#Transform the labels into binary matrices
# ex: [[0], [3]] => [[1,0,0,0,0], [0,0,0,1,0]]
def onehot(labels, n_classes=5):
    y_mat = np.zeros((len(labels), n_classes))
    for idx, v in enumerate(labels):
        y_mat[idx, int(v)] = 1
    y_mat = np.float32(y_mat)
    return y_mat

#Create model, store the calculated layers as dict{layer# : tensor}
#activation is a list, specifying the activation function for each layer
#layers, weights and biases are 0 indexed
#basically this happens: layer[i] = activation[i](layer[i-1] x weight[i] + bias[i])
#the first layer and the out layer are calculated out of the loop
def mlp(data, list_hidden_nodes, weights, biases, activation):
    modeldict = {}
    modeldict['layer0'] = activation[0](tf.add(tf.matmul(data, weights['h0']), biases['b0']))
    for i in range(1, len(list_hidden_nodes)):
        modeldict['layer' + str(i)] = activation[i](tf.add(tf.matmul(\
            modeldict['layer'+str(i-1)], weights['h'+str(i)]), biases['b'+str(i)]))
    modeldict['out_layer'] = tf.matmul(modeldict['layer' + str(len(list_hidden_nodes)-1)],\
                             weights['out']) + biases['out']
    return modeldict['out_layer']

#so this is where magic happens
def neural_net(data, labels, list_hidden_nodes, activation, eval_data, eval_labels, \
            learning_rate=0.001, batch_size=1000, training_epochs=10000,\
             display_step=100, n_classes=5, opt=tf.train.AdamOptimizer):
                
    X = tf.placeholder(dtype=tf.float32, shape=[None, len(data.T)])
    Y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes])
    
    #create weight and bias dictionary with initially random values
    #drawn from standard normal distribution
    weights = {'h0': tf.Variable(tf.random_normal([len(data.T), list_hidden_nodes[0]]), dtype=tf.float32),
        'out': tf.Variable(tf.random_normal([list_hidden_nodes[-1], n_classes]), dtype=tf.float32)}
    biases = {'b0': tf.Variable(tf.random_normal([list_hidden_nodes[0]]), dtype=tf.float32),
            'out': tf.Variable(tf.random_normal([n_classes]), dtype=tf.float32)}
    #calulate all the weight and biases, apart from first and out
    #weight[i].shape := (# hidden nodes on layer[i-1], # hidden nodes on layer[i])
    for i in range(1, len(list_hidden_nodes)):
        weights['h' + str(i)] = tf.Variable(tf.random_normal(\
                [list_hidden_nodes[i-1], list_hidden_nodes[i]]), dtype=tf.float32)
        biases['b' + str(i)] = tf.Variable(tf.random_normal(\
                [list_hidden_nodes[i]]), dtype=tf.float32)

    # Construct model
    logits = mlp(X, list_hidden_nodes, weights, biases, activation)
    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    if opt is tf.train.MomentumOptimizer :
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.01)
    else:
        optimizer = opt(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        #Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(len(data)/batch_size)
            #Loop over all batches
            for i in range(total_batch):
                sampleidxs = np.random.choice(len(data), size=batch_size, replace=True)
                batch_x, batch_y = data[sampleidxs], labels[sampleidxs]

                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})

                #compute average loss
                avg_cost += c / total_batch
            #Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
        print("Optimization finished")

        pred = tf.nn.softmax(logits)
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
        #Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Train Accuracy:", accuracy.eval({X: data, Y: labels}))
        print("Validation Accuracy:", accuracy.eval({X: eval_data, Y: eval_labels}))
        #return predictions for train set and test set
        return pred.eval({X: data,Y: labels}), pred.eval({X: eval_data,Y: eval_labels})
"""
#sample use
data = np.load("fashion_train.npy")
dense_labels = data[:,-1]
onehot_labels = onehot(dense_labels)
data = data[:,:-1]
testdata = np.load("fashion_test.npy")
dense_testlabels= testdata[:,-1]
onehot_testlabels = onehot(dense_testlabels)
testdata = testdata[:,:-1]

hnlist = [50, 50, 50, 50]
actlist = [tf.nn.sigmoid] * len(hnlist)

train_out, test_out = neural_net(data, onehot_labels, hnlist, actlist, testdata, onehot_testlabels, training_epochs=501)
confm_train = calc_confusion_matrix(np.intc(dense_labels), np.intc(np.argmax(train_out, axis=1)))
confm_test = calc_confusion_matrix(np.intc(dense_testlabels), np.intc(np.argmax(test_out, axis=1)))
confm_train = np.intc(confm_train)
confm_test = np.intc(confm_test)
#print scores
allscores(confm_train)
allscores(confm_test)
#plot heatmaps of confusion matrix
ax = sns.heatmap(confm_train,  annot=True, fmt="d", cbar=True, square=True)
plt.show()
ax = sns.heatmap(confm_test,  annot=True, fmt="d", cbar=True, square=True)
plt.show()
"""