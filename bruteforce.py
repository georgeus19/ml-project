import numpy as np
import tensorflow as tf
from mlp import neural_net, onehot, mlp
from interpret import calc_confusion_matrix
import pickle
data = np.load("data_0-1_norm_with_labels.npy")
data = np.float32(data)
oglabels = data[:,-1]
data = data[:,:-1]
labels = onehot(oglabels)
ldadata = np.load("LDA_4d_out.npy")
layers = [1, 2, 3, 4]
nodes = [5, 10, 15, 20, 50]
acts = [tf.nn.sigmoid]
k=10
confdict = dict()
for d, name in [(data, "full"), (ldadata, "lda_reduced")]:
    for layernum in layers:
        for nodenum in nodes:
            hnlist = [nodenum] * layernum
            actlist = [tf.nn.sigmoid] * layernum
            confmat_train = []
            confmat_eval = []
            for i in range(k):
                #create copies of the train and test data
                datacopy = data.copy()
                labelscopy = labels.copy()
                #make validation split
                eval_data = datacopy[int(i*len(datacopy)/k):int((i+1)*len(datacopy)/k),:]
                eval_labels = labelscopy[int(i*len(datacopy)/k):int((i+1)*len(datacopy)/k),:]
                dense_eval_labels = np.argmax(labels_validation, axis=1)
                #make train split
                train_data = np.delete(datacopy, list(range(int(i*len(datacopy)/k),int((i+1)*len(datacopy)/k))), axis=0)
                train_labels = np.delete(labelscopy, list(range(int(i*len(datacopy)/k),int((i+1)*len(datacopy)/k))), axis=0)
                dense_train_labels = np.argmax(train_labels, axis=1)
                #get predictions
                train_confm, eval_confm = neural_net(train_data, train_labels, hnlist, actlist, eval_data, eval_labels, training_epochs=501)
                #get confusion matrices for the current fold
                confmat_train.append(calc_confusion_matrix(np.intc(dense_train_labels), np.intc(np.argmax(train_confm, axis=1))))
                confmat_eval.append(calc_confusion_matrix(np.intc(dense_eval_labels), np.intc(np.argmax(eval_confm, axis=1))))
            
            confmat[(layernum, nodenum, name, "train")] = confmat_train
            confmat[(layernum, nodenum, name, "eval")] = confmat_eval
with open('gridsearch.pickle', 'wb') as f:
    #use pickle.loads() to read it
    pickle.dump(confdict, f)
