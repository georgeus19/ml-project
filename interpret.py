import numpy as np
import pickle

def calc_precision(confusion_mat):
    true_positive_dict = {}
    false_positive_dict = {}
    for i in range(confusion_mat.shape[1]):
        true_positive_dict[i] = 0
        false_positive_dict[i] = 0
        for j in range(confusion_mat.shape[0]):
            if i == j :
                true_positive_dict[i] += confusion_mat[i,j]
            else:
                false_positive_dict[i] += confusion_mat[i,j]
    precision_dict = {}
    for i in range(confusion_mat.shape[1]):
        precision_dict[i] = true_positive_dict[i]/( true_positive_dict[i] + false_positive_dict[i])  
    return precision_dict

def calc_recall(confusion_mat):
    true_positive_dict = {}
    false_negative_dict = {}
    for i in range(confusion_mat.shape[0]):
        true_positive_dict[i] = 0
        false_negative_dict[i] = 0
        for j in range(confusion_mat.shape[1]):
            if i == j :
                true_positive_dict[i] += confusion_mat[j,i] 
            else:
                false_negative_dict[i] += confusion_mat[j,i]
    recall_dict = {}
    for i in range(confusion_mat.shape[0]):
        recall_dict[i] = true_positive_dict[i]/(true_positive_dict[i] + false_negative_dict[i])
    return recall_dict


def calc_f1_score(prec_dict , recall_dict):
    f1_dict = {}
    for i in range(len(prec_dict)):
        f1_dict[i] = (2 * prec_dict[i] * recall_dict[i] ) / (prec_dict[i] + recall_dict[i])
    return f1_dict

def avg_accuracy(confmat):
    accuracy = []
    for clss in range(5):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for i in range(confmat.shape[0]):
            for j in range(confmat.shape[1]):
                if i == clss and j == clss:
                    tp = confmat[i,j]
                elif i == clss:
                    fp += confmat[i,j]
                elif j == clss:
                    fn += confmat[i,j]
                else:
                    tn += confmat[i,j]
        print(tp, tn, fn, fp)
        accuracy.append((tp + tn) / (tp + tn + fp + fn))
    return(sum(accuracy)/confmat.shape[0])
        
def calc_confusion_matrix(true, pred):
    #K = len(np.unique(true)) # Number of classes 
    K = 5
    result = np.zeros((K, K))
    for i in range(len(true)):
        result[true[i]][pred[i]] += 1
    return result

def allscores(confmat):
    precision = calc_precision(confmat)
    recall = calc_recall(confmat)
    avg_acc = avg_accuracy(confmat)
    f1 = calc_f1_score(precision, recall)
    print("acc:", avg_acc, "rec:" , sum(recall.values())/5, "prec:", sum(precision.values())/5, "f1:", sum(f1.values())/5)


"""
infile = open("gridsearch.pickle", 'rb')
confdict = pickle.load(infile)
allconfmats = {}
for k, v in confdict.items():
    temp = np.zeros((5,5))
    for i in v:
        temp += i
    allconfmats[k] = temp
precision_scores = {}
recall_scores = {}
for k, v in allconfmats.items():
    precision_scores[k] = calc_precision(v)
    recall_scores[k] = calc_recall(v)
f1 = {}
for k, v in precision_scores.items():
    for k2, v2 in recall_scores.items():
        if k == k2:
            f1[k] = calc_f1_score(v, v2)
x = []
for k,v in f1.items():
    t = 0
    print(k)
    for v2 in v.values():
        t += v2
    print(t/5)
    x.append(t/5)
"""