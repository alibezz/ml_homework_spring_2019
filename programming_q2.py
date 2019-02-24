'''
This program receives the following arguments:

argv[1] => training file (reviewstrain.txt)
argv[2] => test file (reviewstest.txt)
argv[3] => k
'''

import sys
import numpy as np
from scipy.sparse import csr_matrix
import operator

SEPARATOR = ' '

def get_training_data(filename):
    vocab = {}
    data = []
    indices = []
    indptr = [0]
    classes = []
    with open(filename, 'r') as training_file:
        for line in training_file:
            fields = line.strip().split(SEPARATOR)
            classes.append(fields[0])
            document = fields[1:]
            for term in document:
                index = vocab.setdefault(term, len(vocab))
                indices.append(index)
                data.append(1)
            indptr.append(len(indices))  
        return csr_matrix((data, indices, indptr), dtype=int), classes, vocab 

def indices_test_document(document, vocabulary):
    document_indexes = list(set([vocabulary[term] for term in document if vocabulary.has_key(term)]))
    return np.array(document_indexes)
    
def inverse_intersection_distance(test_indices, train_indices):
    intersection = set(test_indices).intersection(train_indices)
    try:
        return 1./len(intersection)
    except ZeroDivisionError:
        return np.inf

def  predict_common_among_closest(distances, training_classes, k):
    initial_distances = distances[0:k]
    additional_distances = []
    i = 0
    while True:
        if (k + i) < len(distances) and distances[k + i][1] == initial_distances[-1][1]:
            additional_distances.append(distances[k + i])
            i += 1
        else:
            break
    relevant_distances = initial_distances + additional_distances
    classes = [training_classes[i[0]] for i in relevant_distances]
    #print relevant_distances, classes
    count_0 = len([i for i in classes if i == '0'])
    count_1 = len([i for i in classes if i == '1'])
    if count_0 == count_1:
        return '1'
    else:
        return max(set(classes), key=classes.count)

def compute_data_for_confusion_matrix(true_, predicted):
    tp = fp = tn = fn = 0
    #print 't', true_
    #print 'p', predicted
    for t, p in zip(true_, predicted):
        if t == p:
            if t == '1':
                tp += 1
            else:
                tn += 1
        else:
            if t == '1':
                fn += 1
            else:
                fp += 1
    return tp, fp, tn, fn

def test_model(filename, training_data, training_classes, vocab, k):
    with open(filename, 'r') as test_file:
        true_labels = []
        predicted_labels = []
        #zero_r_prediction = max(set(training_classes), key=training_classes.count)
        #print 'zero-r', zero_r_prediction
        for line in test_file:
            fields = line.strip().split(SEPARATOR)
            true_labels.append(fields[0])
            doc_indices = indices_test_document(fields[1:], vocab) 
            distances = []
            for index, train_doc in enumerate(training_data):
                train_doc_indices = train_doc.nonzero()[1]
                dist = inverse_intersection_distance(doc_indices, train_doc_indices)
                distances.append((index, dist))
            distances.sort(key=operator.itemgetter(1))
            predicted_labels.append(predict_common_among_closest(distances, training_classes, k))
            #predicted_labels.append(zero_r_prediction)
        tp, fp, tn, fn = compute_data_for_confusion_matrix(true_labels, predicted_labels)
        print 'tp', tp, 'fp', fp, 'tn', tn, 'fn', fn    

def cross_validation(filename, k):
    training_data, training_classes, vocab = get_training_data(filename)
    #slice the data in 5 parts
    print training_data[1:5].todense()#.todense()#[1:5]

if __name__ == '__main__':

    #Not normalizing the attributes because I believe it's ok to assume that the frequency of the tokens are on the same scale
    training_filename = sys.argv[1]
    test_filename = sys.argv[2]
    k = int(sys.argv[3])
    cross_validation(training_filename, k)
    #training_data, training_classes, vocab = get_training_data(training_filename)
    #test_model(test_filename, training_data, training_classes, vocab, k)
