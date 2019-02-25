'''
This program receives the following arguments:

argv[1] => training file (reviewstrain.txt)
argv[2] => test file (reviewstest.txt)
argv[3] => k
argv[4] => distance function: 'standard' is for the one in the question; 'custom', for the one we implemented
'''

import sys
import numpy as np
from scipy.sparse import csr_matrix, vstack
from scipy.spatial.distance import cosine
from scipy.stats import itemfreq
import operator

SEPARATOR = ' '
STOPWORDS_PUNCTUATION = ["i","me","my","myself","we","our","ours","ourselves","you","your","yours","yourself","yourselves","he","him","his","himself","she","her","hers","herself","it","its","itself","they","them","their","theirs","themselves","what","which","who","whom","this","that","these","those","am","is","are","was","were","be","been","being","have","has","had","having","do","does","did","doing","a","an","the","and","but","if","or","because","as","until","while","of","at","by","for","with","about","against","between","into","through","during","before","after","above","below","to","from","up","down","in","out","on","off","over","under","again","further","then","once","here","there","when","where","why","how","all", ".", ",", "?", "!", ";", "(", ")"]

def preprocess(terms):
    return list(filter(lambda a: a not in STOPWORDS_PUNCTUATION, terms))
    
def get_training_data(filename, dist_function):
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
            if dist_function == 'custom':
                document = preprocess(fields[1:])
            for term in document:
                index = vocab.setdefault(term, len(vocab))
                indices.append(index)
                data.append(1)
            indptr.append(len(indices))  
        return csr_matrix((data, indices, indptr), dtype=int), classes, vocab 

def indices_test_document(document, vocabulary):
    return [vocabulary[term] for term in document if vocabulary.has_key(term)]
        
def inverse_intersection_distance(test_indices, train_indices):
    intersection = set(test_indices).intersection(set(train_indices))
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

def test_example(training_data, document, training_classes, k, dist_function):
    distances = []
    doc_indices = indices_test_document(document, vocab) 
    for index, train_doc in enumerate(training_data):
        train_doc_indices = train_doc.nonzero()[1]
        if dist_function == 'standard':
            dist = inverse_intersection_distance(doc_indices, train_doc_indices)
        else:
            union_of_indices = list(set(np.concatenate((doc_indices, train_doc_indices))))
            freq_doc = dict(itemfreq(doc_indices))
            vector_doc = [freq_doc[i] if freq_doc.has_key(i) else 0 for i in union_of_indices]
            freq_train = dict(itemfreq(train_doc_indices))
            vector_train = [freq_train[i] if freq_train.has_key(i) else 0 for i in union_of_indices]
            #print freq_doc, vector_doc
            #print freq_train, vector_train
            dist = cosine(vector_doc, vector_train)
        distances.append((index, dist))
    distances.sort(key=operator.itemgetter(1))
    return predict_common_among_closest(distances, training_classes, k) 

def test_model(filename, training_data, training_classes, vocab, k, dist_function):
    with open(filename, 'r') as test_file:
        true_labels = []
        predicted_labels = []
        #zero_r_prediction = max(set(training_classes), key=training_classes.count)
        #print 'zero-r', zero_r_prediction
        for line in test_file:
            fields = line.strip().split(SEPARATOR)
            true_labels.append(fields[0])
            doc = fields[1:]
            label = test_example(training_data, doc, training_classes, k, dist_function)
            predicted_labels.append(label)
            #predicted_labels.append(zero_r_prediction)
        tp, fp, tn, fn = compute_data_for_confusion_matrix(true_labels, predicted_labels)
        print 'tp', tp, 'fp', fp, 'tn', tn, 'fn', fn    

def run_slice(training, training_classes, test, test_classes, k):
    predicted_labels = []
    for example in test:
        indices = example.nonzero()[1]
        label = test_example(training, indices, training_classes, k)
        predicted_labels.append(label)
    return compute_data_for_confusion_matrix(test_classes, predicted_labels)

def cross_validation(filename, k, size):
    training_data, training_classes, vocab = get_training_data(filename)
    #slice the data in 5 parts
    data0 = training_data[0:size]; classes0 = training_classes[0:size]
    data1 = training_data[size:size*2]; classes1 = training_classes[size:size*2]
    data2 = training_data[size*2:size*3]; classes2 = training_classes[size*2:size*3]
    data3 = training_data[size*3:size*4]; classes3 = training_classes[size*3:size*4]
    data4 = training_data[size*4:size*5]; classes4 = training_classes[size*4:size*5]

    training0 = vstack((data1, data2, data3, data4)); training_classes0 = classes1 + classes2 + classes3 + classes4
    tp0, fp0, tn0, fn0 = run_slice(training0, training_classes0, data0, classes0, k)

    training1 = vstack((data0, data2, data3, data4)); training_classes1 = classes0 + classes2 + classes3 + classes4
    tp1, fp1, tn1, fn1 = run_slice(training1, training_classes1, data1, classes1, k)

    training2 = vstack((data0, data1, data3, data4)); training_classes2 = classes0 + classes1 + classes3 + classes4
    tp2, fp2, tn2, fn2 = run_slice(training2, training_classes2, data2, classes2, k)
    
    training3 = vstack((data0, data1, data2, data4)); training_classes3 = classes0 + classes1 + classes2 + classes4
    tp3, fp3, tn3, fn3 = run_slice(training3, training_classes3, data3, classes3, k)
    
    training4 = vstack((data0, data1, data2, data3)); training_classes4 = classes0 + classes1 + classes2 + classes3
    tp4, fp4, tn4, fn4 = run_slice(training4, training_classes4, data4, classes4, k)
    
    acc = (tp0 + tn0 + tp1 + tn1 + tp2 + tn2 + tp3 + tn3 + tp4 + tn4)/1500.
    print 'cross-validation accuracy', acc, 'num', tp0 + tn0 + tp1 + tn1 + tp2 + tn2 + tp3 + tn3 + tp4 + tn4

if __name__ == '__main__':

    #Not normalizing the attributes because I believe it's ok to assume that the frequency of the tokens are on the same scale
    training_filename = sys.argv[1]
    test_filename = sys.argv[2]
    k = int(sys.argv[3])
    dist_function = sys.argv[4]
    #cross_validation(training_filename, k, 300)
    training_data, training_classes, vocab = get_training_data(training_filename, dist_function)
    test_model(test_filename, training_data, training_classes, vocab, k, dist_function)
