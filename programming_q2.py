'''
This program receives the following arguments:

argv[1] => training file (reviewstrain.txt)
argv[2] => test file (reviewstest.txt)
'''

import sys
import numpy as np
from scipy.sparse import csr_matrix

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
        return csr_matrix((data, indices, indptr), dtype=int).toarray(), classes 

if __name__ == '__main__':

    training_filename = sys.argv[1]
    training_data, training_classes = get_training_data(training_filename)
    print training_data[0, 4]
