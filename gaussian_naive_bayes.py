import sys
import numpy as np

CLASS_LABEL_FIELD = 9
SEPARATOR = ','
NONSPAM = 0
SPAM = 1

def estimate_priors(class_counts):
    return class_counts/sum(class_counts)

def get_parameters(examples):
    parameters = []
    for attribute in examples.T:
        mean = np.mean(attribute)
        var = np.sum([(i - mean) ** 2 for i in attribute])/(len(attribute) - 1)
        parameters.append((mean, var))
    return parameters

def learn_model(filename):
    class_counts = np.zeros(2)
    examples_non_spam = []
    examples_spam = []
    with open(filename, 'r') as f:
        for line in f:
            fields = line.strip().split(SEPARATOR)
            if int(fields[CLASS_LABEL_FIELD]) == 0:
                examples_non_spam.append(np.array([float(i) for i in fields[:-1]]))
            else:
                examples_spam.append(np.array([float(i) for i in fields[:-1]]))
            class_counts[int(fields[CLASS_LABEL_FIELD])] += 1.0

    class_priors = estimate_priors(class_counts)
    print 'Estimated value of P(C = 0) is', class_priors[0]
    print 'Estimated value of P(C = 1) is', class_priors[1]
    print '============================='

    examples_non_spam = np.array(examples_non_spam)
    examples_spam = np.array(examples_spam)
    non_spam_params = get_parameters(examples_non_spam)
    for i, pair in enumerate(non_spam_params):
        print 'Estimated parameters for attribute x_%d and class C = 0: mean = %lf and variance = %lf' % (i+1, pair[0], pair[1])
    spam_params = get_parameters(examples_spam)
    for i, pair in enumerate(spam_params):
        print 'Estimated parameters for attribute x_%d and class C = 1: mean = %lf and variance = %lf' % (i+1, pair[0], pair[1])
    print '============================='

    return class_priors, non_spam_params, spam_params 

def compute_pdf(value, parameters):
    mean = parameters[0]
    var = parameters[1]
    return 1/(np.sqrt(2 * np.pi * var)) * np.exp(-(value - mean)**2 / (2 * var))

def classify_example(example, priors, non_spam_parameters, spam_parameters):
    #Calculating a posteriori hypothesis of example being non-spam
    estimate_non_spam = np.log(priors[NONSPAM])
    for i, value in enumerate(example):
        print value, non_spam_parameters[i], compute_pdf(value, non_spam_parameters[i])
        estimate_non_spam += np.log(compute_pdf(value, non_spam_parameters[i]))

    #Calculating a posteriori hypothesis of example being spam
    estimate_spam = np.log(priors[SPAM])
    for i, value in enumerate(example):
        estimate_spam += np.log(compute_pdf(value, spam_parameters[i]))
    if estimate_non_spam > estimate_spam:
        return '0'
    else:
        return '1'

def test_time(priors, non_spam_parameters, spam_parameters, filename):
    with open(filename, 'r') as f:
        correctly_classified = 0
        incorrectly_classified = 0
        for index, line in enumerate(f):
            fields = line.strip().split(SEPARATOR)
            true_label = fields[CLASS_LABEL_FIELD]
            example = [float(i) for i in fields[:-1]]
            predicted_label = classify_example(example, priors, non_spam_parameters, spam_parameters) 
            print 'Predicted class for example %d: %s' % (index+1, predicted_label)
            if true_label == predicted_label:
                correctly_classified += 1
            else:
                incorrectly_classified += 1
        print '============================='
        print 'Total number of test examples classified correctly: %d' % correctly_classified
        print '============================='
        print 'Total number of test examples classified incorrectly: %d' % incorrectly_classified
        print '============================='
        fraction = (1. * incorrectly_classified)/(incorrectly_classified + correctly_classified)
        print 'Percentage error on the test examples: %lf' % fraction 

if __name__ == '__main__':
    
    training_filename = sys.argv[1]
    test_filename = sys.argv[2]
    priors, non_spam_parameters, spam_parameters = learn_model(training_filename)
    test_time(priors, non_spam_parameters, spam_parameters, test_filename)
