import csv
import numpy as np
import string
import re
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet, stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import precision_score, accuracy_score, recall_score
from sklearn.decomposition import TruncatedSVD


#reading the data from the CSV file
def create_data(file):
    with open(file, encoding = 'utf8') as csv_file:
        reader = csv.reader(csv_file, delimiter = ",", quotechar = '"')
        title = []
        description = []
        leafnode = []
        for row in reader:
            title.append(row[0])
            description.append(row[1])
            leafnode.append(row[2])
        return title[1:], description[1:], leafnode[1:]

    
#removing stopwords, punctuations and special characters from the description
#lemmatizing the words
def clean_data(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    stops = stopwords.words('english')
    nonan = re.compile(r'[^a-zA-Z ]')
    output = []
    for i in range(len(text)):
        sentence = nonan.sub('', text[i])
        words = word_tokenize(sentence.lower())
        filtered_words = [w for w in words if not w.isdigit() and not w in stops and not w in string.punctuation]
        tags = pos_tag(filtered_words)
        cleaned = ''
        for word, tag in tags:
          if tag == 'NN' or tag == 'NNS' or tag == 'VBZ' or tag == 'JJ' or tag == 'RB' or tag == 'NNP' or tag == 'NNPS' or tag == 'RBR':
            cleaned = cleaned + wordnet_lemmatizer.lemmatize(word) + ' '
        output.append(cleaned.strip())
    return output


#feature extraction - creating a tf-idf matrix
def tfidf(data, ma = 0.6, mi = 0.0001):
    tfidf_vectorize = TfidfVectorizer(max_df = ma, min_df = mi)
    tfidf_data = tfidf_vectorize.fit_transform(data)
    return tfidf_data


#Naive Bayes classifier
def test_NaiveBayes(x_train, x_test, y_train, y_test):
    MNB = MultinomialNB()
    NBClassifier = MNB.fit(x_train, y_train)
    predictions = NBClassifier.predict(x_test)
	a = accuracy_score(y_test, predictions)
    p = precision_score(y_test, predictions, average = 'weighted')
    r = recall_score(y_test, predictions, average = 'weighted')
    return p, r


#SVM classifier
def test_SVM(x_train, x_test, y_train, y_test, label_names):
    SVM = SVC(kernel = 'linear')
    SVMClassifier = SVM.fit(x_train, y_train)
    predictions = SVMClassifier.predict(x_test)
	a = accuracy_score(y_test, predictions)
    p = precision_score(y_test, predictions, average = 'weighted')
    r = recall_score(y_test, predictions, average = 'weighted')
    return p, r


#Multilayer Perceptron classfier
def test_NN(x_train, x_test, y_train, y_test):
    NN = MLPClassifier(solver = 'lbfgs', alpha = 0.00095, learning_rate = 'adaptive', learning_rate_init = 0.005, max_iter = 300, random_state = 0)
    Perceptron = NN.fit(x_train, y_train)
    predictions = Perceptron.predict(x_test)
	a = accuracy_score(y_test, predictions)
    p = precision_score(y_test, predictions, average = 'weighted')
    r = recall_score(y_test, predictions, average = 'weighted')
    return p, r


#SGD classifier
#classifier that minimizes the specified loss using stochastic gradient descent
#hinge loss works pretty well too, the modified huber reports highest precision
def test_SGD(x_train, x_test, y_train, y_test):
    SGD = SGDClassifier(loss = 'modified_huber')
    SGDC = SGD.fit(x_train1, y_train)
    predictions = SGDC.predict(x_test1)
    a = accuracy_score(y_test, predictions)
    p = precision_score(y_test, predictions, average = 'weighted')
    r = recall_score(y_test, predictions, average = 'weighted')
    return p, r


#Voting Classifiers
#SVC and SGD combined with equal weights gave 1% lower precision than SVC
def test_voting(x_train, x_test, y_train, y_test):
    SVM = SVC(kernel = 'linear', probability = True)
    SGD = SGDClassifier(loss = 'modified_huber')
    EnsembleClassifier = VotingClassifier(estimators = [('sgd', SGD), ('svc', SVM)], voting = 'soft', weights = [1,1])
    EnsembleClassifier = EnsembleClassifier.fit(x_train, y_train)
    predictions = EnsembleClassifier.predict(x_test)
	a = accuracy_score(y_test, predictions)
    p = precision_score(y_test, predictions, average = 'weighted')
    r = recall_score(y_test, predictions, average = 'weighted')
    return p, r


	
	

#read data from file
title, desc, leaf = create_data(file)

#clean the data
title = clean_data(title)
desc = clean_data(desc)

#joining the title and description to create a single text document for each product
combined = desc[:]
for i in range(len(combined)):
    combined[i] = title[i] + ' ' + desc[i]

#feature extraction
training = tfidf(combined)

#training and test data splits
x_train, x_test, y_train, y_test = cross_validation.train_test_split(training, leaf, test_size = 0.25, random_state = 0)

#test a classifier
accuracy, precision, recall = test_SVM(x_train, x_test, y_train, y_test)
