import csv
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from gensim.models.keyedvectors import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer

#load dataset
def load_csv():
    lines = csv.reader(open('wordgame.csv', 'rb'))
    dataset = list(lines)
    
    return dataset
    
#feature extraction
def make_features(dataset):
    x1 = []
    x2 = []
    y = []
    
    for i in range (1,len(dataset)):
        #word1
        x1.append(str(dataset[i][1]).lower())
        #word2
        x2.append(str(dataset[i][2]).lower())
        #source
        y.append(str(dataset[i][3]).lower())
    
    w2v_model = KeyedVectors.load_word2vec_format('../../data/external/GoogleNews-vectors-negative300.bin', binary=True)

    #train on 90% and test on 10%
    split = int(len(x1)*.90)
    
    x1_train = x1[0:split]
    x1_test =  x1[split:len(x1)]
    
    x2_train = x2[0:split]
    x2_test = x2[split:len(x2)]
    
    y_train = y[0:split]
    y_test = y[split:len(y)]
    
    #X_train = list(zip(x1_train,x2_train))
    #X_train = np.array(X_train)
    #y_train = np.array(y_train)
    
    #count_vect =  CountVectorizer()
    #x1_train_counts = count_vect.fit_transform(x1_train)
    #print x1_train_counts.shape
    
    #use a Naive Bayes Classifier
    #clf = MultinomialNB()
    #clf.fit(X_train, y_train)
  
    
    #see how well we are doing
    #X_test = []
    #X_test = list(zip(x1_test,x2_test))
    #X_test = np.array(X_test)

    #print clf.score(x1_test,x2_test,y_test)
    
def main():
    make_features(load_csv())
main()