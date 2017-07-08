import csv
import numpy as np
import pandas as pd
#from sklearn.pipeline import Pipeline
#from sklearn.naive_bayes import MultinomialNB
from gensim.models.keyedvectors import KeyedVectors

#load dataset
lines = csv.reader(open('wordgame.csv', 'rb'))
dataset = list(lines)
    
#feature extraction
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

#make dataframe to hold the values  
data = pd.DataFrame(
   {'word1':x1,
    'word2':x2, 
     'source':y,
   })
      
#word2vec model
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

#redefine similarity for the word2vec model
def similarity(row):
    if row.word1 in model.vocab and row.word2 in model.vocab:
        return model.similarity(row.word1, row.word2)
    else:
        return 100

#dataframe with similarity values 
data['similarity'] = data.apply(similarity, axis=1)

#train on 80% and test on 20%
split = np.random.rand(len(data))<0.8
train = data[split] 
test = data[~split]

#build vectors 
def buil_vectors():
    return v
#pass through the pipeline


#x1_train = x1[0:split]
#x1_test =  x1[split:len(x1)]
#x2_train = x2[0:split]
#x2_test = x2[split:len(x2)]
    
#y_train = y[0:split]
#y_test = y[split:len(y)]
    
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
