import csv
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from gensim.models.keyedvectors import KeyedVectors
from sklearn.ensemble import RandomForestClassifier
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
print train 

x = train['similarity'].values.tolist()
x = np.reshape(np.array(x),(len(x),1))

y = train['source'].values.tolist()
y = np.reshape(np.array(y),(len(y),1))

x_test = test['similarity'].values.tolist()
x_test = np.reshape(np.array(x_test),(len(x_test),1))

y_test = test['source'].values.tolist()
y_test = np.reshape(np.array(y_test),(len(y_test),1))

#Gaussian NB
clf1 = GaussianNB()
clf1.fit(x,y)
pred1 = clf1.predict(x_test)
print (np.intersect1d(pred1,y_test)).size/(float(y_test.size+y.size))

#Random forest 
clf2 = RandomForestClassifier()
clf2.fit(x,y.ravel())
pred2 = clf2.predict(x_test)
print (np.intersect1d(pred2,y_test)).size/(float(y_test.size+y.size))

