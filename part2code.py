import numpy as np
import nltk
import sklearn
import operator
import requests
nltk.download('stopwords') # If needed
nltk.download('punkt') # If needed
nltk.download('wordnet') # If needed

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.externals import joblib

import re
from bs4 import BeautifulSoup

def review_to_wordlist(review):
    '''
    Turn IMDB's comments into word sequences
    '''
    # Remove the HTML tag and get the content
    review_text = BeautifulSoup(review, "html.parser").get_text()
    # Use regular expressions to extract portions that conform to the specification
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    # Lowercase all words and turn them into a list of words
    words = review_text.lower().split()
    # return words
    return words

from google.colab import drive
drive.mount('/content/drive/')

'''
path_pos= '/content/drive/My Drive/imdb_dev_pos.txt'
dev_set_pos=open(path_pos).readlines()

path_neg= '/content/drive/My Drive/imdb_dev_neg.txt'
dev_set_neg=open(path_neg).readlines()

path_pos= '/content/drive/My Drive/imdb_train_pos.txt'
train_set_pos=open(path_pos).readlines()

path_neg= '/content/drive/My Drive/imdb_train_neg.txt'
train_set_neg=open(path_neg).readlines()

path_pos= '/content/drive/My Drive/imdb_test_pos.txt'
test_set_pos=open(path_pos).readlines()

path_neg= '/content/drive/My Drive/imdb_test_neg.txt'
test_set_neg=open(path_neg).readlines()
'''

#load local file
#you need add address behind ../Src which are the local places

path_pos= '../Src/IMDb/dev/imdb_dev_pos.txt'
dev_set_pos=open(path_pos).readlines()

path_neg= '../Src/IMDb/dev/imdb_dev_neg.txt'
dev_set_neg=open(path_neg).readlines()

path_pos= '../Src/IMDb/train/imdb_train_pos.txt'
train_set_pos=open(path_pos).readlines()

path_neg= '../Src/IMDb/train/imdb_train_neg.txt'
train_set_neg=open(path_neg).readlines()

path_pos= '../Src/IMDb/test/imdb_test_pos.txt'
test_set_pos=open(path_pos).readlines()

path_neg= '../Src/IMDb/test/imdb_test_neg.txt'
test_set_neg=open(path_neg).readlines()

dev_set=[]
for pos_review in dev_set_pos:
  dev_set.append((pos_review,1))
for neg_review in dev_set_neg:
  dev_set.append((neg_review,0))

train_set=[]
for pos_review in train_set_pos:
  train_set.append((pos_review,1))
for neg_review in train_set_neg:
  train_set.append((neg_review,0))

test_set=[]
for pos_review in test_set_pos:
  test_set.append((pos_review,1))
for neg_review in test_set_neg:
  test_set.append((neg_review,0))

test_x = []
for sentiment in train_set_pos:
  test_x.append(1)
for sentiment in train_set_neg:
  test_x.append(0)

test_y = []
for sentiment in test_set_pos:
  test_y.append(1)
for sentiment in test_set_neg:
  test_y.append(0)

dev_file=[]
for i in range(len(dev_set_pos)):
  dev_file.append(' '.join(review_to_wordlist(dev_set_pos[i])))
for i in range(len(dev_set_neg)):
  #dev_set.append(''+neg_review)
  dev_file.append(' '.join(review_to_wordlist(dev_set_neg[i])))

train_file=[]
for i in range(len(train_set_pos)):
  #train_set.append(''+pos_review)
  train_file.append(' '.join(review_to_wordlist(train_set_pos[i])))
for i in range(len(train_set_neg)):
  #train_set.append(''+neg_review)
  train_file.append(' '.join(review_to_wordlist(train_set_neg[i])))

test_file=[]
for i in range(len(test_set_pos)):
  #test_set.append(''+pos_review)
  test_file.append(' '.join(review_to_wordlist(test_set_pos[i])))
for i in range(len(test_set_neg)):
  #test_set.append(''+neg_review)
  test_file.append(' '.join(review_to_wordlist(test_set_neg[i])))


from sklearn.model_selection import train_test_split
import random

dataset_full = train_file + test_file
size_dataset_full=len(dataset_full)
size_test=int(round(size_dataset_full*0.5,0))

list_test_indices=random.sample(range(size_dataset_full), size_test)
train_list=[]
test_list=[]
for i,example in enumerate(dataset_full):
  if i in list_test_indices: test_list.append(example)
  else: train_list.append(example)

random.shuffle(train_list)
random.shuffle(test_list)
print ("Size dataset full: "+str(size_dataset_full))
print ("Size training set: "+str(len(train_set)))
print ("Size test set: "+str(len(test_set)))

  ##return pre_train_set,pre_test_set
def get_train_test_split(dataset_full,ratio):
  train_set=[]
  test_set=[]
  size_dataset_full=len(dataset_full)
  size_test=int(round(size_dataset_full*ratio,0))
  list_test_indices=random.sample(range(size_dataset_full), size_test)
  for i,example in enumerate(dataset_full):
    if i in list_test_indices: test_set.append(example)
    else: train_set.append(example)
  return train_set,test_set

# To verify
train_null = []
#train_null, train_set=get_train_test_split(train_set,0.5)
#train_file, train_null = get_train_test_split(train_file,0.5)
print ("Size training set: "+str(len(train_set)))
print ("Size training file: "+str(len(train_file)))
'''
original_size_test=len(test_set)
size_dev=int(round(original_size_test*0.5,0))
list_dev_indices=random.sample(range(original_size_test), size_dev)
new_dev_set=[]
new_test_set=[]
for i,example in enumerate(test_set):
  if i in list_dev_indices: new_dev_set.append(example)
  else: new_test_set.append(example)
'''

new_test_set=[]
new_tain_set=[]
new_train_set=train_set
new_test_set=test_set

new_test_file = test_file
new_train_file = train_file

random.shuffle(new_train_set)
random.shuffle(new_test_set)

'''
print ("TRAINING SET")
print ("Size training set: "+str(len(new_train_set)))
for example in new_train_set[:3]:
  print (example)
print ("    \n-------\n")
print ("DEV SET")
print ("Size development set: "+str(len(new_dev_set)))
for example in new_dev_set[:3]:
  print (example)
print ("    \n-------\n")
print ("TEST SET")
print ("Size test set: "+str(len(new_test_set)))
for example in new_test_set[:3]:
  print (example)
'''
print ("Size new_train_set: "+str(len(new_train_set)))
print ("Size new_test_set: "+str(len(new_test_set)))
print ("Size new_train_file: "+str(len(new_train_file)))
print ("Size new_test_file: "+str(len(new_test_file)))


lemmatizer = nltk.stem.WordNetLemmatizer()
stopwords=set(nltk.corpus.stopwords.words('english'))
stopwords.add(".")
stopwords.add(",")
stopwords.add("--")
stopwords.add("``")

# Function taken from Session 1
def get_list_tokens(string): # Function to retrieve the list of tokens from a string
  sentence_split=nltk.tokenize.sent_tokenize(string)
  list_tokens=[]
  for sentence in sentence_split:
    list_tokens_sentence=nltk.tokenize.word_tokenize(sentence)
    for token in list_tokens_sentence:
      list_tokens.append(lemmatizer.lemmatize(token).lower())
  return list_tokens

# Function taken from Session 2
def get_vector_text(list_vocab,string):
  vector_text=np.zeros(len(list_vocab))
  list_tokens_string=get_list_tokens(string)
  for i, word in enumerate(list_vocab):
    if word in list_tokens_string:
      vector_text[i]=list_tokens_string.count(word)
  return vector_text


# Functions slightly modified from Session 2

def get_vocabulary(training_set, num_features): # Function to retrieve vocabulary
  dict_word_frequency={}
  for instance in training_set:
    sentence_tokens=get_list_tokens(instance[0])
    for word in sentence_tokens:
      if word in stopwords: continue
      if word not in dict_word_frequency: dict_word_frequency[word]=1
      else: dict_word_frequency[word]+=1
  sorted_list = sorted(dict_word_frequency.items(), key=operator.itemgetter(1), reverse=True)[:num_features]
  vocabulary=[]
  for word,frequency in sorted_list:
    vocabulary.append(word)
  return vocabulary

def train_svm_classifier(training_set, vocabulary): # Function for training our svm classifier
  X_train=[]
  Y_train=[]
  for instance in training_set:
    vector_instance=get_vector_text(vocabulary,instance[0])
    X_train.append(vector_instance)
    Y_train.append(instance[1])
  # Finally, we train the SVM classifier
  svm_clf=sklearn.svm.SVC(kernel="linear",gamma='auto')
  svm_clf.fit(np.asarray(X_train),np.asarray(Y_train))
  return svm_clf

vocabulary=get_vocabulary(new_train_set, 1000)  # We use the get_vocabulary function to retrieve the vocabulary

X_train=[]
Y_train=[]
for instance in new_train_set:
  vector_instance=get_vector_text(vocabulary,instance[0])
  X_train.append(vector_instance)
  Y_train.append(instance[1])


##TDIDF
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
tfidf = TFIDF(min_df=2, # Minimum support is 2
           max_features=None,
           strip_accents='unicode',
           analyzer='word',
           token_pattern=r'\w{1,}',
           ngram_range=(1, 3),  # Binary grammar model
           use_idf=1,
           smooth_idf=1,
           sublinear_tf=1,
           stop_words = 'english') # Remove English stop words

# Combine training and test sets for TFIDF vectorization
data_all = new_train_file + new_test_file
len_train = len(new_train_file)

tfidf.fit(data_all)
data_all = tfidf.transform(data_all)
# Restore to training set and test set parts
train_x = data_all[:len_train]
test_x = data_all[len_train:]
print ("TF-IDF process end.")

new_final_train_x = train_x + np.asarray(train_x)

# Finally, we train the SVM classifier
svm_clf=sklearn.svm.SVC(kernel="linear",gamma='auto')
#svm_clf.fit(np.asarray(train_x),np.asarray(label))
svm_clf.fit(train_x,label)

'''
X_test=[]
Y_test=[]
for instance in new_test_set:
  vector_instance=get_vector_text(vocabulary,instance[0])
  X_test.append(vector_instance)
  Y_test.append(instance[1])
X_test=np.asarray(X_test)
Y_test_gold=np.asarray(Y_test)
'''
X_test = test_x
Y_test_gold = test_y


#classify
from sklearn.metrics import classification_report
Y_text_predictions=svm_clf.predict(X_test)
print(classification_report(Y_test_gold, Y_text_predictions))

from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
precision=precision_score(Y_test_gold, Y_text_predictions, average='macro')
recall=recall_score(Y_test_gold, Y_text_predictions, average='macro')
f1=f1_score(Y_test_gold, Y_text_predictions, average='macro')
accuracy=accuracy_score(Y_test_gold, Y_text_predictions)

print ("Precision: "+str(round(precision,3)))
print ("Recall: "+str(round(recall,3)))
print ("F1-Score: "+str(round(f1,3)))
print ("Accuracy: "+str(round(accuracy,3)))

from sklearn.metrics import confusion_matrix
print (confusion_matrix(Y_test_gold, Y_text_predictions))
