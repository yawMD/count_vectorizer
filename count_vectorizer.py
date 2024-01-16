#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet


# In[4]:


nltk.download("wordnet")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")


# In[14]:


#https://www.kaggle
get_ipython().system('curl -o bbc_text_cls.csv https://lazyprogrammer.me/course_files/nlp/bbc_text_cls.csv')


# In[15]:


df = pd.read_csv('bbc_text_cls.csv')


# In[16]:


df.head()


# In[18]:


df.describe()


# In[19]:


inputs = df['text']
labels = df['labels']


# In[20]:


labels.hist(figsize=(10,5));


# In[21]:


inputs_train, inputs_test, Ytrain, Ytest = train_test_split(
inputs, labels, random_state=123)


# In[22]:


vectorizer = CountVectorizer()


# In[23]:


Xtrain = vectorizer.fit_transform(inputs_train)
Xtest = vectorizer.transform(inputs_test)


# In[24]:


Xtrain


# In[25]:


print(Xtrain)


# In[26]:


(Xtrain != 0).sum()


# In[27]:


(Xtrain != 0).sum()/np.prod(Xtrain.shape)


# In[28]:


model = MultinomialNB()
model.fit(Xtrain, Ytrain)
print("train score:", model.score(Xtrain, Ytrain))
print("test score:", model.score(Xtest,Ytest))


# In[30]:


#with stopwords
vectorizer = CountVectorizer(stop_words='english')
Xtrain = vectorizer.fit_transform(inputs_train)
Xtest = vectorizer.transform(inputs_test)
model = MultinomialNB()
model.fit(Xtrain,Ytrain)
print("train score:", model.score(Xtrain,Ytrain))
print("test score:", model.score(Xtest,Ytest))


# In[31]:


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


# In[40]:


class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        tokens = word_tokenize(doc)
        words_and_tags = nltk.pos_tag(tokens)
        return [self.wnl.lemmatize(word, pos=get_wordnet_pos(tag)) \
               for word, tag in words_and_tags]


# In[41]:


#with lemmatization
vectorizer = CountVectorizer(tokenizer=LemmaTokenizer())
Xtrain = vectorizer.fit_transform(inputs_train)
Xtest = vectorizer.transform(inputs_test)
model = MultinomialNB()
model.fit(Xtrain,Ytrain)
print("train score:", model.score(Xtrain,Ytrain))
print("test score:", model.score(Xtest,Ytest))


# In[42]:


class StemTokenizer:
    def __init__(self):
        self.porter = PorterStemmer()
    def __call__(self, doc):
        tokens = word_tokenize(doc)
        return [self.porter.stem(t) for t in tokens]


# In[43]:


#with stemming
vectorizer = CountVectorizer(tokenizer=StemTokenizer())
Xtrain = vectorizer.fit_transform(inputs_train)
Xtest = vectorizer.transform(inputs_test)
model = MultinomialNB()
model.fit(Xtrain,Ytrain)
print("train score:", model.score(Xtrain,Ytrain))
print("test score:", model.score(Xtest,Ytest))


# In[46]:


#simple tokenizer
def simple_tokenizer(s):
    return s.split()


# In[48]:


#with simple tokenizer
vectorizer = CountVectorizer(tokenizer=simple_tokenizer)
Xtrain = vectorizer.fit_transform(inputs_train)
Xtest = vectorizer.transform(inputs_test)
model = MultinomialNB()
model.fit(Xtrain,Ytrain)
print("train score:", model.score(Xtrain,Ytrain))
print("test score:", model.score(Xtest,Ytest))


# In[ ]:




