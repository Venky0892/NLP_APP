#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Data Manipulation
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
# NLP
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB


# Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# ML
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
# import xgboost
# Warnings
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Loading the data
source = pd.read_csv('data.csv', parse_dates=['published'])
source = source.fillna(0)
source['published'] = pd.to_datetime(source['published'], errors='coerce')
source['engagement'] = source['engagement'].astype(int)
source['word_count'] = source['word_count'].astype(int)
# source['Tweet_type'] = source['Tweet_type'].astype('category')
source = source[['published', 'content', 'word_count', 'engagement']]
print(source.shape)
print(source.head(5))


# In[3]:


# Checking the data types
source.info()


# In[4]:


# Taking a copy of the source file
data = source.copy()


# In[5]:


# Preprocessing the text data without removing negative stopwords

REPLACE_BY_SPACE_RE = re.compile("[/(){}\[\]\|@,;!]")
BAD_SYMBOLS_RE = re.compile("[^0-9a-z #+_]")
negation = ["no", "nor", "not", "don", "don't", "aren", "aren't", "couldn", "couldn't", "didn", "didn't", "doesn", "doesn't",
           "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "isn", "isn't", "mightn", "mightn't", "mustn", "mustn't",
           "needn", "needn't", "shan", "shan't", "shouldn", "shouldn't",  'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
stop = set(stopwords.words('english')) - set(negation)

# Custom stopwords
stoplist = ['i','me','my','myself','we','our','ours','ourselves','you',"you're","you've","you'll","you'd",'your',
            'yours','yourself','yourselves','he','him','his','himself','she',"she's",'her','hers','herself','it',
            "it's",'its','itself','they','them','their','theirs','themselves','what','which','who','whom','this','that',"that'll",
            'these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did',
            'doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about',
            'against','between','into','through','during','before','after','above','below','to','from','up','down','in','out',
            'on','off','over','under','again','further','then','once','here','there','when','where','why','all','any',
            'both','each','few','more','most','other','some','such','only','own','same','so','than','too',
            'very','s','t','can','will','just','should',"should've",'now','d','ll','m','o','re','ve','y','rt','rt','qt','for',
            'the','with','in','of','and','its','it','this','i','have','has','would','could','you','a','an',
            'be','am','can','edushopper','will','to','on','is','by','ive','im','your','we','are','at','as','any','ebay','thank','hello','know',
            'need','want','look','hi','sorry','http','body','dear','hello','hi','thanks','sir','tomorrow','sent','send','see','there','welcome','what','well','us']

stop.update(set(stoplist))

def text_preprocess(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = re.sub(r'\d', '', str(text))  # removing digits
    text = re.sub(r"(?:\@|https?\://)\S+", "", str(text)) #removing mentions and urls
    text = text.lower() # lowercase text
    text =  re.sub('[0-9]+', '', text)
    text = REPLACE_BY_SPACE_RE.sub(" ", text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub(" ", text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join([word for word in text.split() if word not in stop]) # delete stopwors from text
    text = text.strip()
    return text


# In[6]:


data['content'] = data['content'].apply(text_preprocess)
print(data.head())


# In[7]:


# Checking the distribution of engagment 
fig, ax = plt.subplots(1,2, figsize=(12,5))
data['engagement'].plot(kind='hist', ax=ax[0])
ax[0].set_xlabel("Engagement")

data['word_count'].plot(kind='hist', ax=ax[1])
ax[1].set_xlabel("Word Count")
plt.show()


# The distribution tells us that, most of the tweets received no engagement from the users. So, we will limit our analysis only for engagement values greater than 0. 
# 
# The distribution of word count shows that, the average number of words in the tweets are around 15.

# In[8]:


# Now, Filter only for non zero engagement values
data = data[data['engagement'] > 0]
print(data.shape)
data.head()


# In[9]:


# Now lets check the descriptive stats
data.describe()


# In[10]:


data['engagement_bucket'] = pd.qcut(data['engagement'], q=[0,0.5, 0.75, 1], labels=['Low', 'Medium', 'High'])
data.head()


# In[11]:


sns.countplot(x='engagement_bucket', data=data)
plt.show()


# In[12]:


# Creating time related features such as time, day, etc.
data['day'] = data['published'].dt.day
data['hour'] = data['published'].dt.hour
data['week_day'] = data['published'].dt.weekday


# In[13]:


data.head()


# In[14]:


# Which hour of the day, the posts are getting higher average engagement?
hour = data.groupby('hour')['engagement'].mean()
hour.plot(figsize=(12,5))
plt.xlabel("Hour of Day")
plt.ylabel("Engagement")
plt.show()


# In[15]:


# Which day of the week, the posts are getting higher average engagement? (Monday=0, Sunday=6)
weekday = data.groupby('week_day')['engagement'].mean()
weekday.plot(figsize=(12,5))
plt.xlabel("Day of the Week")
plt.ylabel("Engagement")
plt.show()


# In[16]:


# Which day of the month, the posts are getting higher average engagement? (Monday=0, Sunday=6)
dayofmonth = data.groupby('day')['engagement'].mean()
dayofmonth.plot(figsize=(12,5))
plt.xlabel("Day of the Month")
plt.ylabel("Engagement")
plt.show()


# In[17]:


# Creating Features and Labels
X = data[['word_count', 'hour', 'week_day']]
X = pd.get_dummies(X, drop_first=True)
X['content'] = data['content']
X.reset_index(drop=True,inplace=True)

y= data['engagement_bucket'].values
y = pd.get_dummies(y)


# In[18]:


X.head()


# In[19]:


y.head()


# In[20]:


# Splitting the data into Train and Validation set
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=123)

print(X_train.shape)
print(X_valid.shape)


# In[21]:


# Extracting features from the text column
vec = TfidfVectorizer(strip_accents='unicode', ngram_range=(1,2), max_features=3000, smooth_idf=True, sublinear_tf=True)
train_vec = vec.fit_transform(X_train['content'])
valid_vec = vec.transform(X_valid['content'])


# In[22]:


train_vec.toarray().shape


# In[23]:


X_train = np.hstack([X_train.drop('content', axis=1), train_vec.toarray()])
X_valid = np.hstack([X_valid.drop('content', axis=1), valid_vec.toarray()])


# In[24]:


# After adding the vectorized text columns
print(X_train.shape)
print(X_valid.shape)


# In[25]:


# Fitting a baseline model
logreg = OneVsRestClassifier(LogisticRegression(solver='sag'))
clf = MultinomialNB()

cv_score = cross_val_score(logreg, X_train, y_train.values, cv=10, scoring='roc_auc')
print(cv_score.mean())


# In[26]:


logreg.fit(X_train,y_train)
pred = logreg.predict(X_valid)
# pred2 = clf.predict(X_valid)
print("Predict hererererererer", pred)


# In[27]:


from sklearn.metrics import roc_auc_score, f1_score

roc_auc_score(y_valid, pred)


# In[28]:


f1_score(y_valid, pred, average='weighted')


# In[ ]:

import pickle

file1 = open("vectorizer.pkl", 'wb')
pickle.dump(vec, file1, pickle.HIGHEST_PROTOCOL)

file2 = open("logisticreg.pkl", 'wb')
pickle.dump(logreg, file2, pickle.HIGHEST_PROTOCOL)

file1.close()
file2.close()











