import pandas as pd
import numpy as np
import re
import xlrd
import nltk
import dill as pickle
nltk.download('stopwords')
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dropout
from tensorflow.keras.constraints import min_max_norm
from imblearn.over_sampling import SMOTE
import time
import streamlit as st
from log_util import log
import tensorflow as tf
from tensorflow import keras
import kerastuner as kt





from tensorflow.keras.utils import *

from tensorflow.keras import backend as k
from tensorflow.keras import models
# from keras import models
# from keras import backend
# from tensorflow.keras.utils.generic_utils import transpose_shape
# # import _pywrap_tensorflow_internal
from tensorflow.keras.models import model_from_json

# NLP
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
# Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer



@st.cache(suppress_st_warning=True, allow_output_mutation=True)
class Neural_model:
    def __init__(self, dataset,date, engagement, wordcount, text):
        self.dataset = dataset
        self.date = date
        self.engagement = engagement
        self.wordcount = wordcount
        self.textcolumn = text

    def REPLACE_BY_SPACE_RE(self):
        try:
             REPLACE_BY_SPACE_RE = re.compile("[/(){}\[\]\|@,;!]")

        except Exception as e:
            log('----------Error in REPLACE_BY_SPACE_RE function ----------:{}'.format(e), 'error')
            raise e
        return REPLACE_BY_SPACE_RE

    def BAD_SYMBOLS_RE(self):
        try:
            BAD_SYMBOLS_RE = re.compile("[^0-9a-z #+_]")
        except Exception as e:
            log('----------Error in BAD_SYMBOLS_RE function ----------:{}'.format(e), 'error')
            raise e
        return BAD_SYMBOLS_RE

    @st.cache(suppress_st_warning=True, allow_output_mutation=True)
    def stopwords_update(self):

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

        stop = stop.update(set(stoplist))
        return stop

    @st.cache(suppress_st_warning=True, allow_output_mutation=True)
    def read_data(self, data = None):
        try:
          data = pd.read_excel(data, parse_dates=['published'] )
          data['published'] = pd.to_datetime(data['published'], errors='coerce')
          data['engagement'] = data['engagement'].astype(int)
          data['word_count'] = data['word_count'].astype(int)
        # source['Tweet_type'] = source['Tweet_type'].astype('category')
          source = data[['published', 'content', 'word_count', 'engagement']]

        except Exception as e:
            log('----------Error in Read data ----------:{}'.format(e), 'error')
            raise e
        return source


    # def copy_source(self, data = None):
    #   data = data.copy()
    #   return data
    #
    # copy = copy_source(data)

    @st.cache(suppress_st_warning=True, allow_output_mutation=True)
    def text_preprocess(self, text):
        try:
            """
                text: a string
    
                return: modified initial string
            """
            negation = ["no", "nor", "not", "don", "don't", "aren", "aren't", "couldn", "couldn't", "didn", "didn't",
                        "doesn", "doesn't",
                        "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "isn", "isn't", "mightn", "mightn't",
                        "mustn", "mustn't",
                        "needn", "needn't", "shan", "shan't", "shouldn", "shouldn't", 'wasn', "wasn't", 'weren', "weren't",
                        'won', "won't", 'wouldn', "wouldn't"]
            stop = set(stopwords.words('english')) - set(negation)

            # Custom stopwords
            stoplist = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
                        "you'd", 'your',
                        'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
                        'herself', 'it',
                        "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who',
                        'whom', 'this', 'that', "that'll",
                        'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                        'having', 'do', 'does', 'did',
                        'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
                        'by', 'for', 'with', 'about',
                        'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
                        'from', 'up', 'down', 'in', 'out',
                        'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
                        'why', 'all', 'any',
                        'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than',
                        'too',
                        'very', 's', 't', 'can', 'will', 'just', 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',
                        've', 'y', 'rt', 'rt', 'qt', 'for',
                        'the', 'with', 'in', 'of', 'and', 'its', 'it', 'this', 'i', 'have', 'has', 'would', 'could', 'you',
                        'a', 'an',
                        'be', 'am', 'can', 'edushopper', 'will', 'to', 'on', 'is', 'by', 'ive', 'im', 'your', 'we', 'are',
                        'at', 'as', 'any', 'ebay', 'thank', 'hello', 'know',
                        'need', 'want', 'look', 'hi', 'sorry', 'http', 'body', 'dear', 'hello', 'hi', 'thanks', 'sir',
                        'tomorrow', 'sent', 'send', 'see', 'there', 'welcome', 'what', 'well', 'us']

            stop.update(set(stoplist))
            REPLACE_BY_SPACE_RE = re.compile("[/(){}\[\]\|@,;!]")
            BAD_SYMBOLS_RE = re.compile("[^0-9a-z #+_]")
            text = re.sub(r'\d', '', str(text))  # removing digits
            text = re.sub(r"(?:\@|https?\://)\S+", "", str(text))  # removing mentions and urls
            text = text.lower()  # lowercase text
            text = re.sub('[0-9]+', '', text)
            text = REPLACE_BY_SPACE_RE.sub(" ", text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
            text = BAD_SYMBOLS_RE.sub(" ", text)  # delete symbols which are in BAD_SYMBOLS_RE from text
            text = ' '.join([word for word in text.split() if word not in stop])  # delete stopwors from text
            text = text.strip()
        except Exception as e:
            log('----------Error in Text Processing ----------:{}'.format(e), 'error')
            raise e
        return text

    @st.cache(suppress_st_warning=True, allow_output_mutation=True)
    def data_processing(self, data):

        try:
            data['content'] = data['content'].apply(self.text_preprocess)
            # data['content'] = self.text_preprocess(data['content'])

            data = data[data['engagement'] > 0]

            data['engagement_bucket'] = pd.qcut(data['engagement'], q=[0,0.5, 0.75, 1], labels=['Low', 'Medium', 'High'])

            # Creating time related features such as time, day, etc.
            data['day'] = data['published'].dt.day
            data['hour'] = data['published'].dt.hour
            data['week_day'] = data['published'].dt.weekday

            hour = data.groupby('hour')['engagement'].mean()
            weekday = data.groupby('week_day')['engagement'].mean()
            dayofmonth = data.groupby('day')['engagement'].mean()

            X = data[['word_count', 'hour', 'week_day']]
            X = pd.get_dummies(X, drop_first=True)

            X['content'] = data['content']
            X.reset_index(drop=True,inplace=True)

            y= data['engagement_bucket']
            # y = pd.get_dummies(y)
        except Exception as e:
            log('----------Error in Data Processing ----------:{}'.format(e), 'error')
            raise e
        return X, y

    @st.cache(suppress_st_warning=True, allow_output_mutation=True)
    def TfidfVectorizer(self, X, y ):
        try:
            vec = TfidfVectorizer(strip_accents='unicode', ngram_range=(1,2), max_features=3000, smooth_idf=True, sublinear_tf=True)
            train_vec = vec.fit_transform(X['content'])

            _train = np.hstack([X.drop('content', axis=1), train_vec.toarray()])
            y = LabelEncoder().fit_transform(y)
            scaler = Normalizer().fit(_train)
            _train = scaler.transform(_train)
        except Exception as e:
            log('----------Error in TfidVectorizer ----------:{}'.format(e), 'error')
            raise e
        return _train , y

    @st.cache(suppress_st_warning=True, allow_output_mutation=True)
    def balancing(self,_train , y ):
        try:
            smote = SMOTE('minority')
            _train, y = smote.fit_sample(_train, y)
        except Exception as e:
            log('----------Error in Smote ----------:{}'.format(e), 'error')
            raise e
        return _train, y

    @st.cache(suppress_st_warning=True, allow_output_mutation=True)
    def model(self, kernel_initializer='glorot_uniform',  activation = 'relu', dropout_rate=0.5, weight_constraint=0):
        try:
        # define the keras model
            model = Sequential()
            model.add(Dense(300, input_dim=3003, activation=activation, kernel_initializer=kernel_initializer, kernel_constraint=min_max_norm(min_value=1.0, max_value=1.0)))
            model.add(Dropout(dropout_rate))
            model.add(Dense(200, activation=activation, kernel_initializer=kernel_initializer,kernel_constraint=min_max_norm(min_value=1.0, max_value=1.0)))
            model.add(Dropout(dropout_rate))
            model.add(Dense(100, activation=activation, kernel_initializer=kernel_initializer, kernel_constraint=min_max_norm(min_value=1.0, max_value=1.0)))
            model.add(Dropout(dropout_rate))
            model.add(Dense(3, activation='softmax', kernel_initializer=kernel_initializer, kernel_constraint=min_max_norm(min_value=1.0, max_value=1.0)))

            # compile the keras model
            # optimizer = SGD(lr=learn_rate, momentum=momentum)
            model.compile(loss='sparse_categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
        except Exception as e:
            log('----------Error in Clean_ts function ----------:{}'.format(e), 'error')
            raise e
        return model


    def final_optimised_model(self, model, _train, y, epochs=None, batch_size=None):
        try:
            # fit the keras model on the dataset
            model.fit(_train, y, epochs=epochs, batch_size=batch_size)

            # evaluate the keras model
            _, accuracy = model.evaluate(_train, y)
            print('Accuracy: %.2f' % (accuracy*100))

            # make class predictions with the model
            # predictions_2 = model.predict_classes(_train)
            time.sleep(10)
            try:
                # serialize model to JSON
                model_json = model.to_json()
                with open("model.json", "w") as json_file:
                    json_file.write(model_json)
                # serialize weights to HDF5
                model.save_weights("model.h5")
                time.sleep(100)
            except Exception as e:
                log('----------Error in Model.json ----------:{}'.format(e), 'error')
                raise e

            # if 'model.json' and 'model.h5':

            # load json and create model



            print("End of the model running")
        except Exception as e:
            log('----------Error in Clean_ts function ----------:{}'.format(e), 'error')
            raise e
        return

    @st.cache(suppress_st_warning=True, allow_output_mutation=True)
    def updating_hyperameters(self, create_model=None, X=None, Y=None):
        try:
            model = KerasClassifier(build_fn=create_model, verbose=1)
            # define the grid search parameters
            batch_size = [10, 20, 40, 60, 80, 100]
            epochs = [10, 50, 100]
            optimizer = ['SGD', 'Adam']
            learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
            momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
            # init_mode = ['uniform', 'glorot_uniform',  'glorot_normal', 'normal', 'zero']
            # activation = ['softmax', 'relu', 'tanh', 'sigmoid',  'linear']
            weight_constraint = [1, 2, 3, 4, 5]
            # dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            param_grid = dict(batch_size=batch_size, epochs=epochs)
            grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
            grid_result = grid.fit(X, Y)
            best_score = grid_result.best_score_
            best_params = grid_result.best_params_
        except Exception as e:
            log('----------Error in updating Parameters ----------:{}'.format(e), 'error')
            raise e
        return best_score, best_params
    def optimized_model(self):
        try:
            model = self.model()
            loaded_model = self.final_optimised_model(model, self._train, self.label,self.best_params['epochs'],
                                                      self.best_params['batch_size'])
        except Exception as e:
            log('----------Error in Optimized Modeln ----------:{}'.format(e), 'error')
            raise e
        return loaded_model
    @st.cache(suppress_st_warning=True, allow_output_mutation=True)
    def result(self):
        try:
            data = self.read_data(self.dataset)
            X,y = self.data_processing(data)
            _train, y = self.TfidfVectorizer(X,y)
            _train, y = self.balancing(_train, y)
            self._train = _train
            self.label = y
            best_score, best_params = self.updating_hyperameters(self.model, _train, y)
            self.best_params = best_params
        except Exception as e:
            log('----------Error in Result ----------:{}'.format(e), 'error')
            raise e
        return self.optimized_model()

