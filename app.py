# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 09:05:03 2020

@author: Venkatesh Prasath and Hariharan
"""

import pickle
import base64
import time
import numpy as np
import pandas as pd
import spacy
import streamlit as st
from streamlit import components
# from keras.models import Sequential
# from keras.layers import Dense
# from flair.data import Sentence
# from flair.models import TextClassifier
from pathlib import Path

from neural import *
from tensorflow.keras.models import model_from_json
# from utils import read_data
from tensorflow.keras import backend as k
from tensorflow.keras import models
# import keras.backend.tensorflow_backend as K


# from keras.models import model_from_json
# from keras.models import Sequential
# from keras.layers import Dense



from textblob import TextBlob
import streamlit.components.v1 as components
from Topic_modelling import *
st.set_page_config(layout='wide')
import nltk
nltk.download('stopwords')

# Custom Tokenizer
def tokenize(text):
    return [word for word in text.split() if len(word) > 2]


# Function to get the class label
def get_keys(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key




def getAnalysis(score):
    if score < 0:
        return "Negative"
    elif score == 0:
        return "Neutral"
    else:
        return "Positive"


def getSubjectivity(new_text):
    return TextBlob(new_text).sentiment.subjectivity


def getPolarity(new_text):
    return TextBlob(new_text).sentiment.polarity

def read_csv(file):
    try:
        data = pd.read_csv(file, dtype=object)
    except:
        data = pd.read_excel(file, dtype=object)


    # data = data["content"]
    return data



def main():
    # cs_sidebar()
    cs_body()
    cs_sidebar()
    return None

def cs_sidebar():
    # st.markdown(
    #     """
    # <style>
    # .reportview-container .markdown-text-container {
    #     font-family: monospace;
    # }
    # .sidebar .sidebar-content {
    #     # background-image: linear-gradient(#AED6F1,#AED6F1);
    #     color: white;
    # }
    # .Widget>label {
    #     color: white;
    #     font-family: monospace;
    # }
    # [class^="st-b"]  {
    #     color: black;
    #     font-family: monospace;
    # }
    # .st-bb {
    #     background-color: transparent;
    # }
    # .st-at {
    #     background-color: #AED6F1;
    # }
    # footer {
    #     font-family: monospace;
    # }
    # .reportview-container .main footer, .reportview-container .main footer a {
    #     color: #AED6F1;
    # }
    # header .decoration {
    #     background-image: none;
    # }
    #
    # </style>
    # """,
    #     unsafe_allow_html=True,
    # )
    return None


@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return
#FF5733
# @st.cache(suppress_st_warning=True)
def cs_body():
    # st.markdown("""
    #     <style>
    #     body {
    #         color: #212F3D;
    #         background-color: #E5E8E8;
    #     }
    #     </style>
    #         """, unsafe_allow_html=True)
    st.title("Social Media NLP App")
    # st.subheader('by Venkatesh and Hariharan')
    st.markdown(
        """
       This application helps users to identify basic NLP related tasks.
        """, unsafe_allow_html=True)
    # set_png_as_page_bg('download.png')

    # Types of activity you can perform
    type = ["Sentiment Analyzer", "Engagement Prediction", "NLP Analyzer", "Topic Modelling", "Train_Engagement"]
    deselect = list(set(type))
    activity = st.sidebar.selectbox("What do you want to perform?", deselect)

    if "Train_Engagement" in activity:
        st.markdown("""
                <style>
                body {
                    color: #212F3D;
                    background-color: #FFD7B2K;
                }
                </style>
                    """, unsafe_allow_html=True)
        file = st.sidebar.file_uploader("Upload file")


        if file is not None:
            df = pd.read_excel(file)
            date_column = st.sidebar.selectbox('select date column:', df.columns)
            text_column = st.sidebar.selectbox('select text column:', df.columns)
            word_count = st.sidebar.selectbox('select word_count column:', df.columns)
            engagement_column = st.sidebar.selectbox('select engagement column:', df.columns)

            # start_execution = st.button("Train the model")
            if st.button("Train the model"):
                # gif_runner = st.image("Bars-1s-201px.gif")
                model = Neural_model(file, date_column, engagement_column, word_count, text_column)
                model.result()
                # gif_runner.empty()
                st.success("Your model has been trained, Now predict your engagement")

    if "Topic Modelling" in activity:

        st.markdown("""
                        <style>
                        body {
                            color: #212F3D;
                            background-color: #FFD7B2K;
                        }
                        </style>
                            """, unsafe_allow_html=True)

        file = st.sidebar.file_uploader("Upload a csv file")
        user_input = st.sidebar.text_area("Fill your stopwords here with space!!" )



        if file is not None:
            df = pd.read_excel(file)
            cols = st.multiselect('select text column to form themes:', df.columns, default=[])
            df['text'] = df[cols].astype('str')
            if st.button("Predict"):
                user_input = user_input.split()
                res, vis, dominant_topic, represent_sen = result(df, user_input)
                st.header("Top 5 Topics")
                st.write(res)
                st.header("Dominant Topic And It's Percentage Contribution On Each Document")
                st.write(dominant_topic.head(10))
                st.header("Most Representative Sentence For Each Topic")
                st.write(represent_sen)
                pyLDAvis.save_html(vis, 'LDA_Visualization.html')
                st.header('LDA_Visualization')
                HtmlFile = open("LDA_Visualization.html", 'r', encoding='utf-8')
                source_code = HtmlFile.read()
                components.html(source_code, height = 800, width = 1200)
        else:
            st.success("Please upload the Csv or Xlsx file.")

    if "Engagement Prediction" in activity:
        st.markdown("""
                                <style>
                                body {
                                    color: #212F3D;
                                    background-color: #FFD7B2K;
                                }
                                </style>
                                    """, unsafe_allow_html=True)
        new_text = st.text_area("Enter Post Text", "Type here .....")

        file1 = open('vectorizer.pkl', 'rb')
        # file2 = open('Logisticreg.pkl', 'rb')
        vect = pickle.load(file1)
        # model = pickle.load(file2)
        # file1.close()
        # file2.close()
        #creating a nueral net
        my_file = Path("model.json")
        if my_file.is_file():
            json_file = open('model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights("model.h5")
            word_count = len([i for i in str(new_text).split()])
            hour = st.sidebar.slider("Hour of the day", 0, 23, 16)
            # 		day_of_week = st.sidebar.slider("Day of the week", 0, 6, 2)
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_of_week = st.sidebar.selectbox("Day of the week", days)
            # day_of_month = st.sidebar.slider("Day of the month", 1, 31, 1)
            # tweet_type = st.sidebar.selectbox("Type of tweet", ("Organic", "Reply", "Retweet"))

            if day_of_week == 'Monday':
                day_val = 0
            elif day_of_week == 'Tuesday':
                day_val = 1
            elif day_of_week == 'Wednesday':
                day_val = 2
            elif day_of_week == 'Thursday':
                day_val = 3
            elif day_of_week == 'Friday':
                day_val = 4
            elif day_of_week == 'Saturday':
                day_val = 5
            else:
                day_val = 6

            test = pd.DataFrame({'word_count': word_count,
                                 'hour': hour,
                                 'week_day': day_val,
                                 }, index=[0])

            if st.button("Predict"):
                vect_text = vect.transform([new_text]).toarray()
                final_test = np.hstack([test, vect_text])
                # model = loaded_model()
                # prediction = loaded_model.predict_classes(final_test)
                prediction_1 = np.argmax(loaded_model.predict(final_test), axis=-1)
                # st.success(prediction_1)
                i = 0
                if prediction_1[0] == 0:
                    final_class = 'Low Engagement Post'
                elif prediction_1[0] == 1:
                    final_class = 'Medium Engagement Post'
                else:
                    final_class = 'High Engagement Post'

                st.success("Text Categorized as:: {}".format(final_class))
        else:
            st.success("Train your data before you predict the engagement")

    if "Sentiment Analyzer" in activity:
        st.markdown("""
                                <style>
                                body {
                                    color: #212F3D;
                                    background-color: #FFD7B2K;
                                }
                                </style>
                                    """, unsafe_allow_html=True)

        new_text = st.text_area("Enter Post Text", "Type here .....")
        st.info("Get Your Sentiment")
        type_sentiments = ["Social Media Sentiments", "Movie and Products Sentiments"]
        activity = st.sidebar.radio("Choose The Sentiment Analyzer?", type_sentiments)

        if "Social Media Sentiments" in activity:
            if st.button("Predict"):
                # Create two new columns ???Subjectivity??? & ???Polarity???

                Subjectivity = getSubjectivity(new_text)
                Polarity = getPolarity(new_text)

                Result = getAnalysis(Polarity)

                # blob_object = TextBlob(new_text, analyzer=NaiveBayesAnalyzer())
                #
                # # Running sentiment analysis
                # analysis = blob_object.sentiment
                # print(analysis)
                # if analysis[0] == 'pos':
                # 	Sentence = "Positive"
                # else:
                # 	analysis[0] == 'neg'
                # 	Sentence = "Negative"
                st.success("Text Sentiment:: {}".format(Result))

        else:

            if st.button("Predict"):

                # classifier = TextClassifier.load('en-sentiment')
                # sentence = Sentence(new_text)
                Polarity = getPolarity(new_text)
                Result = getAnalysis(Polarity)
                # classifier.predict(sentence)

                st.success("Text Sentiment:: {}".format(Result))

if __name__ == '__main__':
    main()


