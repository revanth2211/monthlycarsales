import streamlit as st
import pickle
import sklearn
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
from sklearn.pipeline import make_pipeline
text_model = make_pipeline(CountVectorizer(),MultinomialNB())
st.title('sales Prediction')
st.sidebar.header('month')

# FUNCTION
def user_report():
  month = st.sidebar.slider('month', 1,12, 1 )
  sales = text_model
  user_data = {'sales':sales}
  report_data = pd.DataFrame(user_data, index=[0])
  return report_data

user_data = user_report()
st.header('sales')
st.write(user_data)

sales = st.dataframe(user_data)
st.subheader(+str(np.round(sales[0], 2)))