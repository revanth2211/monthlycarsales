import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
st.title('MONTHLY CAR SALES PREDICTION')
model = joblib.load('monthly_carsales')
ip = st.slider("pick number of months",0,100,0)
ip = np.array([1:100])
op = model.predict(ip)
plt.figure(figsize = (25,10))
fig = model.plot(op)
plt.show( )
st.pyplot(fig)

