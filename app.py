import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns



st.markdown('''
# Heart Disease Detector 
- This app detects if you have a heart disease based on Machine Learning!
- App built by Pranav Sawant and Anshuman Shukla of Team Skillocity.
- Note: User inputs are taken from the sidebar. It is located at the top left of the page (arrow symbol). The values of the parameters can be changed from the sidebar.

- Dataset creators:
- Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
- University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
- University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
- V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.
''')
st.write('---')

df = pd.read_csv(r'heart.csv')

# HEADINGS
st.title('Heart Disease Detector')
st.sidebar.header('Patient Data')
st.subheader('Training Dataset')
st.write(df.describe())


# X AND Y DATA
x = df.drop(['Outcome'], axis = 1)
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)


# FUNCTION
def user_report():
  age = st.sidebar.slider('Age', 0,200, 75 )
  trestbps = st.sidebar.slider('Resting Blood Pressure', 60,200, 126 )
  chol = st.sidebar.slider('Cholestrol', 100,500, 330 )
  thalach = st.sidebar.slider('Maximum Heart Rate Achieved', 60,250, 146 )
  oldpeak = st.sidebar.slider('ST Depression Induced by Exercise', 0.0,5.0, 2.50 )
  ca = st.sidebar.slider('Number of major vessels coloured by Flouroscopy', 0,5, 2 )
  

  user_report_data = {
      'age':age,
      'trestbps':trestbps,
      'chol':chol,
      'thalach':thalach,
      'oldpeak':oldpeak,
      'ca':ca,
      
         
  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data




# PATIENT DATA
user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)




# MODEL
rf  = RandomForestClassifier()
rf.fit(x_train, y_train)
user_result = rf.predict(user_data)



# VISUALISATIONS
st.title('Graphical Patient Report')



# COLOR FUNCTION
if user_result[0]==0:
  color = 'blue'
else:
  color = 'red'


# Age vs Trestbps
st.header('Resting Blood Pressure Value Graph (Yours vs Others)')
fig_trestbps = plt.figure()
ax3 = sns.scatterplot(x = 'Age', y = 'Resting Blood Pressure', data = df, hue = 'Outcome' , palette='Purples')
ax4 = sns.scatterplot(x = user_data['age'], y = user_data['trestbps'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(60,220,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_trestbps)


# Age vs Chol
st.header('Cholestrol Value Graph (Yours vs Others)')
fig_i = plt.figure()
ax9 = sns.scatterplot(x = 'Age', y = 'Cholestrol', data = df, hue = 'Outcome', palette='rainbow')
ax10 = sns.scatterplot(x = user_data['age'], y = user_data['chol'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(100,750,50))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_i)


# Age vs Thalach
st.header('Maximum Heart Rate Achieved Value Graph (Yours vs Others)')
fig_thalach = plt.figure()
ax5 = sns.scatterplot(x = 'Age', y = 'Maximum Heart Rate Achieved', data = df, hue = 'Outcome', palette='Blues')
ax6 = sns.scatterplot(x = user_data['age'], y = user_data['thalach'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(40,250,25))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_thalach)


# Age vs Oldpeak
st.header('ST Depression Induced by Exercise Value Graph (Yours vs Others)')
fig_oldpeak = plt.figure()
ax11 = sns.scatterplot(x = 'Age', y = 'ST Depression Induced by Exercise', data = df, hue = 'Outcome', palette='Greens')
ax12 = sns.scatterplot(x = user_data['age'], y = user_data['oldpeak'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0,8,0.5))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_oldpeak)


# Age vs Ca
st.header('Number of major vessels coloured by Flouroscopy Value Graph (Yours vs Others)')
fig_ca = plt.figure()
ax13 = sns.scatterplot(x = 'Age', y = 'Number of major vessels coloured by Flouroscopy', data = df, hue = 'Outcome', palette='rocket')
ax14 = sns.scatterplot(x = user_data['age'], y = user_data['ca'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0,5,1))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_ca)





# OUTPUT
st.subheader('Your Report: ')
output=''
if user_result[0]==0:
  output = 'Congratulations, you do not have any heart diseases.'
else:
  output = 'Unfortunately, it is likely that you may be having a heart disease.'
st.title(output)
st.subheader('Accuracy: ')
st.write(str(accuracy_score(y_test, rf.predict(x_test))*100)+'%')
