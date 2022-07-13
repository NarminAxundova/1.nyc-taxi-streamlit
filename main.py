import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
st.markdown(
    """
    <style>
    .main {background-color:#F5F5F5;
    }
    </style>
    
    """,
    unsafe_allow_html=True
)

@st.cache
def get_data(filename):
    taxi_data=pd.read_csv(filename)
    return taxi_data

header=st.container()
dataset=st.container()
features=st.container()
modelTraining=st.container()

with header:
    st.title('Welcome to my project!')
    st.text('In this project I look into the transactions of taxis in NYC')


with dataset:
    st.header('NYC taxi dataset')
    st.text('I found this dataset from nanana.com')
    taxi_data=get_data('df_alll.csv')
    st.write(taxi_data)
    
    st.subheader('Pick-up location ID distribution on the NYC dataset')
    puloaction_dist=pd.DataFrame(taxi_data['PULocationID'].value_counts()).head(50)
    st.bar_chart(puloaction_dist)
    
with features:
    st.header('The features I created')
    st.markdown('* **first feature:** I created this beacuse ...')
    st.markdown('* **second feature:** I created this beacuse ...')

with modelTraining: 
    st.header('Time to train the model')
    st.text('Choose hyperparameters of model and see how the performance changes!')
    
    sel_col,disp_col=st.columns(2)
    max_depth=sel_col.slider('What should be max_depth of the model?', min_value=10,max_value=100,value=20,step=10)
    n_estimators=sel_col.selectbox('How many trees should  there be?', options=[100,200,300,'No limits'],index=0)
    sel_col.text('Here is list of features in my data')
    sel_col.write(taxi_data.columns)
    
    input_feature=sel_col.text_input('Which feature should be used as the input feature?','PULocationID')
    
    if n_estimators=='No limits':
        regr=RandomForestRegressor(max_depth=max_depth)
    else:
        regr=RandomForestRegressor(max_depth=max_depth,n_estimators=n_estimators)
    X =taxi_data[[input_feature]]
    y=taxi_data[['trip_distance']]
    regr.fit(X,y)
    prediction=regr.predict(y)
    
    disp_col.subheader('Mean absalute error of the model is : ')
    disp_col.write(mean_absolute_error(y,prediction))
    disp_col.subheader('Mean squared error of the model is : ')
    disp_col.write(mean_squared_error(y,prediction))
    disp_col.subheader('r2 score of the model is : ')
    disp_col.write(r2_score(y,prediction))