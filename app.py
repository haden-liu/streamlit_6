import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk




DATA_URL = (
"Jan_2020_ontime.csv"
)

st.title("Flight Delay Analysis Jan 2020")
st.markdown("This application is a Streamlit dashboard that"
"to analyze flight delay in US Cities")

@st.cache(persist=True)
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    column1 = ['DAY_OF_MONTH', 'DAY_OF_WEEK','OP_CARRIER_AIRLINE_ID', 'TAIL_NUM','OP_CARRIER_FL_NUM','DEP_TIME','DEP_DEL15', 'ARR_TIME',  'ARR_DEL15', 'CANCELLED','DIVERTED', 'DISTANCE']
    data=data[column1]
    data.dropna(inplace=True)
    return data

data = load_data(150000)

st.header("How many flight occur during a day of month?")
dayMonth = st.selectbox("Day to look at", range(1,32),1)
data = data[data['DAY_OF_MONTH'] == dayMonth]


if st.checkbox("Show Raw Data", False):
    st.subheader("Raw Data")
    st.write(data)

if st.button("Column Names", False):
    st.write(data.columns)

if st.checkbox('Shape of Dataset'):
    st.write(data.shape)
    data_dim = st.radio("Show Dimension By  ",("Rows","Columns"))
    if data_dim == 'Row':
        st.text("Num of rows")
        st.write(data.shape[0])
    if data_dim == 'Columns':
        st.text("Num of Columns")
        st.write(data.shape[1])
    else:
        st.write(data.shape)

# st.write("""
# # Prediction
# """
# )
# st.write('---')

# X = data.drop(columns=['ARR_DEL15','DEP_DEL15','TAIL_NUM'],axis=1)
# y = data['ARR_DEL15']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# model = RandomForestClassifier(n_estimators=100)

# model.fit(X_train, y_train)

# pred = model.predict(X_test)

# feature_imp_rf = pd.Series(model.feature_importances_,index=X.columns).sort_values(ascending=False)

# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X)

# st.header('Feature Importance')
# plt.title('Feature importance based on shap values')
# shap.summary_plot(shap_values, X)
# st.pyplot(bbox_inches='tight')
# st.write('---')

# plt.title('Feature importance based on shap values (bar)')
# shap.summary_plot(shap_values, X, plot_type='bar')
# st.pyplot(bbox_inches='tight')#
