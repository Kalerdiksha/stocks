import streamlit as st
from datetime import date

import yfinance as yf 
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# set up CSS style
st.markdown("""
    <style>
        background-color: #F5F5F5;


         
        .main {
            background-color: #F5F5F5;
            padding: 5rem;
        }

        .title {
            font-size: 3rem;
            font-weight: bold;
            color: #000080;
            margin-bottom: 2rem;
        }

        .subheader {
            font-size: 2rem;
            font-weight: bold;
            color: #000080;
            margin-top: 3rem;
            margin-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# create Streamlit app
st.markdown('<div class="main">', unsafe_allow_html=True)

st.markdown('<h1 class="title">Stock Analysis</h1>', unsafe_allow_html=True)

stocks =("AAPL","GOOGL","MSFT","GME",
" BPCL.NS","ASIANPAINT.NS","HINDUNILVR.NS","NESTLEIND.NS" ,"TITAN.NS","BAJAJ-AUTO.NS",
"SUNPHARMA.NS",
"BRITANNIA.NS",
"SBIN.NS",
"POWERGRID.NS",
"TATAMOTORS.NS",
"EICHERMOT.NS",
"BAJAJFINSV.NS",
"DIVISLAB.NS",
"AXISBANK.NS",
"TATACONSUM.NS",  
"NTPC.NS",
"GRASIM.NS",
"M&M.NS",
"CIPLA.NS",
"HDFCBANK.NS",
"ULTRACEMCO.NS",
"ITC.NS",
"COALINDIA.NS",
"ONGC.NS",
"HDFC.NS",
"TECHM.NS",
"BAJFINANCE.NS",
"KOTAKBANK.NS",
"ADANIPORTS.NS",
"ADANIENT.NS",
"SBILIFE.NS",
"ICICIBANK.NS",
"LT.NS",
"HEROMOTOCO.NS",
"MARUTI.NS",
"APOLLOHOSP.NS",
"DRREDDY.NS",
"RELIANCE.NS",
"UPL.NS",
"WIPRO.NS",
"TCS.NS",
"HCLTECH.NS",
"INFY.NS",
"BHARTIARTL.NS",
"HDFCLIFE.NS",
"JSWSTEEL.NS",
"INDUSINDBK.NS",
"TATASTEEL.NS",
"HINDALCO.NS",


)
selected_stocks = st.selectbox("Select Dataset for Prediction", stocks) 

START = "2020-01-01"
TODAY= date.today().strftime("%Y-%m-%d")

n_years= st.slider("Years of Prediction:", 1 , 4)
period = n_years*365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state =st.text("Load data...")
data =load_data(selected_stocks)
data_load_state.text("Loading data...done!")

st.markdown('<h2 class="subheader">Raw data</h2>', unsafe_allow_html=True)
st.write(data.tail())

def plot_raw_data():
    fig =go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'],name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()   

#forcasting
df_train = data[['Date','Close']] 
df_train = df_train.rename(columns={'Date':'ds','Close':'y' } )

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast =m.predict(future)

st.markdown('<h2 class="subheader">Forecast data</h2>', unsafe_allow_html=True)
st.write(forecast.tail())

st.write('forecast data')
fig1 = plot_plotly(m,forecast)
st.plotly_chart(fig1)

st.write('forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)

st.markdown('</div>', unsafe_allow_html=True)
