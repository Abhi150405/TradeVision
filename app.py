import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import google.generativeai as genai
from yahoo_fin import news
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# NLTK Setup
# -----------------------------
try:
    nltk.download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()
except Exception as e:
    st.error(f"Error initializing NLTK: {e}")
    st.stop()

# -----------------------------
# Configure Gemini API (Safe)
# -----------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("models/gemini-1.5-pro")
    except Exception as e:
        st.warning(f"Gemini API configuration failed: {e}")

# -----------------------------
# Streamlit UI Setup
# -----------------------------
st.set_page_config(
    page_title="TradeVision Pro - Advanced Stock Analyzer", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .stAlert {
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">📈 TradeVision Pro - Advanced Stock Analyzer</h1>', unsafe_allow_html=True)

# Sidebar Inputs
st.sidebar.header("🔧 Configuration")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, GOOG):", "AAPL").upper()
future_days = st.sidebar.slider("Days to Predict:", 1, 30, 7)

# Advanced Configuration
st.sidebar.subheader("⚙️ Advanced Settings")
time_step = st.sidebar.slider("Time Step (LSTM window):", 10, 100, 50)
epochs = st.sidebar.slider("Training Epochs:", 5, 50, 15)
batch_size = st.sidebar.selectbox("Batch Size:", [16, 32, 64], index=0)
show_technical_indicators = st.sidebar.checkbox("Show Technical Indicators", value=True)

# -----------------------------
# Technical Indicators Functions
# -----------------------------
def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    macd_histogram = macd - macd_signal
    return macd, macd_signal, macd_histogram

def calculate_moving_averages(prices):
    """Calculate various moving averages"""
    ma_20 = prices.rolling(window=20).mean()
    ma_50 = prices.rolling(window=50).mean()
    ma_200 = prices.rolling(window=200).mean()
    return ma_20, ma_50, ma_200

# -----------------------------
# Fetch Stock Data (Enhanced)
# -----------------------------
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_stock_data(ticker):
    try:
        with st.spinner(f"📊 Fetching data for {ticker}..."):
            data = yf.download(ticker, period="2y", progress=False)
            if data.empty:
                raise ValueError("No data found for ticker.")
            
            # Add technical indicators
            if show_technical_indicators:
                data['RSI'] = calculate_rsi(data['Close'])
                data['MACD'], data['MACD_Signal'], data['MACD_Histogram'] = calculate_macd(data['Close'])
                data['MA_20'], data['MA_50'], data['MA_200'] = calculate_moving_averages(data['Close'])
            
            return data
    except Exception as e:
        st.error(f"❌ Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

# Main data fetching
stock_data = fetch_stock_data(ticker)
if stock_data.empty:
    st.stop()

# Display success message
st.sidebar.success(f"✅ Data for {ticker} loaded successfully!")
st.sidebar.info(f"📅 Data range: {stock_data.index[0].strftime('%Y-%m-%d')} to {stock_data.index[-1].strftime('%Y-%m-%d')}")

# -----------------------------
# Data Preprocessing (Enhanced)
# -----------------------------
def create_sequences(data, time_step=50):
    """Create sequences for LSTM training"""
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

# Prepare data with better preprocessing
scaler = MinMaxScaler()
stock_data['Scaled_Close'] = scaler.fit_transform(stock_data[['Close']])

# Create sequences
dataset = stock_data['Scaled_Close'].values.reshape(-1, 1)
X, Y = create_sequences(dataset, time_step)

# Split data with validation set
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)

X_train, Y_train = X[:train_size], Y[:train_size]
X_val, Y_val = X[train_size:train_size+val_size], Y[train_size:train_size+val_size]
X_test, Y_test = X[train_size+val_size:], Y[train_size+val_size:]

# Reshape for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Display data info
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Training Samples", len(X_train))
with col2:
    st.metric("Validation Samples", len(X_val))
with col3:
    st.metric("Test Samples", len(X_test))

# -----------------------------
# Enhanced LSTM Model Setup
# -----------------------------
model_path = f"{ticker}_lstm_model.h5"

def build_enhanced_lstm_model():
    """Build an enhanced LSTM model with better architecture"""
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(time_step, 1)),
        Dropout(0.2),
        BatchNormalization(),
        
        LSTM(100, return_sequences=True),
        Dropout(0.2),
        BatchNormalization(),
        
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        BatchNormalization(),
        
        Dense(25, activation='relu'),
        Dropout(0.1),
        Dense(1)
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    return model

# Model training with callbacks
try:
    if os.path.exists(model_path):
        with st.spinner("🔄 Loading pre-trained model..."):
            model = load_model(model_path)
        st.success("✅ Pre-trained model loaded successfully!")
    else:
        with st.spinner("🤖 Training enhanced LSTM model... This may take a few minutes."):
            model = build_enhanced_lstm_model()
            
            # Callbacks for better training
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=0.0001
            )
            
            # Train the model
            history = model.fit(
                X_train, Y_train,
                validation_data=(X_val, Y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            model.save(model_path)
            st.success("✅ Model trained and saved successfully!")
            
            # Plot training history
            if 'history' in locals():
                fig_history = go.Figure()
                fig_history.add_trace(go.Scatter(
                    y=history.history['loss'],
                    mode='lines',
                    name='Training Loss',
                    line=dict(color='blue')
                ))
                fig_history.add_trace(go.Scatter(
                    y=history.history['val_loss'],
                    mode='lines',
                    name='Validation Loss',
                    line=dict(color='red')
                ))
                fig_history.update_layout(
                    title="Model Training History",
                    xaxis_title="Epoch",
                    yaxis_title="Loss",
                    height=400
                )
                st.plotly_chart(fig_history, use_container_width=True)

except Exception as e:
    st.error(f"❌ Error with model: {e}")
    st.stop()

# -----------------------------
# Enhanced Predictions & Metrics
# -----------------------------
try:
    with st.spinner("🔮 Making predictions..."):
        Y_pred = model.predict(X_test)
        Y_pred = scaler.inverse_transform(Y_pred.reshape(-1, 1))
        Y_test_original = scaler.inverse_transform(Y_test.reshape(-1, 1))
        
        # Calculate comprehensive metrics
        mse = mean_squared_error(Y_test_original, Y_pred)
        mae = mean_absolute_error(Y_test_original, Y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(Y_test_original, Y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((Y_test_original - Y_pred) / Y_test_original)) * 100
        
except Exception as e:
    st.error(f"❌ Error during prediction: {e}")
    st.stop()

# -----------------------------
# Enhanced Model Metrics Display
# -----------------------------
st.subheader("📊 Model Performance Metrics")

# Create metrics in columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Mean Squared Error", f"{mse:.4f}")
with col2:
    st.metric("Mean Absolute Error", f"{mae:.4f}")
with col3:
    st.metric("Root Mean Squared Error", f"{rmse:.4f}")
with col4:
    st.metric("R² Score", f"{r2:.4f}")

# Additional metrics
col5, col6 = st.columns(2)
with col5:
    st.metric("Mean Absolute Percentage Error", f"{mape:.2f}%")
with col6:
    # Calculate directional accuracy
    actual_direction = np.diff(Y_test_original.flatten())
    predicted_direction = np.diff(Y_pred.flatten())
    directional_accuracy = np.mean((actual_direction * predicted_direction) > 0) * 100
    st.metric("Directional Accuracy", f"{directional_accuracy:.2f}%")

# -----------------------------
# Enhanced Visualization - Actual vs Predicted
# -----------------------------
st.subheader("📈 Actual vs Predicted Stock Prices")

# Create enhanced plot with subplots
fig_actual_predicted = make_subplots(
    rows=2, cols=1,
    subplot_titles=('Price Comparison', 'Prediction Error'),
    vertical_spacing=0.1,
    row_heights=[0.7, 0.3]
)

# Main price comparison
fig_actual_predicted.add_trace(
    go.Scatter(
        x=stock_data.index[-len(Y_test_original):],
        y=Y_test_original.flatten(),
        mode='lines',
        name='Actual Price',
        line=dict(color='#2E8B57', width=2)
    ),
    row=1, col=1
)

fig_actual_predicted.add_trace(
    go.Scatter(
        x=stock_data.index[-len(Y_test_original):],
        y=Y_pred.flatten(),
        mode='lines',
        name='Predicted Price',
        line=dict(color='#FF6B6B', width=2, dash='dot')
    ),
    row=1, col=1
)

# Prediction error
error = Y_test_original.flatten() - Y_pred.flatten()
fig_actual_predicted.add_trace(
    go.Scatter(
        x=stock_data.index[-len(Y_test_original):],
        y=error,
        mode='lines',
        name='Prediction Error',
        line=dict(color='#FFA500', width=1),
        fill='tonexty'
    ),
    row=2, col=1
)

fig_actual_predicted.update_layout(
    title=f"Actual vs Predicted Prices ({ticker})",
    height=600,
    showlegend=True
)

fig_actual_predicted.update_xaxes(title_text="Date", row=2, col=1)
fig_actual_predicted.update_yaxes(title_text="Stock Price (USD)", row=1, col=1)
fig_actual_predicted.update_yaxes(title_text="Error", row=2, col=1)

st.plotly_chart(fig_actual_predicted, use_container_width=True)

# -----------------------------
# Enhanced Future Price Predictions
# -----------------------------
def predict_future_prices(model, last_sequence, future_days):
    """Enhanced future price prediction with confidence intervals"""
    future_predictions = []
    confidence_intervals = []
    current_input = last_sequence.reshape(1, -1, 1)
    
    for _ in range(future_days):
        # Make prediction
        next_prediction = model.predict(current_input, verbose=0)[0][0]
        future_predictions.append(next_prediction)
        
        # Simple confidence estimation (can be enhanced with Monte Carlo)
        confidence_intervals.append(next_prediction * 0.05)  # 5% uncertainty
        
        # Update input for next prediction
        current_input = np.append(current_input[:, 1:, :], [[[next_prediction]]], axis=1)
    
    return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)), confidence_intervals

# Generate future predictions
last_sequence = dataset[-time_step:]
future_prices, confidence_intervals = predict_future_prices(model, last_sequence, future_days)
future_dates = pd.date_range(stock_data.index[-1] + pd.Timedelta(days=1), periods=future_days)

# Create future predictions dataframe
future_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted_Close': future_prices.flatten(),
    'Confidence_Interval': confidence_intervals
})

# Calculate price change
current_price = stock_data['Close'].iloc[-1]
future_df['Price_Change'] = future_df['Predicted_Close'] - current_price
future_df['Price_Change_Pct'] = (future_df['Price_Change'] / current_price) * 100

st.subheader(f"🔮 Predicted Stock Prices for Next {future_days} Days")

# Display predictions with enhanced formatting
col1, col2 = st.columns([2, 1])

with col1:
    st.dataframe(
        future_df.round(2),
        use_container_width=True,
        hide_index=True
    )

with col2:
    st.metric("Current Price", f"${current_price:.2f}")
    st.metric("Predicted Price (7 days)", f"${future_df['Predicted_Close'].iloc[6]:.2f}")
    
    # Overall trend
    trend = "📈 Bullish" if future_df['Price_Change'].iloc[-1] > 0 else "📉 Bearish"
    st.metric("Overall Trend", trend)

# -----------------------------
# Technical Indicators Visualization
# -----------------------------
if show_technical_indicators and 'RSI' in stock_data.columns:
    st.subheader("📊 Technical Analysis Indicators")
    
    # Create subplots for technical indicators
    fig_technical = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Price with Moving Averages', 'RSI (Relative Strength Index)', 'MACD'),
        vertical_spacing=0.08,
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # Price with moving averages
    fig_technical.add_trace(
        go.Scatter(x=stock_data.index, y=stock_data['Close'], name='Close Price', line=dict(color='black')),
        row=1, col=1
    )
    
    if 'MA_20' in stock_data.columns:
        fig_technical.add_trace(
            go.Scatter(x=stock_data.index, y=stock_data['MA_20'], name='MA 20', line=dict(color='blue')),
            row=1, col=1
        )
    
    if 'MA_50' in stock_data.columns:
        fig_technical.add_trace(
            go.Scatter(x=stock_data.index, y=stock_data['MA_50'], name='MA 50', line=dict(color='orange')),
            row=1, col=1
        )
    
    if 'MA_200' in stock_data.columns:
        fig_technical.add_trace(
            go.Scatter(x=stock_data.index, y=stock_data['MA_200'], name='MA 200', line=dict(color='red')),
            row=1, col=1
        )
    
    # RSI
    fig_technical.add_trace(
        go.Scatter(x=stock_data.index, y=stock_data['RSI'], name='RSI', line=dict(color='purple')),
        row=2, col=1
    )
    
    # RSI overbought/oversold lines
    fig_technical.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig_technical.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    if 'MACD' in stock_data.columns:
        fig_technical.add_trace(
            go.Scatter(x=stock_data.index, y=stock_data['MACD'], name='MACD', line=dict(color='blue')),
            row=3, col=1
        )
        
        fig_technical.add_trace(
            go.Scatter(x=stock_data.index, y=stock_data['MACD_Signal'], name='Signal', line=dict(color='red')),
            row=3, col=1
        )
        
        # MACD Histogram
        colors = ['green' if val >= 0 else 'red' for val in stock_data['MACD_Histogram']]
        fig_technical.add_trace(
            go.Bar(x=stock_data.index, y=stock_data['MACD_Histogram'], name='Histogram', marker_color=colors),
            row=3, col=1
        )
    
    fig_technical.update_layout(height=800, showlegend=True)
    fig_technical.update_yaxes(title_text="Price", row=1, col=1)
    fig_technical.update_yaxes(title_text="RSI", row=2, col=1)
    fig_technical.update_yaxes(title_text="MACD", row=3, col=1)
    
    st.plotly_chart(fig_technical, use_container_width=True)
    
    # Technical indicators summary
    col1, col2, col3 = st.columns(3)
    with col1:
        current_rsi = stock_data['RSI'].iloc[-1]
        rsi_signal = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
        st.metric("RSI", f"{current_rsi:.1f}", rsi_signal)
    
    with col2:
        if 'MACD' in stock_data.columns:
            macd_trend = "Bullish" if stock_data['MACD'].iloc[-1] > stock_data['MACD_Signal'].iloc[-1] else "Bearish"
            st.metric("MACD Signal", macd_trend)
    
    with col3:
        if 'MA_20' in stock_data.columns and 'MA_50' in stock_data.columns:
            ma_trend = "Bullish" if stock_data['MA_20'].iloc[-1] > stock_data['MA_50'].iloc[-1] else "Bearish"
            st.metric("MA Trend", ma_trend)

def fetch_stock_news(stock_name, max_articles=5):
    """Enhanced news fetching with better error handling"""
    try:
        with st.spinner("📰 Fetching latest news..."):
            news_articles = news.get_yf_rss(stock_name)
            return news_articles[:max_articles]
    except Exception as e:
        st.warning(f"Could not fetch news: {e}")
        return []

stock_news_articles = fetch_stock_news(ticker)
st.subheader(f"📰 Latest News for {ticker}")

if not stock_news_articles:
    st.info("No news articles found for this ticker.")
else:
    news_text_combined = ""
    
    # Display news articles in a more organized way
    for i, article in enumerate(stock_news_articles, 1):
        with st.expander(f"📰 Article {i}: {article['title'][:80]}..."):
            st.write(f"**Title:** {article['title']}")
            st.write(f"**Link:** {article['link']}")
            if 'summary' in article:
                st.write(f"**Summary:** {article['summary']}")
        
        news_text_combined += article['title'] + " "

# -----------------------------
# Enhanced Sentiment Analysis & AI Insights
# -----------------------------
def get_gemini_insights(news_text):
    """Enhanced Gemini AI insights with better prompting"""
    if not news_text.strip():
        return "No news available for analysis."
    
    prompt = f"""
    Analyze the following stock news for {ticker} and provide:
    1. Key market sentiment trends
    2. Potential impact on stock price
    3. Risk factors to consider
    4. Investment outlook (short-term)
    
    News headlines: {news_text}
    
    Please provide a concise analysis in 3-4 bullet points.
    """
    
    if gemini_model:
        try:
            response = gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error fetching insights from Gemini AI: {e}"
    else:
        return "Gemini AI not available. Please configure your API key for AI insights."

# Perform sentiment analysis
if news_text_combined:
    sentiment = sia.polarity_scores(news_text_combined)
    insights = get_gemini_insights(news_text_combined)
    
    st.subheader(f"🧠 Sentiment Analysis & AI Insights for {ticker}")
    
    # Display insights
    st.markdown("### 🤖 AI Analysis")
    st.write(insights)
    
    # Enhanced sentiment visualization
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Sentiment pie chart
        sentiment_labels = ["Positive", "Neutral", "Negative"]
        sentiment_values = [sentiment["pos"], sentiment["neu"], sentiment["neg"]]
        colors = ['#2E8B57', '#FFA500', '#FF6B6B']
        
        fig_pie = px.pie(
            values=sentiment_values, 
            names=sentiment_labels, 
            title="Sentiment Distribution",
            color_discrete_sequence=colors
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Sentiment scores
        st.markdown("### 📊 Sentiment Scores")
        st.metric("Positive", f"{sentiment['pos']:.2f}")
        st.metric("Neutral", f"{sentiment['neu']:.2f}")
        st.metric("Negative", f"{sentiment['neg']:.2f}")
        
        # Overall sentiment
        compound_score = sentiment['compound']
        if compound_score > 0.05:
            overall_sentiment = "😊 Positive"
        elif compound_score < -0.05:
            overall_sentiment = "😞 Negative"
        else:
            overall_sentiment = "😐 Neutral"
        
        st.metric("Overall Sentiment", overall_sentiment, f"{compound_score:.3f}")

# -----------------------------
# Enhanced Sidebar Information
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 About TradeVision Pro")
st.sidebar.info("""
**TradeVision Pro** provides advanced stock analysis with:

🔮 **AI-Powered Predictions**: Enhanced LSTM models with technical indicators
📊 **Technical Analysis**: RSI, MACD, Moving Averages
📰 **News Sentiment**: Real-time sentiment analysis
🤖 **AI Insights**: Gemini-powered market analysis
📈 **Interactive Charts**: Advanced visualizations

**Disclaimer**: This tool is for educational purposes only. Not financial advice.
""")

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🚀 <strong>TradeVision Pro</strong> - Advanced Stock Analysis Platform</p>
    <p>Powered by TensorFlow, Plotly, NLTK, and Google Gemini AI</p>
</div>
""", unsafe_allow_html=True)
