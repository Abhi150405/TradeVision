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
# Currency Conversion Setup
# -----------------------------
USD_TO_INR_RATE = 83.0  # Approximate USD to INR rate (can be updated with real-time data)

def convert_to_inr(usd_amount):
    """Convert USD amount to INR"""
    return float(usd_amount) * USD_TO_INR_RATE

def format_currency(amount, currency="INR"):
    """Format amount with currency symbol"""
    if currency == "INR":
        return f"₹{amount:,.2f}"
    else:
        return f"${amount:,.2f}"

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
        # Try different model names in order of preference
        model_names = [
            "gemini-2.5-flash-lite",
            "gemini-1.5-pro",
            "gemini-1.5-flash", 
            "gemini-pro",
            "models/gemini-2.5-flash-lite",
            "models/gemini-1.5-pro",
            "models/gemini-1.5-flash",
            "models/gemini-pro"
        ]
        
        gemini_model = None
        for model_name in model_names:
            try:
                gemini_model = genai.GenerativeModel(model_name)
                # Test the model with a simple request
                test_response = gemini_model.generate_content("Hello")
                st.success(f"✅ Gemini AI configured successfully with model: {model_name}")
                break
            except Exception as model_error:
                continue
        
        if gemini_model is None:
            st.warning("⚠️ Could not initialize any Gemini model. Please check your API key and model availability.")
            
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

# Currency Configuration
st.sidebar.subheader("💰 Currency Settings")
currency_display = st.sidebar.selectbox("Display Currency:", ["INR (₹)", "USD ($)"], index=0)
if currency_display == "INR (₹)":
    use_inr = True
    currency_symbol = "₹"
    currency_name = "INR"
else:
    use_inr = False
    currency_symbol = "$"
    currency_name = "USD"

# Display conversion rate
if use_inr:
    st.sidebar.info(f"💱 USD to INR Rate: ₹{USD_TO_INR_RATE:.2f}")
else:
    st.sidebar.info(f"💱 USD to INR Rate: ₹{USD_TO_INR_RATE:.2f}")

# Debug section for Gemini API
if st.sidebar.checkbox("🔧 Debug Gemini API"):
    st.sidebar.subheader("Gemini API Debug Info")
    if GEMINI_API_KEY:
        st.sidebar.success("✅ API Key Found")
        try:
            models = genai.list_models()
            available_models = [model.name for model in models if 'generateContent' in model.supported_generation_methods]
            st.sidebar.write("Available models:")
            for model in available_models[:3]:
                st.sidebar.write(f"- {model}")
        except Exception as e:
            st.sidebar.error(f"❌ Error listing models: {e}")
    else:
        st.sidebar.error("❌ No API Key Found")

# Sentiment Analysis Configuration
st.sidebar.subheader("🧠 Sentiment Analysis")
enable_sentiment_analysis = st.sidebar.checkbox("Enable Sentiment Analysis", value=True)
max_news_articles = st.sidebar.slider("Max News Articles:", 3, 10, 5)
sentiment_threshold = st.sidebar.slider("Sentiment Threshold:", 0.0, 0.5, 0.05, 0.01)

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

# Prepare data with enhanced error handling
try:
    scaler = MinMaxScaler()
    stock_data['Scaled_Close'] = scaler.fit_transform(stock_data[['Close']])
    
    # Create sequences
    dataset = stock_data['Scaled_Close'].values.reshape(-1, 1)
    
    # Validate data
    if len(dataset) < time_step + 10:
        st.error(f"❌ Insufficient data for analysis. Need at least {time_step + 10} data points, got {len(dataset)}")
        st.stop()
    
    X, Y = create_sequences(dataset, time_step)
    
    # Validate sequences
    if len(X) == 0 or len(Y) == 0:
        st.error("❌ Could not create sequences from data. Please check your data quality.")
        st.stop()
    
    # Split data with validation set
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)
    
    # Ensure we have enough data for splits
    if train_size < 1 or val_size < 1:
        st.error("❌ Insufficient data for train/validation split")
        st.stop()
    
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
    
    # Additional data validation
    st.info("📊 Data Summary:")
    st.write(f"- Total data points: {len(dataset)}")
    st.write(f"- Sequence length: {time_step}")
    st.write(f"- Generated sequences: {len(X)}")
    
    # Fix the price range formatting with currency conversion
    min_price_usd = float(stock_data['Close'].min())
    max_price_usd = float(stock_data['Close'].max())
    
    if use_inr:
        min_price_display = convert_to_inr(min_price_usd)
        max_price_display = convert_to_inr(max_price_usd)
        st.write(f"- Price range: ₹{min_price_display:,.2f} - ₹{max_price_display:,.2f}")
    else:
        st.write(f"- Price range: ${min_price_usd:.2f} - ${max_price_usd:.2f}")

except Exception as e:
    st.error(f"❌ Error in data preprocessing: {e}")
    st.error("🔍 Debugging information:")
    st.write(f"- Stock data shape: {stock_data.shape if 'stock_data' in locals() else 'Not available'}")
    st.write(f"- Time step: {time_step}")
    st.write(f"- Dataset length: {len(dataset) if 'dataset' in locals() else 'Not available'}")
    st.stop()

# -----------------------------
# Enhanced LSTM Model Setup
# -----------------------------
model_path = f"{ticker}_lstm_model.h5"

def clear_corrupted_model(model_path):
    """Remove corrupted model files"""
    try:
        if os.path.exists(model_path):
            os.remove(model_path)
            st.info(f"🗑️ Removed potentially corrupted model: {model_path}")
    except Exception as e:
        st.warning(f"⚠️ Could not remove model file: {e}")

def validate_model(model):
    """Validate that the model is properly loaded"""
    try:
        # Test model with dummy data
        dummy_input = np.random.random((1, time_step, 1))
        prediction = model.predict(dummy_input, verbose=0)
        return True
    except Exception as e:
        st.warning(f"⚠️ Model validation failed: {e}")
        return False

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

# Model training with enhanced error handling
model = None
history = None

try:
    if os.path.exists(model_path):
        with st.spinner("🔄 Loading pre-trained model..."):
            try:
                model = load_model(model_path)
                
                # Validate the loaded model
                if validate_model(model):
                    st.success("✅ Pre-trained model loaded and validated successfully!")
                else:
                    st.warning("⚠️ Model loaded but validation failed. Building new model...")
                    clear_corrupted_model(model_path)
                    model = None
                    
            except Exception as load_error:
                st.warning(f"⚠️ Could not load pre-trained model: {load_error}")
                st.info("🔄 Building new model instead...")
                clear_corrupted_model(model_path)
                model = None
    
    if model is None:
        with st.spinner("🤖 Training enhanced LSTM model... This may take a few minutes."):
            try:
                # Check data shapes
                st.info(f"📊 Training data shape: {X_train.shape}")
                st.info(f"📊 Validation data shape: {X_val.shape}")
                
                model = build_enhanced_lstm_model()
                
                # Display model summary
                st.info("🏗️ Model Architecture:")
                model_summary = []
                model.summary(print_fn=lambda x: model_summary.append(x))
                st.text('\n'.join(model_summary))
                
                # Callbacks for better training
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True,
                    verbose=1
                )
                
                reduce_lr = ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=0.0001,
                    verbose=1
                )
                
                # Train the model with progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def on_epoch_end(epoch, logs):
                    progress = (epoch + 1) / epochs
                    progress_bar.progress(progress)
                    status_text.text(f'Epoch {epoch + 1}/{epochs} - Loss: {logs.get("loss", 0):.4f}')
                
                # Custom callback for progress
                from tensorflow.keras.callbacks import LambdaCallback
                progress_callback = LambdaCallback(on_epoch_end=on_epoch_end)
                
                # Train the model
                history = model.fit(
                    X_train, Y_train,
                    validation_data=(X_val, Y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopping, reduce_lr, progress_callback],
                    verbose=0
                )
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Save the model
                try:
                    model.save(model_path)
                    st.success("✅ Model trained and saved successfully!")
                except Exception as save_error:
                    st.warning(f"⚠️ Model trained but could not save: {save_error}")
                
                # Plot training history
                if history and 'loss' in history.history:
                    fig_history = go.Figure()
                    fig_history.add_trace(go.Scatter(
                        y=history.history['loss'],
                        mode='lines',
                        name='Training Loss',
                        line=dict(color='blue')
                    ))
                    if 'val_loss' in history.history:
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
                
            except Exception as train_error:
                st.error(f"❌ Error during model training: {train_error}")
                st.error("🔧 Trying to build a simpler model...")
                
                # Fallback to simpler model
                try:
                    model = Sequential([
                        LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
                        LSTM(50, return_sequences=False),
                        Dense(25),
                        Dense(1)
                    ])
                    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                    
                    # Simple training without callbacks
                    history = model.fit(
                        X_train, Y_train,
                        epochs=min(epochs, 10),  # Limit epochs for fallback
                        batch_size=batch_size,
                        verbose=0
                    )
                    
                    st.success("✅ Fallback model trained successfully!")
                    
                except Exception as fallback_error:
                    st.error(f"❌ Fallback model also failed: {fallback_error}")
                    st.stop()

except Exception as e:
    st.error(f"❌ Critical error with model: {e}")
    st.error("🔍 Debugging information:")
    st.write(f"- Model path: {model_path}")
    st.write(f"- X_train shape: {X_train.shape if 'X_train' in locals() else 'Not available'}")
    st.write(f"- Y_train shape: {Y_train.shape if 'Y_train' in locals() else 'Not available'}")
    st.write(f"- Time step: {time_step}")
    
    # Troubleshooting tips
    st.error("🔧 Troubleshooting Tips:")
    st.write("1. **Clear model cache**: Delete any `.h5` files in your directory")
    st.write("2. **Reduce time step**: Try a smaller time step (e.g., 20 instead of 50)")
    st.write("3. **Check data**: Ensure you have enough historical data")
    st.write("4. **Restart app**: Sometimes a fresh start helps")
    st.write("5. **Check TensorFlow**: Ensure TensorFlow is properly installed")
    
    # Add a button to clear models
    if st.button("🗑️ Clear All Model Files"):
        import glob
        model_files = glob.glob("*.h5")
        for file in model_files:
            try:
                os.remove(file)
                st.success(f"Removed {file}")
            except:
                st.warning(f"Could not remove {file}")
        st.info("Please refresh the page and try again.")
    
    st.stop()

# Verify model is loaded
if model is None:
    st.error("❌ Model could not be created. Please check your data and try again.")
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
    st.metric("Mean Absolute Percentage Error", f"{float(mape):.2f}%")
with col6:
    # Calculate directional accuracy
    actual_direction = np.diff(Y_test_original.flatten())
    predicted_direction = np.diff(Y_pred.flatten())
    directional_accuracy = np.mean((actual_direction * predicted_direction) > 0) * 100
    st.metric("Directional Accuracy", f"{float(directional_accuracy):.2f}%")

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
fig_actual_predicted.update_yaxes(title_text=f"Stock Price ({currency_name})", row=1, col=1)
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
current_price_usd = stock_data['Close'].iloc[-1]
future_df['Price_Change'] = future_df['Predicted_Close'] - current_price_usd
future_df['Price_Change_Pct'] = (future_df['Price_Change'] / current_price_usd) * 100

# Convert prices to display currency
if use_inr:
    current_price_display = convert_to_inr(current_price_usd)
    future_df['Predicted_Close_Display'] = future_df['Predicted_Close'].apply(convert_to_inr)
    future_df['Price_Change_Display'] = future_df['Price_Change'].apply(convert_to_inr)
else:
    current_price_display = current_price_usd
    future_df['Predicted_Close_Display'] = future_df['Predicted_Close']
    future_df['Price_Change_Display'] = future_df['Price_Change']

st.subheader(f"🔮 Predicted Stock Prices for Next {future_days} Days")

# Display predictions with enhanced formatting
col1, col2 = st.columns([2, 1])

with col1:
    # Create display dataframe with proper currency formatting
    display_df = future_df[['Date', 'Predicted_Close_Display', 'Price_Change_Display', 'Price_Change_Pct']].copy()
    display_df.columns = ['Date', f'Predicted Price ({currency_name})', f'Price Change ({currency_name})', 'Change %']
    
    # Format currency columns
    if use_inr:
        display_df[f'Predicted Price ({currency_name})'] = display_df[f'Predicted Price ({currency_name})'].apply(lambda x: f"₹{x:,.2f}")
        display_df[f'Price Change ({currency_name})'] = display_df[f'Price Change ({currency_name})'].apply(lambda x: f"₹{x:,.2f}")
    else:
        display_df[f'Predicted Price ({currency_name})'] = display_df[f'Predicted Price ({currency_name})'].apply(lambda x: f"${x:.2f}")
        display_df[f'Price Change ({currency_name})'] = display_df[f'Price Change ({currency_name})'].apply(lambda x: f"${x:.2f}")
    
    display_df['Change %'] = display_df['Change %'].apply(lambda x: f"{x:.2f}%")
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

with col2:
    if use_inr:
        st.metric("Current Price", f"₹{current_price_display:,.2f}")
        st.metric("Predicted Price (7 days)", f"₹{float(future_df['Predicted_Close_Display'].iloc[6]):,.2f}")
    else:
        st.metric("Current Price", f"${current_price_display:.2f}")
        st.metric("Predicted Price (7 days)", f"${float(future_df['Predicted_Close_Display'].iloc[6]):.2f}")
    
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
    fig_technical.update_yaxes(title_text=f"Price ({currency_name})", row=1, col=1)
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
            # Handle Indian stock tickers (.NS suffix)
            if stock_name.endswith('.NS'):
                # Try both with and without .NS suffix
                search_names = [stock_name, stock_name.replace('.NS', '')]
            else:
                search_names = [stock_name]
            
            news_articles = []
            for search_name in search_names:
                try:
                    news_articles = news.get_yf_rss(search_name)
                    if news_articles:
                        break
                except:
                    continue
            
            return news_articles[:max_articles] if news_articles else []
    except Exception as e:
        st.warning(f"Could not fetch news for {stock_name}: {e}")
        return []

# Initialize variables
stock_news_articles = []
news_text_combined = ""

if enable_sentiment_analysis:
    stock_news_articles = fetch_stock_news(ticker, max_news_articles)
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
# Advanced Sentiment Analysis Functions
# -----------------------------
def analyze_individual_articles(news_articles):
    """Analyze sentiment for each individual article"""
    article_sentiments = []
    for article in news_articles:
        article_text = article['title']
        if 'summary' in article:
            article_text += " " + article['summary']
        
        sentiment = sia.polarity_scores(article_text)
        article_sentiments.append({
            'title': article['title'][:50] + "...",
            'sentiment': sentiment,
            'overall': 'Positive' if sentiment['compound'] > 0.05 else 'Negative' if sentiment['compound'] < -0.05 else 'Neutral'
        })
    return article_sentiments

def calculate_sentiment_trends(article_sentiments):
    """Calculate sentiment trends and patterns"""
    positive_count = sum(1 for art in article_sentiments if art['overall'] == 'Positive')
    negative_count = sum(1 for art in article_sentiments if art['overall'] == 'Negative')
    neutral_count = sum(1 for art in article_sentiments if art['overall'] == 'Neutral')
    
    total_articles = len(article_sentiments)
    
    return {
        'positive_pct': float((positive_count / total_articles) * 100) if total_articles > 0 else 0.0,
        'negative_pct': float((negative_count / total_articles) * 100) if total_articles > 0 else 0.0,
        'neutral_pct': float((neutral_count / total_articles) * 100) if total_articles > 0 else 0.0,
        'sentiment_ratio': float(positive_count / negative_count) if negative_count > 0 else float(positive_count),
        'sentiment_strength': float(abs(sum(art['sentiment']['compound'] for art in article_sentiments)) / total_articles) if total_articles > 0 else 0.0
    }

def get_sentiment_recommendation(sentiment_data, trends, threshold=0.05):
    """Generate investment recommendation based on sentiment"""
    compound = sentiment_data['compound']
    strength = trends['sentiment_strength']
    ratio = trends['sentiment_ratio']
    
    if compound > threshold * 2 and ratio > 2:
        return "🟢 Strong Buy Signal", "Very positive sentiment with strong bullish momentum"
    elif compound > threshold and ratio > 1.5:
        return "🟡 Buy Signal", "Positive sentiment with moderate bullish momentum"
    elif compound > -threshold and compound < threshold:
        return "⚪ Hold", "Neutral sentiment, wait for clearer signals"
    elif compound < -threshold and ratio < 0.67:
        return "🔴 Sell Signal", "Negative sentiment with bearish momentum"
    else:
        return "🟠 Caution", "Mixed signals, proceed with caution"

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
            # Enhanced error handling with model debugging
            error_msg = f"Error fetching insights from Gemini AI: {e}"
            
            # Try to get available models for debugging
            try:
                models = genai.list_models()
                available_models = [model.name for model in models if 'generateContent' in model.supported_generation_methods]
                error_msg += f"\n\nAvailable models: {', '.join(available_models[:5])}"
            except:
                pass
                
            return error_msg
    else:
        return "Gemini AI not available. Please configure your API key for AI insights."

# -----------------------------
# Comprehensive Sentiment Analysis & AI Insights
# -----------------------------
if enable_sentiment_analysis and news_text_combined and stock_news_articles:
    # Individual article analysis
    article_sentiments = analyze_individual_articles(stock_news_articles)
    sentiment_trends = calculate_sentiment_trends(article_sentiments)
    
    # Overall sentiment analysis
    overall_sentiment = sia.polarity_scores(news_text_combined)
    insights = get_gemini_insights(news_text_combined)
    recommendation, recommendation_reason = get_sentiment_recommendation(overall_sentiment, sentiment_trends, sentiment_threshold)
    
    st.subheader(f"🧠 Advanced Sentiment Analysis & AI Insights for {ticker}")
    
    # Recommendation Section
    st.markdown("### 🎯 Sentiment-Based Investment Recommendation")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"**Recommendation:** {recommendation}")
    with col2:
        st.markdown(f"**Reasoning:** {recommendation_reason}")
    
    # AI Analysis
    st.markdown("### 🤖 AI Market Analysis")
    st.write(insights)
    
    # Comprehensive Sentiment Dashboard
    st.markdown("### 📊 Comprehensive Sentiment Dashboard")
    
    # Main sentiment metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Overall Sentiment Score", f"{overall_sentiment['compound']:.3f}")
    with col2:
        st.metric("Sentiment Strength", f"{sentiment_trends['sentiment_strength']:.3f}")
    with col3:
        st.metric("Bullish Ratio", f"{float(sentiment_trends['sentiment_ratio']):.2f}")
    with col4:
        st.metric("Articles Analyzed", len(stock_news_articles))
    
    # Detailed sentiment visualization
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Enhanced sentiment pie chart
        sentiment_labels = ["Positive", "Neutral", "Negative"]
        sentiment_values = [overall_sentiment["pos"], overall_sentiment["neu"], overall_sentiment["neg"]]
        colors = ['#2E8B57', '#FFA500', '#FF6B6B']
        
        fig_pie = px.pie(
            values=sentiment_values, 
            names=sentiment_labels, 
            title="Overall Sentiment Distribution",
            color_discrete_sequence=colors
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Article-level sentiment breakdown
        article_labels = [art['overall'] for art in article_sentiments]
        article_counts = {
            'Positive': article_labels.count('Positive'),
            'Neutral': article_labels.count('Neutral'),
            'Negative': article_labels.count('Negative')
        }
        
        fig_bar = px.bar(
            x=list(article_counts.keys()),
            y=list(article_counts.values()),
            title="Sentiment by Article Count",
            color=list(article_counts.keys()),
            color_discrete_map={'Positive': '#2E8B57', 'Neutral': '#FFA500', 'Negative': '#FF6B6B'}
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Individual article sentiment analysis
    st.markdown("### 📰 Individual Article Sentiment Analysis")
    
    # Create a detailed table
    sentiment_df = pd.DataFrame([
        {
            'Article': art['title'],
            'Sentiment': art['overall'],
            'Positive': f"{art['sentiment']['pos']:.3f}",
            'Neutral': f"{art['sentiment']['neu']:.3f}",
            'Negative': f"{art['sentiment']['neg']:.3f}",
            'Compound': f"{art['sentiment']['compound']:.3f}"
        }
        for art in article_sentiments
    ])
    
    st.dataframe(sentiment_df, use_container_width=True, hide_index=True)
    
    # Sentiment trends over time (if we had timestamps)
    st.markdown("### 📈 Sentiment Analysis Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Sentiment Breakdown:**")
        st.write(f"• **Positive Articles:** {sentiment_trends['positive_pct']:.1f}%")
        st.write(f"• **Neutral Articles:** {sentiment_trends['neutral_pct']:.1f}%")
        st.write(f"• **Negative Articles:** {sentiment_trends['negative_pct']:.1f}%")
    
    with col2:
        st.markdown("**Market Implications:**")
        if overall_sentiment['compound'] > 0.1:
            st.write("• 🚀 **Strong bullish momentum**")
            st.write("• 📈 **Potential price increase**")
        elif overall_sentiment['compound'] < -0.1:
            st.write("• 📉 **Bearish sentiment**")
            st.write("• ⚠️ **Potential price decline**")
        else:
            st.write("• ⚖️ **Mixed market signals**")
            st.write("• 🔍 **Monitor for clearer trends**")
    
    # Sentiment strength indicator
    st.markdown("### 💪 Sentiment Strength Analysis")
    
    strength_level = "Very Strong" if sentiment_trends['sentiment_strength'] > 0.3 else "Strong" if sentiment_trends['sentiment_strength'] > 0.2 else "Moderate" if sentiment_trends['sentiment_strength'] > 0.1 else "Weak"
    
    st.progress(sentiment_trends['sentiment_strength'], text=f"Sentiment Strength: {strength_level} ({sentiment_trends['sentiment_strength']:.3f})")

elif enable_sentiment_analysis and news_text_combined:
    # Fallback for when we have text but no individual articles
    overall_sentiment = sia.polarity_scores(news_text_combined)
    insights = get_gemini_insights(news_text_combined)
    
    st.subheader(f"🧠 Sentiment Analysis & AI Insights for {ticker}")
    
    # Display insights
    st.markdown("### 🤖 AI Analysis")
    st.write(insights)
    
    # Basic sentiment visualization
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Sentiment pie chart
        sentiment_labels = ["Positive", "Neutral", "Negative"]
        sentiment_values = [overall_sentiment["pos"], overall_sentiment["neu"], overall_sentiment["neg"]]
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
        st.metric("Positive", f"{float(overall_sentiment['pos']):.2f}")
        st.metric("Neutral", f"{float(overall_sentiment['neu']):.2f}")
        st.metric("Negative", f"{float(overall_sentiment['neg']):.2f}")
        
        # Overall sentiment
        compound_score = overall_sentiment['compound']
        if compound_score > 0.05:
            overall_sentiment_text = "😊 Positive"
        elif compound_score < -0.05:
            overall_sentiment_text = "😞 Negative"
        else:
            overall_sentiment_text = "😐 Neutral"
        
        st.metric("Overall Sentiment", overall_sentiment_text, f"{compound_score:.3f}")

elif enable_sentiment_analysis:
    st.subheader(f"🧠 Sentiment Analysis for {ticker}")
    st.info("No news articles found for sentiment analysis. Try a different ticker or check your internet connection.")
else:
    st.subheader(f"🧠 Sentiment Analysis for {ticker}")
    st.info("Sentiment analysis is disabled. Enable it in the sidebar to see sentiment insights.")

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
💰 **Multi-Currency**: Display prices in INR or USD

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
