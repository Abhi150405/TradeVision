# TradeVision Pro - Advanced Stock Analyzer

TradeVision Pro is an advanced Streamlit application that provides comprehensive stock analysis and forecasting using cutting-edge AI technologies. It combines LSTM neural networks, technical analysis indicators, sentiment analysis, and AI-powered insights for sophisticated market analysis.

## 🚀 Features

### 🔮 AI-Powered Predictions
- **Enhanced LSTM Models**: Multi-layer LSTM with dropout and batch normalization
- **Advanced Architecture**: 3-layer LSTM with 100-100-50 neurons
- **Smart Training**: Early stopping and learning rate reduction
- **Model Validation**: Comprehensive metrics including R², MAPE, and directional accuracy

### 📊 Technical Analysis
- **RSI (Relative Strength Index)**: Momentum oscillator with overbought/oversold signals
- **MACD**: Moving Average Convergence Divergence with signal lines
- **Moving Averages**: 20, 50, and 200-day moving averages
- **Interactive Charts**: Multi-panel technical analysis visualizations

### 📰 News & Sentiment Analysis
- **Real-time News**: Latest stock news from Yahoo Finance
- **Sentiment Scoring**: VADER sentiment analysis
- **AI Insights**: Google Gemini-powered market analysis
- **Visual Sentiment**: Interactive pie charts and sentiment metrics

### 📈 Advanced Visualizations
- **Price Predictions**: Actual vs predicted with error analysis
- **Training History**: Model performance over epochs
- **Technical Indicators**: Multi-panel technical analysis charts
- **Future Forecasts**: Confidence intervals and trend analysis

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd stock-predictor
```

### 2. Create Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Configure API Keys (Optional)
For AI insights, add your Gemini API key:

**Option A: Environment Variable**
```powershell
$env:GEMINI_API_KEY = "your-gemini-api-key"
```

**Option B: Streamlit Secrets**
Create `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your-gemini-api-key"
```

### 5. Run the Application
```powershell
streamlit run app.py
```

## 🎯 Usage

### Basic Usage
1. **Enter Stock Ticker**: Input any valid stock symbol (e.g., AAPL, TSLA, GOOG)
2. **Configure Settings**: Adjust prediction days, time steps, and training parameters
3. **View Analysis**: Explore predictions, technical indicators, and sentiment analysis

### Advanced Configuration
- **Time Step**: LSTM window size (10-100 days)
- **Training Epochs**: Model training iterations (5-50)
- **Batch Size**: Training batch size (16, 32, 64)
- **Technical Indicators**: Toggle RSI, MACD, and moving averages

## 📊 Model Performance

The enhanced LSTM model provides:
- **MSE/MAE**: Mean squared and absolute errors
- **R² Score**: Coefficient of determination
- **MAPE**: Mean absolute percentage error
- **Directional Accuracy**: Trend prediction accuracy
- **Training History**: Loss curves and validation metrics

## 🔧 Technical Architecture

### Data Processing
- **2-year Historical Data**: Comprehensive price history
- **MinMax Scaling**: Normalized data for optimal training
- **Sequence Creation**: Time-series sequences for LSTM
- **Train/Validation/Test Split**: 70/15/15 data distribution

### Model Architecture
```
Input Layer: (time_step, 1)
├── LSTM Layer 1: 100 units + Dropout(0.2) + BatchNorm
├── LSTM Layer 2: 100 units + Dropout(0.2) + BatchNorm
├── LSTM Layer 3: 50 units + Dropout(0.2) + BatchNorm
├── Dense Layer: 25 units + ReLU + Dropout(0.1)
└── Output Layer: 1 unit (price prediction)
```

### Technical Indicators
- **RSI**: 14-period relative strength index
- **MACD**: 12,26,9 exponential moving averages
- **Moving Averages**: 20, 50, 200-day simple moving averages

## 🚀 Deployment

### Streamlit Cloud
1. Push repository to GitHub
2. Connect to Streamlit Cloud
3. Add `GEMINI_API_KEY` in secrets
4. Deploy with default settings

### Local Production
```powershell
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## ⚠️ Important Notes

### Performance Considerations
- **First Run**: Model training may take 5-10 minutes
- **Memory Usage**: Requires ~2GB RAM for training
- **GPU Support**: TensorFlow will use GPU if available
- **Caching**: Data is cached for 5 minutes to improve performance

### Model Persistence
- **Pre-trained Models**: Saved as `{TICKER}_lstm_model.h5`
- **Automatic Loading**: Models are reused on subsequent runs
- **Training Skip**: Pre-trained models skip training phase

### API Limitations
- **Gemini API**: Rate limits apply (optional feature)
- **Yahoo Finance**: Free tier with reasonable limits
- **News API**: Subject to Yahoo Finance availability

## 🔒 Security & Privacy

- **API Keys**: Never commit API keys to repository
- **Environment Variables**: Use secure environment variable storage
- **Streamlit Secrets**: Recommended for production deployment
- **Data Privacy**: No personal data is stored or transmitted

## 🐛 Troubleshooting

### Common Issues
1. **TensorFlow Installation**: Use `pip install tensorflow-cpu` for CPU-only systems
2. **Memory Errors**: Reduce batch size or time step for low-memory systems
3. **API Errors**: Check API key configuration and rate limits
4. **Model Training**: Increase patience or reduce epochs for faster training

### Performance Optimization
- **Pre-trained Models**: Upload pre-trained models to avoid training
- **Data Caching**: Leverage Streamlit's caching mechanisms
- **Resource Monitoring**: Monitor CPU/memory usage during training

## 📈 Future Enhancements

- **Multi-timeframe Analysis**: 1-minute to monthly predictions
- **Portfolio Analysis**: Multi-stock portfolio optimization
- **Advanced Indicators**: Bollinger Bands, Stochastic, Williams %R
- **Real-time Updates**: Live data streaming and alerts
- **Backtesting**: Historical strategy performance testing

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

**This application is for educational and research purposes only. It is not intended as financial advice. Always consult with qualified financial advisors before making investment decisions. Past performance does not guarantee future results.**

---

**Powered by**: TensorFlow, Streamlit, Plotly, NLTK, Google Gemini AI, and Yahoo Finance
