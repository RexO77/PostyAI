# PostyAI - AI-Powered Post Generator 🚀

Transform your ideas into engaging LinkedIn posts with the power of AI! PostyAI combines advanced language models with machine learning to generate professional, high-quality content tailored to your specific needs.

## ✨ Features

### 🎯 **Core Generation Features**
- **AI-Powered Content**: Uses Groq's DeepSeek model for high-quality, contextual post generation
- **Smart Few-Shot Learning**: Learns from 300+ real LinkedIn posts to understand engagement patterns
- **10 Topic Categories**: Content Creation, Career Advice, Productivity, Leadership, UX Design, Technology, Wellness, Networking, Personal Branding, and Industry Insights
- **Multiple Languages**: Generate posts in English and Hinglish
- **Flexible Tones**: Professional, Casual, Inspirational, Educational, and Conversational
- **Smart Length Control**: 
  - Short (1-4 lines) for quick insights
  - Medium (5-10 lines) for detailed thoughts  
  - Long (11+ lines) for comprehensive content

### 🔥 **Advanced Features**
- **Batch Generation**: Create up to 5 posts at once for content planning
- **ML Engagement Prediction**: Machine learning models predict post engagement with confidence scores
- **Real-time Analytics**: Instant analysis of word count, reading time, and engagement potential
- **Post History**: Track all your generated content with session storage
- **Multiple Export Formats**: Download posts as TXT or JSON files
- **One-Click Actions**: Copy to clipboard, regenerate, and export with notifications
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices

### 🧠 **Machine Learning & Analytics**
- **Engagement Prediction Models**: Multiple ML algorithms (Random Forest, XGBoost, LightGBM, etc.)
- **Advanced Feature Engineering**: 60+ text analysis features including sentiment, complexity, and engagement indicators
- **Confidence Scoring**: Get prediction confidence levels to guide content decisions
- **Real-time Performance Monitoring**: Live analytics dashboard with system metrics
- **A/B Testing Framework**: Test different generation strategies and track performance
- **Trending Analysis**: Discover popular topics and content patterns

### ⚡ **Performance & User Experience**
- **Advanced Caching**: Redis-powered caching for 3x faster response times
- **Rate Limiting**: Smart API protection with user-friendly limits
- **Beautiful UI**: Modern design with smooth animations and gradients
- **Dark/Light Mode**: Elegant theme switching for comfortable use
- **Keyboard Shortcuts**: 
  - `Ctrl+Enter`: Generate post
  - `Ctrl+R`: Regenerate last post
  - `Ctrl+C`: Copy to clipboard
- **Real-time Notifications**: Instant feedback for all actions
- **Error Handling**: User-friendly error messages with actionable suggestions

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ (tested with Python 3.13)
- Free Groq API key from [console.groq.com](https://console.groq.com)

### 1. Clone & Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/PostyAI.git
cd PostyAI

# Create virtual environment
python3 -m venv ml_env
source ml_env/bin/activate  # On Windows: ml_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file in the project root:
```env
# Get your free API key from https://console.groq.com
GROQ_API_KEY=your_groq_api_key_here

# Optional Flask configuration
FLASK_ENV=development
FLASK_DEBUG=True
```

### 3. Run the Application
```bash
python app.py
```

Access the application at: **http://localhost:8000**

## 📊 API Endpoints

### Core Routes
- `GET /` - Main application interface
- `GET /api/health` - System health check
- `GET /api/tags` - Get available topic categories

### Generation
- `POST /api/generate` - Generate single post
- `POST /api/batch-generate` - Generate multiple posts
- `POST /api/regenerate` - Regenerate with new parameters

### Analytics & ML
- `POST /api/ml/predict-engagement` - Get ML engagement predictions
- `GET /api/ml/model-info` - View model performance metrics
- `POST /api/ml/analyze-text` - Detailed text analysis
- `GET /api/analytics/live` - Real-time system analytics
- `GET /api/analytics/trending` - Trending topics analysis

### Utilities
- `GET /api/history` - Get post generation history
- `POST /api/export` - Export posts in various formats
- `GET /api/ab-testing/tests` - A/B testing framework

### Example API Usage
```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "length": "Medium",
    "language": "English", 
    "tag": "Career",
    "tone": "Professional"
  }'
```

## 🛠️ Project Structure

```
PostyAI/
├── app.py                          # Main Flask application with all API routes
├── few_shot.py                     # Few-shot learning system for context
├── post_generator.py               # Core post generation logic
├── llm_helper.py                   # LLM integration utilities
├── requirements.txt                # Python dependencies
├── .env                           # Environment variables (create this)
├── data/
│   └── processed_posts.json       # Training data (300+ LinkedIn posts)
├── models/                        # ML models and metadata
│   ├── engagement_predictor_*.pkl  # Various ML models
│   └── engagement_predictor_metadata.json
├── static/
│   ├── style.css                  # All CSS styles and themes
│   ├── app.js                     # Frontend JavaScript logic
│   └── images/
│       └── favicon.svg            # Animated favicon
├── templates/
│   ├── base.html                  # Base template for error pages
│   ├── index.html                 # Main application interface
│   ├── 404.html                   # Custom 404 page
│   └── 500.html                   # Custom 500 page
└── src/
    ├── ml/                        # Machine Learning components
    │   ├── engagement_predictor.py
    │   ├── feature_engineering.py
    │   └── train_models.py
    └── utils/                     # Utility modules
        ├── post_analytics.py      # Post analysis functions
        ├── real_time_analytics.py # Live analytics engine
        ├── ab_testing.py         # A/B testing framework
        └── caching.py            # Redis caching system
```

## 🧪 ML Models & Performance

### Available Models
- **Random Forest** (Primary): Best overall performance with 70%+ accuracy
- **XGBoost**: Gradient boosting for complex patterns
- **LightGBM**: Fast gradient boosting variant
- **Gradient Boosting**: Traditional ensemble method
- **Ridge/Lasso/Elastic Net**: Regularized linear models
- **Support Vector Regression**: Non-linear pattern recognition

### Feature Engineering
Our ML pipeline extracts 60+ features from text:
- **Engagement Features**: Call-to-action detection, question count, curiosity indicators
- **Sentiment Analysis**: Emotional tone and polarity
- **Complexity Metrics**: Reading difficulty, cognitive load
- **Format Features**: Emoji usage, hashtag patterns, line structure
- **Semantic Features**: Vocabulary diversity, expertise indicators
- **Industry Signals**: Domain-specific terminology detection

### Model Performance
- **Primary Model R² Score**: 70%+ (Random Forest)
- **Confidence Prediction**: Available for all models
- **Feature Importance**: Transparency in prediction factors
- **Fallback System**: Heuristic-based predictions if ML fails

## 🔧 Configuration

### Environment Variables
| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `GROQ_API_KEY` | Your Groq API key | ✅ Yes | None |
| `FLASK_ENV` | Flask environment | ❌ No | `development` |
| `FLASK_DEBUG` | Enable debug mode | ❌ No | `True` |

### API Rate Limits
- **Single Generation**: 5 requests per minute
- **Batch Generation**: 2 requests per 5 minutes  
- **Regeneration**: 3 requests per minute
- **ML Predictions**: 10 requests per minute

## 🐳 Deployment

### Local Development
```bash
python app.py  # Runs on http://localhost:8000
```

### Production with Gunicorn
```bash
pip install gunicorn
gunicorn --bind 0.0.0.0:8000 --workers 4 app:app
```

### Docker Deployment
```bash
# Build image
docker build -t postyai .

# Run container
docker run -p 8000:8000 --env-file .env postyai
```

### Production Environment Variables
```env
GROQ_API_KEY=your_production_api_key
FLASK_ENV=production
FLASK_DEBUG=False
```

## 🔍 Troubleshooting

### Common Issues & Solutions

**1. Import or Module Errors**
```bash
# Ensure virtual environment is activated
source ml_env/bin/activate
pip install -r requirements.txt
```

**2. Missing API Key**
```bash
# Check .env file exists and has correct format
cat .env
# Should show: GROQ_API_KEY=your_key_here
```

**3. Port Already in Use**
```bash
# Kill existing processes on port 8000
lsof -ti:8000 | xargs kill -9
```

**4. ML Model Loading Issues**
- Models will fallback to heuristic predictions if loading fails
- Check `models/` directory contains .pkl files and metadata.json
- Verify Python version compatibility (tested with 3.13)

**5. Performance Issues**
- Enable Redis caching for production: `pip install redis`
- Check available memory for ML models (requires ~200MB)
- Consider using fewer models if memory is limited

### Debug Mode
Run with detailed logging:
```bash
FLASK_DEBUG=True python app.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Test thoroughly with various inputs
5. Commit changes: `git commit -am 'Add feature'`
6. Push to branch: `git push origin feature-name`
7. Submit a pull request

### Development Guidelines
- Follow Python PEP 8 style guidelines
- Add comprehensive error handling
- Test both API endpoints and UI functionality
- Update documentation for new features
- Ensure ML models have fallback mechanisms

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Groq**: For providing fast and reliable AI inference
- **DeepSeek**: For the powerful language model capabilities
- **Flask**: For the lightweight and flexible web framework
- **Scikit-learn**: For comprehensive machine learning tools
- **The LinkedIn Community**: For inspiring the training data patterns

## 🆘 Support

If you encounter issues or have questions:

1. Check the [Troubleshooting](#-troubleshooting) section above
2. Review the console/terminal output for error details
3. Verify your `.env` file is properly configured
4. Test with a simple generation request first

For bug reports, please include:
- Python version (`python --version`)
- Error messages or stack traces
- Steps to reproduce the issue
- Your operating system

---

**Ready to revolutionize your LinkedIn content strategy? Get your free Groq API key and start creating engaging posts that drive real engagement! 🎉**

*Made with ❤️ for content creators, marketers, and professionals who want to leverage AI for better social media presence.*