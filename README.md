# PostyAI - AI-Powered LinkedIn Post Generator 🚀

Transform your ideas into engaging LinkedIn posts with the power of AI! PostyAI uses advanced language models to generate professional, engaging content tailored to your needs.

## ✨ Features

### 🎯 **Core Features**
- **AI-Powered Generation**: Uses Groq's DeepSeek model for high-quality content
- **Multi-Language Support**: Generate posts in English and Hinglish
- **Various Tones**: Professional, Casual, Inspirational, and more
- **Smart Length Control**: Short (1-5 lines), Medium (6-10 lines), Long (14-18 lines)
- **Topic Categories**: UX Design, Career Advice, Productivity, Leadership, and more

### 🔥 **Advanced Features**
- **Batch Generation**: Generate up to 5 posts at once
- **Post History**: Track your generated content with session storage
- **Export Options**: Download posts as TXT or JSON files
- **Real-time Analytics**: Word count, reading time, engagement score
- **Copy to Clipboard**: One-click copying with notifications
- **Dark/Light Mode**: Beautiful theme switching
- **Rate Limiting**: API protection with smart throttling
- **Responsive Design**: Works perfectly on all devices

### ⚡ **User Experience**
- **Keyboard Shortcuts**: 
  - `Ctrl+Enter`: Generate post
  - `Ctrl+R`: Regenerate last post
  - `Ctrl+C`: Copy to clipboard
- **Real-time Notifications**: Success and error feedback
- **Beautiful UI**: Modern gradients and animations
- **Professional Error Pages**: Custom 404 and 500 pages

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher (tested with Python 3.13)
- Groq API key (free at [console.groq.com](https://console.groq.com))

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/PostyAI.git
cd PostyAI
```

### 2. Set Up Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Key
Create a `.env` file in the project root:
```bash
# Create .env file
touch .env
```

Add your Groq API key to the `.env` file:
```env
# Get your free API key from https://console.groq.com
GROQ_API_KEY=your_actual_groq_api_key_here

# Optional Flask configuration
FLASK_ENV=development
FLASK_DEBUG=True
```

### 5. Run the Application
```bash
python app.py
```

The application will be available at: **http://localhost:8000**

## 🔧 Configuration

### Environment Variables
| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `GROQ_API_KEY` | Your Groq API key | ✅ Yes | None |
| `FLASK_ENV` | Flask environment | ❌ No | `development` |
| `FLASK_DEBUG` | Enable Flask debug mode | ❌ No | `True` |

### API Rate Limits
- **Post Generation**: 5 requests per minute
- **Batch Generation**: 2 requests per 5 minutes
- **Regeneration**: 3 requests per minute

## 📊 API Endpoints

### 🏠 **Main Routes**
- `GET /` - Main application interface
- `GET /api/tags` - Get available topic tags

### 🤖 **Generation Routes**
- `POST /api/generate` - Generate single post
- `POST /api/batch-generate` - Generate multiple posts
- `POST /api/regenerate` - Regenerate existing post

### 📈 **Utility Routes**
- `GET /api/history` - Get post generation history
- `GET /api/export/<post_id>` - Export specific post

### Example API Request
```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "length": "Medium",
    "language": "English", 
    "tag": "UX Design",
    "tone": "Professional"
  }'
```

## 🛠️ Development

### Project Structure
```
PostyAI/
├── app.py                 # Main Flask application
├── few_shot.py           # Few-shot learning system
├── post_generator.py     # Post generation logic
├── llm_helper.py         # LLM integration
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables
├── data/
│   └── processed_posts.json  # Training data
├── static/
│   ├── css/
│   │   └── main.css     # Styles and themes
│   ├── js/
│   │   ├── main.js      # Core utilities
│   │   └── app.js       # Application logic
│   └── favicon.svg      # Animated favicon
├── templates/
│   ├── base.html        # Base template
│   ├── index.html       # Main page
│   ├── 404.html         # Not found page
│   └── 500.html         # Server error page
└── src/
    └── utils/
        └── post_analytics.py  # Post analysis
```

### Adding New Features
1. **New Topic Categories**: Add to `data/processed_posts.json`
2. **Custom Prompts**: Modify `post_generator.py`
3. **UI Enhancements**: Update templates and static files
4. **API Extensions**: Add routes to `app.py`

## 🐳 Docker Deployment

Build and run with Docker:
```bash
# Build the image
docker build -t postyai .

# Run the container
docker run -p 8000:8000 --env-file .env postyai
```

## 🚀 Production Deployment

### Using Gunicorn
```bash
pip install gunicorn
gunicorn --bind 0.0.0.0:8000 --workers 4 app:app
```

### Environment Variables for Production
```env
GROQ_API_KEY=your_production_api_key
FLASK_ENV=production
FLASK_DEBUG=False
```

## 🔍 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Make sure you're in the virtual environment
source venv/bin/activate
pip install -r requirements.txt
```

**2. API Key Issues**
```bash
# Check if .env file exists and has correct key
cat .env
# Should show: GROQ_API_KEY=your_key_here
```

**3. Port Already in Use**
```bash
# Kill existing processes
lsof -ti:8000 | xargs kill -9
# Or use a different port
python app.py  # Will run on port 8000 by default
```

**4. Missing Data Files**
```bash
# Ensure data directory exists
ls -la data/
# Should contain processed_posts.json
```

### Performance Tips
- Use Redis for rate limiting in production
- Implement caching for frequently requested tags
- Consider using a production WSGI server like Gunicorn
- Enable gzip compression for static assets

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Run tests: `python -m pytest` (if available)
5. Commit changes: `git commit -am 'Add feature'`
6. Push to branch: `git push origin feature-name`
7. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Groq**: For providing fast and reliable AI inference
- **DeepSeek**: For the powerful language model
- **Flask**: For the lightweight web framework
- **Community**: For feedback and contributions

## 🆘 Support

If you encounter any issues or have questions:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Search [existing issues](https://github.com/yourusername/PostyAI/issues)
3. Create a [new issue](https://github.com/yourusername/PostyAI/issues/new) with:
   - Python version
   - Error messages
   - Steps to reproduce

---

**Made with ❤️ for content creators who want to leverage AI for better LinkedIn engagement!**

Ready to transform your LinkedIn presence? Get your free Groq API key and start generating amazing content! 🎉