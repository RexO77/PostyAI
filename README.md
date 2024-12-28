# Posty AI - AI Post Generator 🚀

A powerful AI-driven tool that analyzes LinkedIn influencers' writing patterns to generate engaging, personalized content that matches their unique style.

## 📖 Overview

Posty AI helps LinkedIn influencers maintain consistent content creation by:
- Analyzing existing posts to understand writing style and patterns
- Extracting key topics and themes automatically
- Generating new posts that maintain the author's authentic voice
- Supporting multiple languages (English and Hinglish)
- Offering customizable post length and tone settings

## 🏗️ Technical Architecture

<img src="resources/architecture.jpg"/>

### Stage 1: Content Analysis
- **Ingests LinkedIn posts in JSON format**: The system takes in raw LinkedIn posts provided in JSON format.
- **Extracts metadata**: It extracts important metadata such as topics, language, length, and engagement metrics (likes, comments, shares).
- **Processes and unifies content tags**: The tool processes the content to identify and unify tags for better categorization. For example, tags like "Jobseekers" and "Job Hunting" are unified under "Job Search".
- **Stores processed data**: The processed data, including the extracted metadata and unified tags, is stored in a structured format for efficient retrieval during content generation.

### Stage 2: Content Generation
- **Utilizes few-shot learning with filtered relevant examples**: The system uses few-shot learning, leveraging a small number of relevant examples to guide the AI in generating new content.
- **Considers post length, language, and topic preferences**: The AI takes into account the desired post length (Short, Medium, Long), language (English, Hinglish), and topic preferences to tailor the generated content.
- **Maintains consistent tone and style through AI prompting**: The AI is prompted to maintain a consistent tone and style that matches the user's previous posts, ensuring authenticity.
- **Generates engaging content with optimal formatting**: The final output is engaging, well-formatted content that aligns with the user's style and preferences.

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager
- Groq API access

### Setup Steps

1. Clone the repository:
```bash
git clone https://github.com/yourusername/posty-ai.git
cd posty-ai
```
2. To get started we first need to get an API_KEY from here: https://console.groq.com/keys. Inside `.env` update the value of `GROQ_API_KEY` with the API_KEY you created. 
3. To get started, first install the dependencies using:
    ```commandline
     pip install -r requirements.txt
    ```
4. Run the streamlit app:
   ```commandline
   streamlit run main.py
   ```
## 🎯 Features
- **Smart Topic Extraction: Automatically identifies and categorizes post themes**
- **Multi-language Support: Generates content in English and Hinglish**
- **Customizable Length: Short (1-5 lines), Medium (6-10 lines), Long (14-18 lines)**
- **Tone Control: Professional, Casual, Inspirational, or Educational**
- **Engagement Analytics: Provides metrics for word count, reading time, and engagement potential**
- **One-Click Actions: Copy to clipboard and regenerate options**