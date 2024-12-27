# Posty AI - LinkedIn Post Generator 🚀

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
- Ingests LinkedIn posts in JSON format
- Extracts metadata: topics, language, length, and engagement metrics
- Processes and unifies content tags for better categorization
- Stores processed data for efficient retrieval

### Stage 2: Content Generation
- Utilizes few-shot learning with filtered relevant examples
- Considers post length, language, and topic preferences
- Maintains consistent tone and style through AI prompting
- Generates engaging content with optimal formatting

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
Copyright (C) Codebasics Inc. All rights reserved.


**Additional Terms:**
This software is licensed under the MIT License. However, commercial use of this software is strictly prohibited without prior written permission from the author. Attribution must be given in all copies or substantial portions of the software.
