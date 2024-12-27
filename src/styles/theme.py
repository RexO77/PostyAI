def get_theme_styles():
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@100..900&family=Lexend+Deca:wght@100..900&display=swap');
    
    :root {
        --background-color: #0E1117;
        --card-background: #1E1E1E;
        --text-color: #E6E6E6;
        --accent-color: #7C3AED;
        --accent-light: #8B5CF6;
        --border-color: #2D2D2D;
    }
    
    .stApp {
        background-color: var(--background-color) !important;
    }
    
    * {
        font-family: 'Lexend Deca', sans-serif;
        color: var(--text-color);
    }
    
    .main {
        padding: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, var(--accent-color) 0%, var(--accent-light) 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 0.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        margin-top: 1rem;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px -1px rgba(0, 0, 0, 0.3);
    }
    
    .output-container {
        background: var(--card-background);
        padding: 2rem;
        border-radius: 1rem;
        border: 1px solid var(--border-color);
        margin: 2rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    
    h1 {
        font-size: 3rem !important;
        font-weight: 700 !important;
        margin-bottom: 1rem !important;
        background: linear-gradient(135deg, var(--accent-color) 0%, var(--accent-light) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #9CA3AF;
        margin-bottom: 2rem;
    }
    
    .tag {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background: rgba(124, 58, 237, 0.1);
        color: var(--accent-light);
        border-radius: 2rem;
        font-size: 0.875rem;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        border: 1px solid var(--accent-color);
    }

    .stSelectbox>div>div {
        background: var(--card-background) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 0.5rem;
    }

    .stSlider>div>div {
        background: var(--card-background) !important;
    }

    .stSpinner>div {
        border-color: var(--accent-color) !important;
    }
    
    /* Analytics Cards */
    .metric-card {
        background: var(--card-background);
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        border: 1px solid var(--border-color);
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--accent-light);
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #9CA3AF;
    }
    
    /* Hide Streamlit branding */
    #MainMenu, footer {display: none;}
    .viewerBadge_container__1QSob {display: none;}

    /* Improved spacing */
    .stSelectbox {
        margin-bottom: 0.5rem;
    }

    .stSlider {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """