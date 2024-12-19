import streamlit as st
from few_shot import FewShotPosts
from post_generator import generate_post

# Configure the page
st.set_page_config(
    page_title="Posty - AI Post Generator",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS with dark mode
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
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
        font-family: 'Inter', sans-serif;
        color: var(--text-color);
    }
    
    .main {
        padding: 2rem;
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
        margin-bottom: 3rem;
    }
    
    .card {
        background: var(--card-background);
        padding: 2rem;
        border-radius: 1rem;
        border: 1px solid var(--border-color);
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
    }

    .stSpinner>div {
        border-color: var(--accent-color) !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu, footer {display: none;}
    .viewerBadge_container__1QSob {display: none;}
    </style>
""", unsafe_allow_html=True)

def main():
    # Hero Section with minimal design
    st.markdown("<h1>✨ Posty AI</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Transform your ideas into engaging social content</p>", unsafe_allow_html=True)
    
    # Main Container
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        fs = FewShotPosts()
        tags = fs.get_tags()
        
        with col1:
            selected_tag = st.selectbox("💡 Topic", options=tags)
        with col2:
            selected_length = st.selectbox("📏 Length", options=["Short", "Medium", "Long"])
        with col3:
            selected_language = st.selectbox("🌐 Language", options=["English", "Hinglish"])
        
        st.markdown("</div>", unsafe_allow_html=True)

        # Generate Button
        if st.button("✨ Generate Post"):
            with st.spinner('Creating something amazing...'):
                try:
                    post = generate_post(selected_length, selected_language, selected_tag)
                    st.markdown("<div class='output-container'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='tag'>{selected_tag}</div>", unsafe_allow_html=True)
                    st.write(post)
                    st.markdown("</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error("✨ Something went wrong. Let's try again!")

if __name__ == "__main__":
    main()