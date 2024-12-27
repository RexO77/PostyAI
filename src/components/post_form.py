import streamlit as st
from src.utils.post_analytics import get_post_stats

def render_post_form(tags, on_generate):
    # Column layout for the main inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_tag = st.selectbox("💡 Topic", options=tags)
    with col2:
        selected_length = st.selectbox("📏 Length", options=["Short", "Medium", "Long"])
    with col3:
        selected_language = st.selectbox("🌐 Language", options=["English", "Hinglish"])
    
    # Add styling for the tone selector container
    st.markdown("""
        <style>
        .stSlider {
            background-color: var(--card-background);
            border: 1px solid var(--border-color);
            border-radius: 0.5rem;
            padding: 1.5rem;
            margin-top: 1rem;
        }
        .stSlider > div > div > div {
            background-color: var(--accent-color) !important;
        }
        .stSlider .stSlider-value {
            color: var(--text-color) !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Tone selector with consistent styling
    tone = st.select_slider(
        "🎭 Select Post Tone",
        options=["Professional", "Casual", "Inspirational", "Educational"],
        value="Professional",  # Default value
    )
    
    # Generate button
    if st.button("✨ Generate Post", type="primary"):
        return {
            "tag": selected_tag,
            "length": selected_length,
            "language": selected_language,
            "tone": tone
        }
    return None