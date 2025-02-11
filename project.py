import streamlit as st
from few_shot import FewShotPosts
from post_generator import generate_post
from src.components.header import render_header
from src.components.post_form import render_post_form
from src.components.post_display import render_post
from src.styles.theme import get_theme_styles

# Configure the page
st.set_page_config(
    page_title="Posty - AI Post Generator",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply theme
st.markdown(get_theme_styles(), unsafe_allow_html=True)

def main():
    render_header()
    
    fs = FewShotPosts()
    tags = fs.get_tags()
    
    form_data = render_post_form(tags, generate_post)
    
    if form_data:
        with st.spinner('Creating something amazing...'):
            try:
                post = generate_post(
                    form_data["length"], 
                    form_data["language"], 
                    form_data["tag"],
                    form_data["tone"]
                )
                should_regenerate = render_post(post, form_data["tag"])
                
                if should_regenerate:
                    st.rerun()
                    
            except Exception as e:
                st.error(f"✨ Something went wrong. Let's try again!: {e}")

if __name__ == "__main__":
    main()
