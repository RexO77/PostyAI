import streamlit as st
from src.utils.post_analytics import get_post_stats

def render_post(post, tag):
    
    # Display tags
    st.markdown(f"<div class='tag'>{tag}</div>", unsafe_allow_html=True)
    
    # Display the post
    st.write(post)
    
    # Display analytics
    stats = get_post_stats(post)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Word Count", stats["word_count"])
    with col2:
        st.metric("Reading Time", f"{stats['reading_time']} min")
    with col3:
        st.metric("Engagement Score", f"{stats['engagement_score']}/10")
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📋 Copy to Clipboard"):
            st.toast("Post copied to clipboard!", icon="✅")
    with col2:
        if st.button("🔄 Regenerate"):
            return True
            
    return False