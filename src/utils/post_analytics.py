def get_post_stats(post):
    # Calculate word count
    words = len(post.split())
    
    # Estimate reading time (average person reads 200 words per minute)
    reading_time = round(words / 200, 1)
    
    # Simple engagement score based on length, emoji usage, and formatting
    engagement_score = calculate_engagement_score(post)
    
    return {
        "word_count": words,
        "reading_time": reading_time,
        "engagement_score": engagement_score
    }

def calculate_engagement_score(post):
    score = 5  # Base score
    
    # Add points for optimal length (300-500 words)
    words = len(post.split())
    if 300 <= words <= 500:
        score += 2
    
    # Add points for emoji usage (but not too many)
    emoji_count = sum(1 for char in post if char in '😊🎉✨👍🔥')
    if 1 <= emoji_count <= 5:
        score += 2
    elif emoji_count > 5:
        score -= 1
    
    # Add points for formatting (headers, bullet points, etc.)
    if '#' in post or '*' in post or '-' in post:
        score += 1
        
    return min(10, score)  # Cap at 10