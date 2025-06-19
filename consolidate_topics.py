#!/usr/bin/env python3
"""
Script to consolidate duplicate and similar topics in processed_posts.json
"""

import json
from collections import Counter

# Define consolidation mapping
TAG_CONSOLIDATION = {
    # Humor consolidation
    "Dev Humor": "Humor",
    "Programming Humor": "Humor",
    
    # Development consolidation  
    "Development": "Web Development",
    "Software Development": "Web Development",
    
    # Career consolidation
    "Career": "Career Advice",
    "Career Development": "Career Advice", 
    "Career Growth": "Career Advice",
    "Professional Growth": "Career Advice",
    "Tech Careers": "Career Advice",
    
    # Side projects consolidation
    "Side Project": "Side Projects",
    
    # Git/GitHub consolidation
    "Git": "Version Control",
    "GitHub": "Version Control",
    
    # Self improvement consolidation
    "Self Care": "Self Improvement",
    
    # Work/life consolidation
    "Work From Home": "Remote Work",
    "Work Stress": "Work Life Balance",
    
    # Learning consolidation
    "Learning in Public": "Learning",
    "Knowledge Sharing": "Learning",
    
    # Tech consolidation
    "Tech Burnout": "Mental Health",
    "Tech Journey": "Career Advice",
    "Tech Conferences": "Networking",
    "Tech Interviews": "Job Search",
    
    # Design consolidation
    "UI/UX": "UX Design",
    "Design Principles": "Design Tips",
    "Design Systems": "Design Tips",
    "Mobile Design": "UX Design",
    
    # Motivation consolidation
    "Monday Motivation": "Motivation",
    
    # Other consolidations
    "Coding Interview": "Job Search",
    "Tech Trends": "Programming",
    "Software Architecture": "Best Practices",
    "Code Review": "Best Practices",
    "Friday Deploy": "DevOps",
    "Containers": "DevOps",
    "Docker": "DevOps",
    "Microservices": "Backend",
    "Database Design": "Backend",
    "API Design": "Backend",
}

def consolidate_topics():
    """Load, consolidate, and save the topics"""
    
    # Load the data
    with open('data/processed_posts.json', 'r') as f:
        data = json.load(f)
    
    print("Original data loaded. Processing consolidation...")
    
    # Track changes
    original_tags = []
    for post in data:
        original_tags.extend(post['tags'])
    
    original_unique = len(set(original_tags))
    
    # Apply consolidation
    for post in data:
        # Update tags using consolidation mapping
        new_tags = []
        for tag in post['tags']:
            consolidated_tag = TAG_CONSOLIDATION.get(tag, tag)
            if consolidated_tag not in new_tags:  # Avoid duplicates within same post
                new_tags.append(consolidated_tag)
        post['tags'] = new_tags
    
    # Get new stats
    new_tags = []
    for post in data:
        new_tags.extend(post['tags'])
    
    new_unique = len(set(new_tags))
    
    # Save the consolidated data
    with open('data/processed_posts.json', 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"\n✅ Consolidation complete!")
    print(f"📊 Unique tags: {original_unique} → {new_unique} (reduced by {original_unique - new_unique})")
    print(f"📋 Total tag instances: {len(original_tags)} → {len(new_tags)}")
    
    # Show the most common tags now
    tag_counts = Counter(new_tags)
    print(f"\n📈 Top 15 tags after consolidation:")
    for tag, count in tag_counts.most_common(15):
        print(f"   {tag}: {count}")
    
    print(f"\n💾 Updated data saved to data/processed_posts.json")
    
    return len(set(new_tags))

if __name__ == "__main__":
    consolidate_topics()
