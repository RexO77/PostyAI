#!/usr/bin/env python3
"""
Script to further consolidate topics to a core set of ~20-30 essential categories
"""

import json
from collections import Counter

# Define aggressive consolidation to core categories
CORE_CONSOLIDATION = {
    # Programming & Tech (merge into fewer categories)
    "Programming": "Programming",
    "Web Development": "Programming", 
    "JavaScript": "Programming",
    "TypeScript": "Programming",
    "Python": "Programming",
    "Rust": "Programming",
    "CSS": "Programming",
    "Backend": "Programming",
    "Frontend": "Programming",
    "DevOps": "Programming",
    "AI": "Programming",
    "AI Tools": "Programming",
    "Machine Learning": "Programming",
    "Data Science": "Programming",
    "Version Control": "Programming",
    "Best Practices": "Programming",
    "Debugging": "Programming",
    "Documentation": "Programming",
    "Performance": "Programming",
    "Optimization": "Programming",
    "Automation": "Programming",
    "Open Source": "Programming",
    "Technical Writing": "Programming",
    "Software Architecture": "Programming",
    "Engineering": "Programming",
    "Computer Science": "Programming",
    
    # Design & UX
    "UX Design": "Design",
    "Design Tips": "Design",
    "Color Theory": "Design",
    "Layout": "Design",
    "Accessibility": "Design",
    
    # Career & Professional
    "Career Advice": "Career",
    "Job Search": "Career", 
    "Leadership": "Career",
    "Networking": "Career",
    "Personal Brand": "Career",
    "Public Speaking": "Career",
    "Soft Skills": "Career",
    "Communication": "Career",
    "Teamwork": "Career",
    "Hiring": "Career",
    
    # Productivity & Work
    "Productivity": "Productivity",
    "Time Management": "Productivity",
    "Morning Routine": "Productivity",
    "Procrastination": "Productivity",
    "Life Hacks": "Productivity",
    "Work Life Balance": "Productivity",
    "Remote Work": "Productivity",
    "Meetings": "Productivity",
    
    # Mental Health & Wellness
    "Mental Health": "Wellness",
    "Self Improvement": "Wellness",
    "Motivation": "Wellness",
    "Digital Wellness": "Wellness",
    "Imposter Syndrome": "Wellness",
    "Consistency": "Wellness",
    
    # Learning & Growth
    "Learning": "Learning",
    "Education": "Learning",
    "Knowledge Sharing": "Learning",
    "Problem Solving": "Learning",
    "Coding Bootcamp": "Learning",
    
    # Business & Entrepreneurship
    "Entrepreneurship": "Business",
    "Startup Life": "Business",
    "Business Tips": "Business",
    "Product Launch": "Business",
    "Freelancing": "Business",
    "Side Projects": "Business",
    "Growth": "Business",
    
    # Social & Content
    "Social Media": "Content",
    "LinkedIn": "Content",
    "Influencer": "Content",
    "Organic Growth": "Content",
    "Community": "Content",
    
    # Lifestyle & Personal
    "Humor": "Lifestyle",
    "Relationships": "Lifestyle",
    "Parenting": "Lifestyle",
    "Philosophy": "Lifestyle",
    "Culture": "Lifestyle",
    "Confession": "Lifestyle",
    "Relatability": "Lifestyle",
    "Sunday Thoughts": "Lifestyle",
    "Satire": "Lifestyle",
    "Question": "Lifestyle",
    "Short": "Lifestyle",
    
    # Work Culture & Issues
    "Corporate Life": "Work Culture",
    "Red Flags": "Work Culture",
    "Scams": "Work Culture",
    "Inclusion": "Work Culture",
    "Production": "Work Culture",
    
    # Special/Unique (keep as-is or merge)
    "Milestone": "Achievements",
    "Project Ideas": "Learning",
    "Tech Conferences": "Learning",
    "Sapne": "Lifestyle",  # Keep unique cultural content
}

def aggressive_consolidation():
    """Consolidate to core categories"""
    
    # Load the data
    with open('data/processed_posts.json', 'r') as f:
        data = json.load(f)
    
    print("🎯 Performing aggressive consolidation to core topics...")
    
    # Track changes
    original_tags = []
    for post in data:
        original_tags.extend(post['tags'])
    
    original_unique = len(set(original_tags))
    
    # Apply aggressive consolidation
    for post in data:
        new_tags = []
        for tag in post['tags']:
            core_tag = CORE_CONSOLIDATION.get(tag, tag)
            if core_tag not in new_tags:  # Avoid duplicates within same post
                new_tags.append(core_tag)
        post['tags'] = new_tags
    
    # Get new stats
    new_tags = []
    for post in data:
        new_tags.extend(post['tags'])
    
    new_unique = len(set(new_tags))
    
    # Save the consolidated data
    with open('data/processed_posts.json', 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"\n✅ Aggressive consolidation complete!")
    print(f"📊 Unique tags: {original_unique} → {new_unique} (reduced by {original_unique - new_unique})")
    print(f"📋 Total tag instances: {len(original_tags)} → {len(new_tags)}")
    
    # Show all core categories
    tag_counts = Counter(new_tags)
    print(f"\n🎯 Core categories ({len(tag_counts)} total):")
    for i, (tag, count) in enumerate(tag_counts.most_common(), 1):
        print(f"   {i:2d}. {tag}: {count} posts")
    
    print(f"\n💾 Updated data saved to data/processed_posts.json")
    
    return new_unique

if __name__ == "__main__":
    aggressive_consolidation()
