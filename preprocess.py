import json
import logging
from typing import Optional, List, Dict, Any

from llm_helper import llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_chain_response(chain: PromptTemplate, input_data: Dict[str, Any], context_msg: str) -> Dict[str, Any]:
    """
    Invokes a chain with input_data and parses the JSON response.
    Raises an OutputParserException with a detailed message if parsing fails.
    """
    response = chain.invoke(input=input_data)
    json_parser = JsonOutputParser()
    try:
        result = json_parser.parse(response.content)
    except OutputParserException as err:
        error_msg = f"{context_msg} - Unable to parse response: {err}"
        logging.error(error_msg)
        raise OutputParserException(error_msg)
    return result


def extract_metadata(post: str) -> Dict[str, Any]:
    """
    Extract metadata from a LinkedIn post using an LLM.
    
    The response should include:
    - line_count
    - language (English or Hinglish)
    - tags (array of up to two text tags)
    
    Returns:
        A dict with the extracted metadata.
    """
    template = '''
    You are given a LinkedIn post. You need to extract number of lines, language of the post and tags.
    1. Return a valid JSON. No preamble. 
    2. JSON object should have exactly three keys: line_count, language and tags. 
    3. tags is an array of text tags. Extract maximum two tags.
    4. Language should be English or Hinglish (Hinglish means hindi + english)
    
    Here is the actual post on which you need to perform this task:  
    {post}
    '''
    pt = PromptTemplate.from_template(template)
    chain = pt | llm
    return parse_chain_response(chain, {"post": post}, "Extract Metadata")


def get_unified_tags(posts_with_metadata: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Unify the tags from multiple posts by invoking the LLM to merge similar tags.
    
    Returns:
        A mapping dictionary of original tags to the unified tag.
    """
    unique_tags = set()
    for post in posts_with_metadata:
        unique_tags.update(post.get('tags', []))
    unique_tags_list = ','.join(unique_tags)
    
    template = '''I will give you a list of tags. You need to unify tags with the following requirements,
    1. Tags are unified and merged to create a shorter list. 
       Example 1: "Jobseekers", "Job Hunting" can be all merged into a single tag "Job Search". 
       Example 2: "Motivation", "Inspiration", "Drive" can be mapped to "Motivation"
       Example 3: "Personal Growth", "Personal Development", "Self Improvement" can be mapped to "Self Improvement"
       Example 4: "Scam Alert", "Job Scam" etc. can be mapped to "Scams"
    2. Each tag should follow title case convention. Example: "Motivation", "Job Search"
    3. Output should be a JSON object, No preamble.
    4. Output should have mapping of original tag to the unified tag. 
       For example: {{"Jobseekers": "Job Search", "Job Hunting": "Job Search", "Motivation": "Motivation"}}
    
    Here is the list of tags: 
    {tags}
    '''
    pt = PromptTemplate.from_template(template)
    chain = pt | llm
    return parse_chain_response(chain, {"tags": unique_tags_list}, "Unify Tags")


def process_posts(raw_file_path: str, processed_file_path: Optional[str] = None) -> None:
    """
    Processes raw LinkedIn posts by enriching them with metadata and then unifying tags.
    
    It reads posts from raw_file_path, processes each one, and writes the enriched posts to processed_file_path.
    """
    # Read raw posts
    with open(raw_file_path, encoding='utf-8') as file:
        posts = json.load(file)
    
    enriched_posts = []
    for post in posts:
        # Extract metadata for each post
        metadata = extract_metadata(post.get('text', ''))
        # Merge original post with metadata (requires Python 3.9+)
        post_with_metadata = post | metadata
        enriched_posts.append(post_with_metadata)
    
    # Get unified tags mapping
    unified_tags_mapping = get_unified_tags(enriched_posts)
    
    # Replace original tags with unified tags, filtering out any missing mapping
    for post in enriched_posts:
        current_tags = post.get('tags', [])
        new_tags = {unified_tags_mapping[tag] for tag in current_tags if tag in unified_tags_mapping}
        post['tags'] = list(new_tags)
    
    # Write enriched posts to processed file if provided
    if processed_file_path:
        with open(processed_file_path, 'w', encoding='utf-8') as outfile:
            json.dump(enriched_posts, outfile, indent=4)
        logging.info(f"Processed posts have been written to {processed_file_path}")
    else:
        logging.info("No processed_file_path provided, skipping writing to file.")


if __name__ == "__main__":
    process_posts("data/raw_posts.json", "data/processed_posts.json")


