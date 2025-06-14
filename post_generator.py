from llm_helper import llm
from few_shot import FewShotPosts

few_shot = FewShotPosts()


def get_length_str(length):
    if length == "Short":
        return "1 to 5 lines"
    if length == "Medium":
        return "6 to 10 lines"
    if length == "Long":
        return "14 to 18 lines"


# ...existing code...
def generate_post(length, language, tag, tone=None):
    prompt = get_prompt(length, language, tag, tone)
    response = llm.invoke(prompt)
    content = response.content
    
    # Clean up the response by removing thinking process
    if "<think>" in content and "</think>" in content:
        # Extract content after </think>
        content = content.split("</think>")[-1].strip()
    
    # Remove any remaining thinking patterns
    if content.startswith('"') and content.endswith('"'):
        content = content[1:-1]
    
    return content.strip()

def get_prompt(length, language, tag, tone=None):
    length_str = get_length_str(length)
    prompt = f'''
    Generate a LinkedIn post using the below information. No preamble.

    1) Topic: {tag}
    2) Length: {length_str}
    3) Language: {language}
    4) Tone: {tone}
    If Language is Hinglish then it means it is a mix of Hindi and English.
    The script for the generated post should always be English.
    '''

    examples = few_shot.get_filtered_posts(length, language, tag)

    if len(examples) > 0:
        prompt += "4) Use the writing style as per the following examples."

    for i, post in enumerate(examples):
        post_text = post['text']
        prompt += f'\n\n Example {i+1}: \n\n {post_text}'

        if i == 1: # Use max two samples
            break

    return prompt


if __name__ == "__main__":
    print(generate_post("Medium", "English", "UX Design"))