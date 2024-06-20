import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s: %(message)s',
                    datefmt='%d.%m.%Y %H:%M:%S')

def clean_text(text: str) -> str:
    """
    Removes or replaces unwanted characters from the input text and checks for inappropriate content.
    
    Args:
        text (str): The text to be cleaned.
        
    Returns:
        str: The cleaned text.
    """
    replacements = {
        '\u2014': '-',    # Em dash to regular dash
        '\u2019': "'",    # Right single quotation mark to apostrophe
        '\U0001f680': '', # Remove rocket emoji
        '\"': '',         # Remove double quote
        # Add further unwanted symbols here
    }

    # Replace unwanted characters
    for key, value in replacements.items():
        text = text.replace(key, value)
    
    # Remove newline characters
    text = text.replace('\n', ' ').replace('\r', ' ')

    # Additional cleaning logic to filter out certain phrases or toxic responses
    inappropriate_phrases = ["toxic response", "additional unwanted content"]
    for phrase in inappropriate_phrases:
        if phrase.lower() in text.lower():
            logging.warning(f"Filtered out inappropriate content: {phrase}")
            return ""  # Return an empty string or handle as needed
            
    return text.strip()  # Ensure no leading or trailing whitespace