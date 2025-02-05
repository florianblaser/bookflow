import json
import os
import re
import time
from bs4 import BeautifulSoup
import textstat
from groq import Groq

folder_path = 'data/100_clean'
CEFR_LEVEL_THRESHOLD = 20
throttle_delay = 1  # seconds
dict_file = 'difficult_word_to_help.json'

def is_difficult(word):
    try:
        return textstat.flesch_kincaid_grade(word) > CEFR_LEVEL_THRESHOLD
    except Exception:
        return False

# Step 1: Collect all difficult words from all xHTML files.
difficult_words = set()
for filename in os.listdir(folder_path):
    if filename.endswith('.xhtml'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')
            text = soup.get_text()
            words = re.findall(r'\b\w+\b', text, flags=re.UNICODE)
            for word in words:
                lower_word = word.lower()
                if is_difficult(lower_word):
                    difficult_words.add(lower_word)

# Load existing dictionary if it exists.
if os.path.exists(dict_file):
    with open(dict_file, 'r', encoding='utf-8') as f:
        difficult_word_to_help = json.load(f)
else:
    difficult_word_to_help = {}

# Remove words already processed from the set.
remaining_words = difficult_words - set(difficult_word_to_help.keys())
difficult_words_list = list(remaining_words)

# Initialize Groq client.
client = Groq(api_key='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')

# Determine total batches (each batch of 100 words).
batch_size = 100
total_batches = (len(difficult_words_list) + batch_size - 1) // batch_size

# Process difficult words in batches.
for batch_index in range(total_batches):
    start = batch_index * batch_size
    batch = difficult_words_list[start:start + batch_size]
    print(f"Processing batch {batch_index + 1} of {total_batches} (words {start + 1} to {start + len(batch)})")
    
    words_str = ", ".join(batch)
    prompt = (
        f"Provide a defining translation in English for each of the following words: {words_str}. "
        "Use no more than three words per defining translation. "
        "Return the result as a JSON array where each element is an object with the keys 'word' and 'help'. "
        "Do not include any extra text outside the JSON."
    )

    # First attempt with gemma2-9b-it.
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gemma2-9b-it",
            stream=False,
        )
        response_text = chat_completion.choices[0].message.content.strip()
        definitions = json.loads(response_text)
        for entry in definitions:
            # Normalize keys to lowercase for consistency.
            word_key = entry['word'].lower()
            difficult_word_to_help[word_key] = entry['help']
    except Exception:
        # Second attempt with llama-3.3-70b-versatile.
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                stream=False,
            )
            response_text = chat_completion.choices[0].message.content.strip()
            definitions = json.loads(response_text)
            for entry in definitions:
                word_key = entry['word'].lower()
                difficult_word_to_help[word_key] = entry['help']
        except json.JSONDecodeError:
            print(f"Failed to parse JSON for batch: {batch}")
            print("Response was:")
            print(response_text)

    # Optional throttle delay.
    time.sleep(throttle_delay)

# Save the updated dictionary locally.
with open(dict_file, 'w', encoding='utf-8') as f:
    json.dump(difficult_word_to_help, f, ensure_ascii=False, indent=2)

# Step 3: Process each XHTML file and replace difficult words with "original_word [help]".
for filename in os.listdir(folder_path):
    if filename.endswith('.xhtml'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Build a regex pattern matching any word from difficult_words_list (case-insensitive)
        pattern = re.compile(
            r'\b(' + '|'.join(re.escape(word) for word in difficult_words_list) + r')\b',
            flags=re.IGNORECASE
        )

        def replacer(match):
            original_word = match.group(0)
            # Lookup using the dictionary (normalized to lowercase).
            help_text = difficult_word_to_help.get(original_word.lower(), '')
            return f"{original_word} [{help_text}]" if help_text else original_word

        new_content = pattern.sub(replacer, content)

        # Overwrite the file with the updated content.
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(new_content)
