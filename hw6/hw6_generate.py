import random
import re
from collections import defaultdict


def load_text(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()


def clean_text(text):
    # Normalize whitespace and remove weird characters, keep basic punctuation
    text = re.sub(r'[^\w\s\u4e00-\u9fff.,!?，。！？、]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def build_model(text):
    words = text.split()
    bigram_model  = defaultdict(list)  # (w1, w2) -> [next_word, ...]
    trigram_model = defaultdict(list)  # (w1, w2, w3) -> [next_word, ...]
    sentence_starters = []             # words that begin sentences

    for i in range(len(words)):
        # Track sentence starters
        if i == 0 or words[i - 1] in '.!?。！？':
            if i + 1 < len(words):
                sentence_starters.append((words[i], words[i + 1]))

        # Bigram: (w1, w2) -> w3
        if i + 2 < len(words):
            key2 = (words[i], words[i + 1])
            bigram_model[key2].append(words[i + 2])

        # Trigram: (w1, w2, w3) -> w4
        if i + 3 < len(words):
            key3 = (words[i], words[i + 1], words[i + 2])
            trigram_model[key3].append(words[i + 3])

    return bigram_model, trigram_model, sentence_starters


def generate(bigram_model, trigram_model, sentence_starters, num_words=40):
    # Start from a sentence beginning if possible
    if sentence_starters:
        key = random.choice(sentence_starters)
    else:
        key = random.choice(list(bigram_model.keys()))

    result = list(key)

    for _ in range(num_words):
        w1, w2 = result[-2], result[-1]

        # Try trigram first (uses last 3 words for more coherent output)
        if len(result) >= 3:
            trigram_key = (result[-3], w1, w2)
            if trigram_key in trigram_model:
                next_word = random.choice(trigram_model[trigram_key])
                result.append(next_word)
                continue

        # Fall back to bigram
        bigram_key = (w1, w2)
        if bigram_key in bigram_model:
            next_word = random.choice(bigram_model[bigram_key])
            result.append(next_word)
        else:
            # Dead end: jump to a new random sentence starter
            if sentence_starters:
                new_start = random.choice(sentence_starters)
                result.extend(list(new_start))
            else:
                break

    return ' '.join(result)


if __name__ == '__main__':
    filename = 'tw.txt'
    text = load_text(filename)
    text = clean_text(text)

    bigram_model, trigram_model, sentence_starters = build_model(text)

    print("=== Generated Text ===")
    print(generate(bigram_model, trigram_model, sentence_starters, num_words=40))
