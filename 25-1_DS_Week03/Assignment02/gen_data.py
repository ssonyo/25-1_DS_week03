import os, random

def generate_meaningful_sentence():
    subjects = ["I", "You", "We", "They", "He", "She"]
    be_verbs = {
        "I": "am", "You": "are", "We": "are", "They": "are", "He": "is", "She": "is"
    }
    aux_verbs = ["can", "should", "must", "will"]
    verbs = ["eat", "drink", "play", "watch", "read", "like", "love", "hate", "prefer"]
    objects = ["pizza", "coffee", "tea", "a book", "a game", "a movie", "music", "chocolate"]
    adjectives = ["happy", "sad", "tired", "excited", "bored", "hungry", "angry", "good", "bad"]

    pattern = random.choice([
        lambda s: f"{s} {be_verbs[s]} {random.choice(adjectives)}",
        lambda s: f"{s} {random.choice(verbs)} {random.choice(objects)}",
        lambda s: f"{s} {random.choice(aux_verbs)} {random.choice(verbs)}",
        lambda s: f"{s} {random.choice(aux_verbs)} {random.choice(verbs)} {random.choice(objects)}",
        lambda s: f"{s} {be_verbs[s]} not {random.choice(adjectives)}",
        lambda s: f"{s} does not {random.choice(verbs)} {random.choice(objects)}"
    ])

    subj = random.choice(subjects)
    return pattern(subj).strip()

sentences = [generate_meaningful_sentence() for _ in range(2000)]
sentences = sorted(list(set(sentences)))

sentences = sentences[:2000]

save_path = "data/simple_corpus.txt"
os.makedirs("data", exist_ok=True)

with open(save_path, "w", encoding="utf-8") as f:
    for sentence in sentences:
        f.write(sentence + "\n")

save_path
