import os
import random

def generate_meaningful_sentence():
    templates = []

    subjects = ["I", "You", "We", "They", "He", "She"]
    be_verb = {"I": "am", "You": "are", "We": "are", "They": "are", "He": "is", "She": "is"}

    # emotion group
    emotions = ["happy", "sad", "angry", "tired", "bored", "excited", "nervous", "confident", "jealous", "curious"]
    for subj in subjects:
        for emo in emotions:
            templates.append(f"{subj} {be_verb[subj]} {emo}")
            templates.append(f"{subj} {be_verb[subj]} not {emo}")
            templates.append(f"{subj} feels {emo}")
            templates.append(f"{subj} does not feel {emo}")

    # food group
    foods = ["pizza", "coffee", "tea", "chocolate", "noodles", "burger", "sushi", "salad", "steak", "ice cream"]
    verbs = ["like", "love", "hate", "prefer", "eat", "drink", "want", "need"]
    for subj in subjects:
        for v in verbs:
            for f in foods:
                templates.append(f"{subj} {v} {f}")
                templates.append(f"{subj} do not {v} {f}")

    # activity group
    activities = ["watch a movie", "read a book", "play a game", "listen to music",
                  "go shopping", "take a walk", "cook dinner", "do yoga"]
    for subj in subjects:
        for act in activities:
            templates.append(f"{subj} want to {act}")
            templates.append(f"{subj} do not want to {act}")
            templates.append(f"{subj} can {act}")
            templates.append(f"{subj} cannot {act}")
            templates.append(f"{subj} is going to {act}")

    thoughts = ["go out", "stay home", "try something new", "talk to you", "sleep more", "work hard"]
    for subj in subjects:
        for t in thoughts:
            templates.append(f"{subj} thinks they should {t}")
            templates.append(f"{subj} does not think they should {t}")

    sentences = list(set(templates))
    random.shuffle(sentences)
    return sentences

# sentence generation
sentences = generate_meaningful_sentence()
sentences = sentences[:2000]

save_path = "data/simple_corpus.txt"
os.makedirs("data", exist_ok=True)

with open(save_path, "w", encoding="utf-8") as f:
    for sentence in sentences:
        f.write(sentence + "\n")

print(f"Saved {len(sentences)} structured sentences to {save_path}")
