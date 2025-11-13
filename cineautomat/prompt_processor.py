import torch
from transformers import pipeline


def process_prompt(prompt):
    # Load a pre-trained NLP model for text analysis
    nlp = pipeline("ner", model="dslim/bert-base-NER")

    # Extract entities from the prompt
    entities = nlp(prompt)
    scene_details = {
        "characters": [],
        "setting": [],
        "actions": []
    }

    for entity in entities:
        if entity['entity'].startswith('B-PER') or entity['entity'].startswith('I-PER'):
            scene_details["characters"].append(entity['word'])
        elif entity['entity'].startswith('B-LOC') or entity['entity'].startswith('I-LOC'):
            scene_details["setting"].append(entity['word'])
        # Simple heuristic for actions (verbs)
        if entity['word'].endswith('ing'):
            scene_details["actions"].append(entity['word'])

    return scene_details


# Example usage
prompt = "A quirky fox in a pastel-colored cafe reading a book"
details = process_prompt(prompt)
print(details)