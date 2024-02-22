import spacy

from pathlib import Path


CONFIGS_DIR = Path(Path(__file__).parent.resolve(), "../configs/models")

nlp_for_extract_nouns = spacy.load("en_core_web_sm")


def extract_nouns(prompt, nlp=None):
    if nlp is None:
        nlp = nlp_for_extract_nouns
    doc = nlp(prompt)
    return [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]

