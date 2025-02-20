import spacy

# Load the English NLP model from spaCy
nlp = spacy.load("en_core_web_sm")


def process_text(text):
    """Tokenize, stem, and perform NER on text."""
    if not isinstance(text, str) or text.strip() == "":
        return "No content"
    
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # Tokenization and normalized text output
    tokens = [token.text for token in doc if not token.is_punct]
    return f"Tokens: {tokens}, Entities: {entities}"
