import os
import torch
from transformers import BertTokenizer, logging
from bert_ate_absa_models import bert_ATE, bert_ABSA
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.set_verbosity_error()

# Load device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load models and tokenizer
pretrain_model_name = "bert-large-uncased"
tokenizer = BertTokenizer.from_pretrained(pretrain_model_name)

# User's choice for model selection
print("=========================== MENU ===========================")
print("Choose which model files to use:\n")
print("1) Models fine-tuned by us --> (ate_model.pkl and absa_model.pkl)\n")
print("2) Models trained by user if he/she run the training files before running this --> (ate_model_v1.pkl and absa_model_v1.pkl)\n")

choice = input("Enter 1 or 2: ").strip()

if choice == "2":
    print("Loading ate_model_v1.pkl and absa_model_v1.pkl models.")
    print("================================================================")
    ate_model_file = "ate_model_v1.pkl"
    absa_model_file = "absa_model_v1.pkl"
    if not (os.path.exists(ate_model_file) and os.path.exists(absa_model_file)):
        print(f"Files not found. Falling back to pre-trained models --> (ate_model.pkl and absa_model.pkl).\n")
        print("================================================================")
        ate_model_file = "ate_model.pkl"
        absa_model_file = "absa_model.pkl"
else:
    print("Loading ate_model.pkl and absa_model.pkl models.")
    print("================================================================")
    ate_model_file = "ate_model.pkl"
    absa_model_file = "absa_model.pkl"

# Load trained models
ate_model = bert_ATE.from_pretrained(pretrain_model_name, num_labels=3).to(DEVICE)
ate_model.load_state_dict(torch.load(ate_model_file))
ate_model.eval()

absa_model = bert_ABSA.from_pretrained(pretrain_model_name, num_labels=3).to(DEVICE)
absa_model.load_state_dict(torch.load(absa_model_file))
absa_model.eval()

# Function to extract aspect terms
def extract_aspect_terms(sentence):
    tokens = tokenizer.tokenize(sentence)
    input_ids = tokenizer.encode(sentence, return_tensors="pt").to(DEVICE)
    attention_mask = (input_ids != 0).long()

    with torch.no_grad():
        outputs = ate_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]
        predictions = torch.argmax(logits, dim=2).squeeze().cpu().tolist()

    aspect_terms = []
    current_term = ""
    for token, label in zip(tokens, predictions[1:-1]):  # Exclude [CLS] and [SEP]
        # If a token is labeled B-Term but starts with "##", treat it as I-Term
        # so it merges into the current aspect term.
        if token.startswith("##") and label == 1:
            label = 2

        if label == 1:  # B-Term
            # If we already have an ongoing term, close it out
            if current_term:
                aspect_terms.append(current_term)
            current_term = token.lstrip("##")  # Start a new aspect term
        elif label == 2:  # I-Term
            current_term += token.lstrip("##")  # Continue the current aspect term
        else:  # Non-Aspect token
            if current_term:
                aspect_terms.append(current_term)
                current_term = ""
    # Append any remaining aspect term
    if current_term:
        aspect_terms.append(current_term)

    return aspect_terms

# Function to determine polarities of aspect terms
def determine_polarity(sentence, aspect_terms):
    polarities = {}
    for term in aspect_terms:
        # Prepare input for ABSA
        term_with_context = f"[CLS] {term} [SEP] {sentence} [SEP]"
        input_ids = tokenizer.encode(term_with_context, return_tensors="pt").to(DEVICE)
        attention_mask = (input_ids != 0).long()

        with torch.no_grad():
            outputs = absa_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            polarity = torch.argmax(logits, dim=1).item()

        # Map polarity
        polarity_label = {0: "Negative", 1: "Neutral", 2: "Positive"}
        polarities[term] = polarity_label[polarity]

    return polarities

# Main 
def main_pipeline(sentence):
    print("\n=========================== Input Sentence ===========================")
    print(f"Input Sentence: {sentence}")
    aspect_terms = extract_aspect_terms(sentence)
    print(f"Extracted Aspect Terms: {aspect_terms}")

    if aspect_terms:
        polarities = determine_polarity(sentence, aspect_terms)
        print("\n================================================================")
        print("Aspect Terms and Their Polarities:")
        for term, polarity in polarities.items():
            print(f"  {term}: {polarity}")
    else:
        print("No aspect terms found.")
        print("================================================================")

if __name__ == "__main__":
    while True:
        print("\n================================================================")
        sentence = input("Enter a sentence (or type 'q' to quit): ")
        if sentence.lower() in ["q", "quit"]:
            print("Exiting...")
            print("================================================================")
            break
        main_pipeline(sentence)
