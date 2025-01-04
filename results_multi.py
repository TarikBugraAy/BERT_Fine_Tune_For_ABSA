import csv
import json
import random
import torch
import warnings
import os

from transformers import BertTokenizer, logging
from bert_ate_absa_models import bert_ATE, bert_ABSA 

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.set_verbosity_error()


# LOAD MODELS AND TOKENIZER
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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


# Instantiate model classes and load their saved weights
ate_model = bert_ATE.from_pretrained(pretrain_model_name, num_labels=3).to(DEVICE)
ate_model.load_state_dict(torch.load(ate_model_file, map_location=DEVICE))
ate_model.eval()

absa_model = bert_ABSA.from_pretrained(pretrain_model_name, num_labels=3).to(DEVICE)
absa_model.load_state_dict(torch.load(absa_model_file, map_location=DEVICE))
absa_model.eval()


#CSV LOADING UTILITY
def load_csv_data(file_path):

    data = {}
    with open(file_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # Skip header if present

        for row in reader:
            if not row:
                continue
            sentence = row[0].strip()
            aspect_term = row[1].strip()
            polarity = row[2].strip()

            if sentence not in data:
                data[sentence] = []
            data[sentence].append((aspect_term, polarity))
    return data


#MODEL-INFERENCE FUNCTIONS
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
    # Exclude [CLS] and [SEP]
    for token, label in zip(tokens, predictions[1:-1]):
        # If a token is labeled B-Term but starts with "##", treat it as I-Term
        if token.startswith("##") and label == 1:
            label = 2

        if label == 1:  # B-Term
            if current_term:
                aspect_terms.append(current_term)
            current_term = token.lstrip("##")
        elif label == 2:  # I-Term
            current_term += token.lstrip("##")
        else:  # Non-aspect
            if current_term:
                aspect_terms.append(current_term)
                current_term = ""
    if current_term:
        aspect_terms.append(current_term)

    return aspect_terms

def determine_polarity(sentence, aspect_terms):
    polarities = {}
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

    for term in aspect_terms:
        term_with_context = f"[CLS] {term} [SEP] {sentence} [SEP]"
        input_ids = tokenizer.encode(term_with_context, return_tensors="pt").to(DEVICE)
        attention_mask = (input_ids != 0).long()

        with torch.no_grad():
            outputs = absa_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            polarity_idx = torch.argmax(logits, dim=1).item()

        polarities[term] = label_map[polarity_idx]

    return polarities

def predict_aspects_for_sentence(sentence):
    aspects = extract_aspect_terms(sentence)
    if aspects:
        polarities = determine_polarity(sentence, aspects)
        return [(a, polarities[a]) for a in aspects]
    else:
        return []


# MODE 1: RUN ENTIRE CSV, SAVE JSON
def run_entire_csv_and_save_json(data_dict, output_json="results.json"):
    results = []
    for sentence, gt_list in data_dict.items():
        predicted = predict_aspects_for_sentence(sentence)
        entry = {
            "sentence": sentence,
            "predicted_aspects": [
                {"aspect": asp, "polarity": pol} for (asp, pol) in predicted
            ],
            "ground_truth_aspects": [
                {"aspect": g_asp, "polarity": g_pol} for (g_asp, g_pol) in gt_list
            ],
        }
        results.append(entry)

    with open(output_json, mode="w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll sentences processed. Results saved to '{output_json}'.\n")


# MODE 2: RANDOM SAMPLE IN TERMINAL
def run_random_sample(data_dict):
    try:
        n = int(input("How many random sentences to sample? "))
    except ValueError:
        print("Invalid number. Returning to menu.\n")
        return

    all_sentences = list(data_dict.keys())
    if n > len(all_sentences):
        print(f"Requested {n}, but only {len(all_sentences)} sentences are available.")
        return

    chosen_sentences = random.sample(all_sentences, n)
    print(f"\nRandomly Selected {n} Sentence(s):\n")

    for i, sentence in enumerate(chosen_sentences, start=1):
        print(f"{i}. Sentence: {sentence}")
        predicted = predict_aspects_for_sentence(sentence)
        ground_truth = data_dict[sentence]  # List of (aspect, polarity)

        print("   Model Predictions:")
        if predicted:
            for (asp, pol) in predicted:
                print(f"     - {asp}: {pol}")
        else:
            print("     - No aspects found.")

        print("   Ground Truth:")
        for (gt_asp, gt_pol) in ground_truth:
            print(f"     - {gt_asp}: {gt_pol}")
        print()  # Blank line


# MAIN
if __name__ == "__main__":

    print("Loading the input data ...")
    # Input for test - the CSV file path
    csv_file_path = "data/testing.csv"  
    data_dict = load_csv_data(csv_file_path)

    while True:
        print("\n=========================== OPTIONS ===========================")
        print("\n1) Run model on entire testing file and save results to results.json")
        print("\n2) Randomly sample sentences from testing file (show results in console)")
        print("\n3) Quit\n")
        choice = input("Choose an option (1, 2, or 3): ").strip()

        if choice == "1":
            run_entire_csv_and_save_json(data_dict)
            print("================================================================")
        elif choice == "2":
            run_random_sample(data_dict)
            print("================================================================")
        elif choice == "3":
            print("Exiting...")
            print("================================================================")
            break
        else:
            print("Invalid choice. Please choose 1, 2, or 3.\n")