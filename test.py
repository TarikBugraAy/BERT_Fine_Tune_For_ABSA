import sys
import os
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch
from train import test_ate, test_absa, ate_test_loader, absa_test_loader, ate_model, absa_model, load_model_pkl
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)



DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Paths for pre-trained and user-trained models
PRETRAINED_ATE_MODEL_PATH = "ate_model.pkl"
PRETRAINED_ABSA_MODEL_PATH = "absa_model.pkl"
USER_TRAINED_ATE_MODEL_PATH = "ate_model_v1.pkl"
USER_TRAINED_ABSA_MODEL_PATH = "absa_model_v1.pkl"

def test_model(loader, model, model_name, target_names):
    print(f"\nTesting {model_name}...")
    truths, predictions = test_ate(loader, model) if model_name == "ATE" else test_absa(loader, model)
    print(classification_report(truths, predictions, target_names=target_names))

    # Generate and display confusion matrix
    cm = confusion_matrix(truths, predictions, labels=range(len(target_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix for {model_name}")
    plt.show()

def load_and_test_models(choice):
    if choice in ["1", "3"]:  # Pre-trained models
        print("\nUsing Pre-trained Models...")
        pretrained_ate_model = load_model_pkl(ate_model, PRETRAINED_ATE_MODEL_PATH).to(DEVICE)
        pretrained_absa_model = load_model_pkl(absa_model, PRETRAINED_ABSA_MODEL_PATH).to(DEVICE)

        # Test pre-trained models
        test_model(ate_test_loader, pretrained_ate_model, "ATE", ["Non-Aspect", "B-Term", "I-Term"])
        test_model(absa_test_loader, pretrained_absa_model, "ABSA", ["Negative", "Neutral", "Positive"])

    if choice in ["2", "3"]:  # User-trained models
        print("\nUsing User-trained Models...")
        user_trained_ate_model = load_model_pkl(ate_model, USER_TRAINED_ATE_MODEL_PATH).to(DEVICE)
        user_trained_absa_model = load_model_pkl(absa_model, USER_TRAINED_ABSA_MODEL_PATH).to(DEVICE)

        # Test user-trained models
        test_model(ate_test_loader, user_trained_ate_model, "ATE", ["Non-Aspect", "B-Term", "I-Term"])
        test_model(absa_test_loader, user_trained_absa_model, "ABSA", ["Negative", "Neutral", "Positive"])

if __name__ == "__main__":
    print("\n=========================== Testing ===============================")
    print("Classification Report and Confusion Matrix\n")
    print("Select testing mode:")
    print("1) Test pre-trained models")
    print("2) Test user-trained models")
    print("3) Test both pre-trained and user-trained models")
    print("=====================================================================\n")

    choice = input("Enter your choice (1, 2, or 3): ").strip()

    if choice not in ["1", "2", "3"]:
        print("\nInvalid choice. Exiting...")
        sys.exit(1)

    load_and_test_models(choice)
    print("\nTesting completed.")
