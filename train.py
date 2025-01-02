import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from dataset import dataset_ATM, dataset_ABSA
from bert import bert_ATE, bert_ABSA
import torch
from transformers import logging

# Suppress warnings
logging.set_verbosity_error()

# Initialize device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

if DEVICE.type == "cuda":
    print(f"GPU in use: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

pretrain_model_name = "bert-large-uncased"
tokenizer = BertTokenizer.from_pretrained(pretrain_model_name)

# Initialize models
ate_model = bert_ATE.from_pretrained(pretrain_model_name, num_labels=3).to(DEVICE)
absa_model = bert_ABSA.from_pretrained(pretrain_model_name, num_labels=3).to(DEVICE)

# Optimizers
lr = 2e-5
optimizer_ATE = AdamW(ate_model.parameters(), lr=lr, weight_decay=1e-4)
optimizer_ABSA = AdamW(absa_model.parameters(), lr=lr, weight_decay=1e-4)

# Helper functions
def evl_time(t):
    min, sec = divmod(t, 60)
    hr, min = divmod(min, 60)
    return int(hr), int(min), int(sec)

# Save and Load Models as .pkl
def save_model_pkl(model, path):
    torch.save(model.state_dict(), path)

def load_model_pkl(model, path):
    model.load_state_dict(torch.load(path))
    return model

# ATE: Create mini-batches
def create_mini_batch(samples):
    ids_tensors = [s[1] for s in samples]
    ids_tensors = pad_sequence(ids_tensors, batch_first=True)

    tags_tensors = [s[2] for s in samples]
    tags_tensors = pad_sequence(tags_tensors, batch_first=True)

    masks_tensors = torch.zeros(ids_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(ids_tensors != 0, 1)

    return ids_tensors, tags_tensors, masks_tensors

# ATE: Train Model with Debugging
def train_ate(loader, model, optimizer, epochs):
    model.train()
    print("Starting ATE Training...")
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}...")
        epoch_loss = 0
        batch_count = 0
        for ids_tensors, tags_tensors, masks_tensors in loader:
            batch_count += 1
            ids_tensors, tags_tensors, masks_tensors = (
                ids_tensors.to(DEVICE),
                tags_tensors.to(DEVICE),
                masks_tensors.to(DEVICE),
            )
            optimizer.zero_grad()
            outputs = model(input_ids=ids_tensors, attention_mask=masks_tensors, labels=tags_tensors)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Print progress every 10 batches
            if batch_count % 10 == 0:
                print(f"  Batch {batch_count}/{len(loader)} - Loss: {loss.item():.4f}")

        print(f"Epoch {epoch + 1} completed. Average Loss: {epoch_loss / len(loader):.4f}")
    print("ATE Training Completed.")

# ATE: Test Model with Flattened Output
def test_ate(loader, model):
    model.eval()
    predictions, truths = [], []
    with torch.no_grad():
        for ids_tensors, tags_tensors, masks_tensors in loader:
            ids_tensors, tags_tensors, masks_tensors = (
                ids_tensors.to(DEVICE),
                tags_tensors.to(DEVICE),
                masks_tensors.to(DEVICE),
            )
            outputs = model(input_ids=ids_tensors, attention_mask=masks_tensors)
            logits = outputs["logits"]
            _, preds = torch.max(logits, dim=2)

            # Flatten the batch outputs
            for pred, truth in zip(preds, tags_tensors):
                predictions.extend(pred.cpu().tolist())
                truths.extend(truth.cpu().tolist())

    return truths, predictions

# ABSA: Create mini-batches
def create_mini_batch_absa(samples):
    max_len = max([len(s[1]) for s in samples])
    ids_tensors = [torch.cat([s[1], torch.zeros(max_len - len(s[1]), dtype=torch.long)]) for s in samples]
    ids_tensors = torch.stack(ids_tensors)
    segments_tensors = [torch.cat([s[2], torch.zeros(max_len - len(s[2]), dtype=torch.long)]) for s in samples]
    segments_tensors = torch.stack(segments_tensors)
    label_ids = torch.stack([s[3] for s in samples])
    masks_tensors = torch.zeros(ids_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(ids_tensors != 0, 1)
    return ids_tensors, segments_tensors, masks_tensors, label_ids

# ABSA: Train Model with Debugging and Dynamic Batch Adjustment
def train_absa(loader, val_loader, model, optimizer, epochs):
    print("Starting ABSA Training...")
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * total_steps, num_training_steps=total_steps)
    best_val_loss = float("inf")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}...")
        model.train()
        total_loss = 0
        batch_count = 0

        for ids_tensors, segments_tensors, masks_tensors, label_ids in loader:
            try:
                batch_count += 1
                ids_tensors, segments_tensors, masks_tensors, label_ids = (
                    ids_tensors.to(DEVICE),
                    segments_tensors.to(DEVICE),
                    masks_tensors.to(DEVICE),
                    label_ids.to(DEVICE),
                )
                optimizer.zero_grad()
                outputs = model(input_ids=ids_tensors, attention_mask=masks_tensors, token_type_ids=segments_tensors, labels=label_ids)
                loss = outputs["loss"]
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

                # Print progress every 10 batches
                if batch_count % 10 == 0:
                    print(f"  Batch {batch_count}/{len(loader)} - Loss: {loss.item():.4f}")

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print("  [WARNING] CUDA out of memory. Consider reducing batch size or model size.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")

        # Validation
        print("  Validating...")
        val_loss = validate_absa(val_loader, model)
        print(f"  Validation Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model_pkl(model, "absa_model_v1.pkl")
            print("  Model saved with improved validation loss.")
    print("ABSA Training Completed.")

# ABSA: Validate Model
def validate_absa(loader, model):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for ids_tensors, segments_tensors, masks_tensors, label_ids in loader:
            ids_tensors, segments_tensors, masks_tensors, label_ids = (
                ids_tensors.to(DEVICE),
                segments_tensors.to(DEVICE),
                masks_tensors.to(DEVICE),
                label_ids.to(DEVICE),
            )
            outputs = model(input_ids=ids_tensors, attention_mask=masks_tensors, token_type_ids=segments_tensors, labels=label_ids)
            loss = outputs["loss"]
            total_loss += loss.item()
    return total_loss / len(loader)

# ABSA: Test Model
def test_absa(loader, model):
    model.eval()
    predictions, truths = [], []
    with torch.no_grad():
        for ids_tensors, segments_tensors, masks_tensors, label_ids in loader:
            ids_tensors, segments_tensors, masks_tensors, label_ids = (
                ids_tensors.to(DEVICE),
                segments_tensors.to(DEVICE),
                masks_tensors.to(DEVICE),
                label_ids.to(DEVICE),
            )
            outputs = model(input_ids=ids_tensors, attention_mask=masks_tensors, token_type_ids=segments_tensors)
            logits = outputs["logits"]
            _, preds = torch.max(logits, dim=1)
            predictions.extend(preds.cpu().tolist())
            truths.extend(label_ids.cpu().tolist())
    return truths, predictions

# ATE Data
ate_train_ds = dataset_ATM(pd.read_csv("data/restaurants_train.csv"), tokenizer)
ate_test_ds = dataset_ATM(pd.read_csv("data/restaurants_test.csv"), tokenizer)
ate_train_loader = DataLoader(ate_train_ds, batch_size=8, collate_fn=create_mini_batch, shuffle=True)
ate_test_loader = DataLoader(ate_test_ds, batch_size=8, collate_fn=create_mini_batch, shuffle=False)

# ABSA Data
absa_train_ds = dataset_ABSA(pd.read_csv("data/ABSArestaurants_train.csv"), tokenizer)
absa_val_ds = dataset_ABSA(pd.read_csv("data/restaurants_val.csv"), tokenizer)
absa_test_ds = dataset_ABSA(pd.read_csv("data/restaurants_test.csv"), tokenizer)
absa_train_loader = DataLoader(absa_train_ds, batch_size=16, collate_fn=create_mini_batch_absa, shuffle=True)
absa_val_loader = DataLoader(absa_val_ds, batch_size=16, collate_fn=create_mini_batch_absa, shuffle=False)
absa_test_loader = DataLoader(absa_test_ds, batch_size=16, collate_fn=create_mini_batch_absa, shuffle=False)


#PROMPT USER FOR TRAINING MODE
if __name__ == "__main__":

    train_mode = input(
        "Select training mode:\n"
        "1) ATE Only\n"
        "2) ABSA Only\n"
        "3) Both ATE and ABSA\n"
        "Enter choice (1, 2, or 3 (or type 'q' to quit)): "
    ).strip()

    # TRAIN/TEST ATE IF CHOSEN
    if train_mode in ["1", "3"]:
        print("\nStarting to train ATE model...")
        train_ate(ate_train_loader, ate_model, optimizer_ATE, epochs=5)
        save_model_pkl(ate_model, "ate_model_v1.pkl")

        print("Starting to test ATE model...")
        truths, predictions = test_ate(ate_test_loader, ate_model)
        print(classification_report(truths, predictions, target_names=["Non-Aspect", "B-Term", "I-Term"]))

    # TRAIN/TEST ABSA IF CHOSEN
    if train_mode in ["2", "3"]:
        print("\nStarting to train ABSA model...")
        train_absa(absa_train_loader, absa_val_loader, absa_model, optimizer_ABSA, epochs=8)

        print("Starting to test ABSA model...")
        truths, predictions = test_absa(absa_test_loader, absa_model)
        print(classification_report(truths, predictions, target_names=["Negative", "Neutral", "Positive"]))

    if train_mode.lower() in ["q", "quit"]:
        print("Exiting...")
        sys.exit(0)