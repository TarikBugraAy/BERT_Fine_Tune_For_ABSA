# Aspect-Term Extraction and Aspect-Based Sentiment Analysis (ABSA)

## Overview
This project implements an **Aspect-Based Sentiment Analysis (ABSA)** pipeline using fine-tuned **bert-large-uncased**[1] model. It extracts aspect terms from text (e.g., "sevice," "food", "ambiance") and determines their sentiment polarity (Positive, Neutral, or Negative). The pipeline consists of two primary components:

1. **Aspect Term Extraction (ATE):** Identifies aspect terms within sentences.
2. **Aspect-Based Sentiment Analysis (ABSA):** Determines the sentiment polarity of extracted aspect terms.

The project employs pre-trained BERT models fine-tuned for these tasks and provides testing.

## Features
- Fine-tuned BERT models for ATE and ABSA tasks.
- User-friendly training and testing scripts.
- Single and multi processing modes for predictions.
- Visualization of  performance using classification reports and confusion matrices.

---

## Dependencies
Install the dependencies by running the following command:

```bash
chmod +x main.sh
./main.sh
```
This script installs all necessary Python libraries listed in `requirements.txt`.

### Key Libraries:
- Python: 3.11.9
- PyTorch: 2.5.1
- Transformers: 4.45.2
- Pandas: 2.2.3
- NumPy: 1.26.4
- scikit-learn: 1.6.0

---

## Task Description
**Aspect-Based Sentiment Analysis (ABSA)** aims to provide  sentiment analysis at the aspect level. Given a sentence, the goal is to identify aspects and classify the sentiment polarity associated with each aspect.

### Example:
**Input:** "The food was delicious, but the service was slow."

**Output:**
- Aspect: "food" - Sentiment: Positive
- Aspect: "service" - Sentiment: Negative

---

## Dataset
The models are trained and tested using the restaurant reviews dataset from a preprocessed version of the SemEval-2014 ABSA Task dataset.Preprocesed version available at [2], of the SemEval-2014 ABSA Task [3]

### Dataset Format:
The dataset is a CSV file with the following columns:
- **Tokens:** Tokenized sentence.
- **Tags:** BIO tagging for aspect terms (`0`: Non-aspect, `1`: Beginning of aspect term, `2`: Inside aspect term).
- **Polarities:** Sentiment polarities (`0`: Negative, `1`: Neutral, `2`: Positive, `-1`: Non-aspect terms).

### Dataset Files:
- **ABSArestaurants_train.csv:** Balanced version of `restaurants_train.csv` used in ABSA training.
- **restaurants_test.csv:** Used in testing both ATE and ABSA for generating classification reports and confusion matrices.
- **restaurants_train.csv:** Used in training the ATE model.
- **restaurants_val.csv:** Balanced version of `restaurants_test.csv` used for validating ABSA during training.
- **testing.csv:** Generated using GPT-4o with a mix of sentences for user testing purposes.

---

## Models
### Aspect Term Extraction (ATE):
- Model: Fine-tuned BERT
- Output: BIO tags for aspect terms.

### Aspect-Based Sentiment Analysis (ABSA):
- Model: Fine-tuned BERT
- Output: Sentiment polarity for each extracted aspect term.

---

## Implementation
### Structure:
- **Training:** Implemented in `train.py` with separate pipelines for ATE and ABSA.
- **Testing:** Implemented in `test.py`, generating classification reports and confusion matrices.
- **Inference:** Real-time single-sentence analysis (`result_single.py`) and batch analysis from CSV (`results_multi.py`).
- **UX Integration:** The project can be run using `main.sh`, which provides an interactive interface to manage the pipeline phases (installation, training, testing, and results).

### Pipeline:
1. **Text Preprocessing:** Tokenization and BIO tagging.
2. **Model Training:** Fine-tuning BERT for ATE and ABSA tasks.
3. **Evaluation:** Classification reports and visualization.
4. **Inference:** Predict aspect terms and polarities from user input.

---

## Results
### Aspect Term Extraction (ATE):
```
              precision    recall  f1-score   support

  Non-Aspect       0.99      0.99      0.99     32262
      B-Term       0.89      0.89      0.89      4022
      I-Term       0.86      0.80      0.83      2141

    accuracy                           0.97     38425
   macro avg       0.91      0.90      0.90     38425
weighted avg       0.97      0.97      0.97     38425
```

### Aspect-Based Sentiment Analysis (ABSA):
```
              precision    recall  f1-score   support

    Negative       0.70      0.78      0.73       196
     Neutral       0.53      0.53      0.53       196
    Positive       0.91      0.88      0.90       727

    accuracy                           0.80      1119
   macro avg       0.71      0.73      0.72      1119
weighted avg       0.81      0.80      0.80      1119
```

ATE performs nearly excellently, but ABSA has significant room for improvement and does not yet meet the desired performance.


---

## Usage
Run the project interactively using the `main.sh` script:

```bash
./main.sh
```

This script provides options to:
1. Install dependencies.
2. Train models (ATE, ABSA, or both).
3. Test models (pre-trained or user-trained).
4. View results (batch or single-sentence inference).

Follow the on-screen instructions to navigate through the options.

---

## References
1. **bert-large-uncased:** “google-bert/bert-large-uncased · Hugging Face,” Huggingface.co, Mar. 11, 2024. https://huggingface.co/google-bert/bert-large-uncased (accessed Jan. 04, 2025).
2. **Aspect-Term-Extraction-and-Analysis:** https://github.com/1tangerine1day/Aspect-Term-Extraction-and-Analysis
3. **SemEval-2014 Task 4:** C. Chugh, “SemEval 2014 Task 4: AspectBasedSentimentAnalysis,” Kaggle.com, 2022. https://www.kaggle.com/datasets/charitarth/semeval-2014-task-4-aspectbasedsentimentanalysis.
‌

---

## Acknowledgements
While initially inspired by [1], this project has undergone significant modifications and is largely different.
---

## Fine Tuned Models
[You can download fine tuned models from here.](https://drive.google.com/drive/folders/1N7dg0kDJlknSpdYs7nfs8dK4yu7cCS5U?usp=sharing)

---

## Contact
For questions or issues, please reach out to the project contributors or open an issue on the GitHub repository.
