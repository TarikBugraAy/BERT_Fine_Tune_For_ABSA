# Aspect-term extraction and Aspect-based sentiment analysis
 
**Dependencies**
run ```chmod +x main.sh```
Run ```./main.sh``` to install the dependencies, to run training and result codes.

# Introduction



# Task description





# Dataset

In order to train and test our models we use a dataset containing restaurant reviews, taken from a preprocessed version, available at [1], of the SemEval-2014 ABSA Task [2].

Data are organized in a csv file, with the following columns:
- **Tokens**: tokenized sentence;
- **Tags**: list of tags associated to each token: '0' for non-aspect terms, '1' for beginning of terms and '2' for marks of terms;
- **Polarities**: list of polarities associated to each token: '0' for negative, '1' for neutral and '2' for positive and '-1' for non-aspect terms;

# Models







## Overview
The main notebook is structured as follows:
1. Text preprocessing and normalization;
2. Aspect term extraction: training overview and comparison, testing, evaluation, and visualization;
3. Aspect-based sentiment analysis:  training overview and comparison, testing, evaluation, and visualization.

Since we trained 8 different models, training has been performed separately and loaded in the notebook. Nevertheless, we show the process of training by means of a plot of the training loss.

## Results:


## References:

[1] **Aspect-Term-Extraction-and-Analysis**, https://github.com/1tangerine1day/Aspect-Term-Extraction-and-Analysis

[2] Maria Pontiki, Dimitris Galanis, John Pavlopoulos, Harris Papageorgiou, Ion Androutsopoulos, and Suresh Manandhar. 2014. **SemEval-2014 Task 4: Aspect Based Sentiment Analysis.** In Proceedings of the 8th International Workshop on Semantic Evaluation (SemEval 2014), pages 27–35, Dublin, Ireland. Association for Computational Linguistics.

[3] Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina, **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**, DOI: 10.48550/ARXIV.1810.04805

[4] **HuggingFace BERT models**, https://huggingface.co/docs/transformers/model_doc/bert

[5] **AdapterHub: A Framework for Adapting Transformers**, Jonas Pfeiffer et al., Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP 2020): Systems Demonstrations, https://www.aclweb.org/anthology/2020.emnlp-demos.7

[6] Houlsby, Neil and Giurgiu, Andrei and Jastrzebski, Stanislaw and Morrone, Bruna and de Laroussilhe, Quentin and Gesmundo, Andrea and Attariyan, Mona and Gelly, Sylvain, **Parameter-Efficient Transfer Learning for NLP**, DOI: 10.48550/ARXIV.1902.00751

[7] Loshchilov Ilya, Hutter Frank, **Decoupled Weight Decay Regularization**, DOI: 10.48550/ARXIV.1711.05101

[8] PyTorch: An Imperative Style, High-Performance Deep Learning Library}, Paszke Adam,  Gross Sam, Massa Francisco, Lerer Adamm, Bradbury James, Chanan Gregory, Killeen Trevor, Lin Zeming, Gimelshein Natalia, Antiga Luca, Desmaison Alban, Kopf Andreas, Yang Edward, DeVit Zachary, Raison Martin, Tejani Alykhan, Chilamkurthy Sasank, Steiner Benoit, Fang Lu, Bai Junjie, Chintala Soumith, http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf

## Aknowledgements

The general structure of the model (i.e. padding, dataset construction...) has been taken from [1], nevetheless we organized the model into a user-friendly class structure which provides a simple interface to the model. Moreover, we changed the optimization strategy, using AdamW instead of a generic Adam, introducing the learning rate scheduling and Adapter option as an alternative to fine-tuning.
Finally, we add a detailed analysis of the model performance, including the study of the training history, confusion matrix and visualization