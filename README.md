# Multi-Label-Text-Classification

Multi-Label Text Classification using BERT and HuggingFace Transformers
## Overview
This project implements a multi-label text classification pipeline using a pre-trained BERT model fine-tuned with HuggingFace’s Trainer. The process encompasses robust data cleaning, multi-label transformation, tokenization, model training, and postprocessing to yield accurate predictions on product review texts.

## Dataset
### Input Dataset
Structure:
The input dataset comprises two columns:

Core Item: Contains raw product review texts.

Level 1 Factors: Represents the corresponding target label for each review.

### Size:
A total of 7744 rows.

## Predicted Results DataFrame
### Structure: 
The result DataFrame contains:
Core Item: Original product review texts.

Predicted Level 1 Factors: Predicted multi-label outputs (e.g., "Fragrance, Skin Care", "Brand Value").

### Size: 
The predictions are available for 127 rows.

## Project Structure
### Data Loading & Preprocessing:
Utilizes libraries such as Pandas, Numpy, re, and nltk to load and clean the textual data. Text preprocessing includes link removal, punctuation stripping, lowercasing, and stopword removal.

### Multi-Label Transformation:
Employs sklearn's MultiLabelBinarizer to convert categorical labels into a multi-hot encoded format.

### Model and Tokenization:
Implements BertTokenizerFast and BertForSequenceClassification from the transformers library for tokenizing input text and fine-tuning the classification model on multi-label data.

### Dataset Splitting:
Uses sklearn’s train_test_split to divide the data into training and validation sets.

### Training and Evaluation:
The HuggingFace Trainer API is used for training the model over 5 epochs. Model performance is evaluated using sklearn’s f1_score. A postprocessing step leverages numpy to ensure that every prediction contains at least one label.
