# Text Analysis using Deep Learning Models

This project demonstrates the use of deep learning models for text analysis tasks, including web scraping, text processing, sentiment analysis, and model evaluation.

## Code Overview

The provided code performs the following tasks:

### 1. Web Scraping and Text Extraction

- Utilizes the `requests` library to retrieve HTML content from Arabic websites related to a specified topic.
- Employs `BeautifulSoup` for parsing HTML and extracting text from paragraphs (`<p>`) on the webpages.
- Defines the `scrape_text()` function to scrape text from a given URL.

### 2. Text Scoring

- Defines a function `calculate_text_score()` to assign scores to the extracted text based on certain criteria.
- Calculates the score for each extracted text, where the score is determined by the length of the text.

### 3. Data Preparation

- Scrapes text from each specified URL, calculates its score, and stores the text-score pairs in a dictionary.
- Tokenizes the extracted text using the `Tokenizer` class from Keras.
- Pads the sequences of tokenized text to ensure uniform length using `pad_sequences` function from Keras.
- Converts the scores to a numpy array.

### 4. Model Training and Evaluation

- Splits the dataset into training and test sets using `train_test_split` from `sklearn.model_selection`.
- Defines and trains three deep learning models for text analysis: RNN, Bidirectional RNN, and GRU using Keras.
- Compiles each model with appropriate loss and optimization functions.
- Evaluates each model's performance on the test set using mean squared error and mean absolute error metrics.

## Requirements

- Python 3.x
- requests
- BeautifulSoup
- Keras
- numpy

## Usage

1. Ensure that you have Python and the required libraries installed.
2. Update the list of URLs with Arabic websites related to your topic.
3. Run the provided Python script to scrape text, preprocess data, train deep learning models, and evaluate their performance.

```bash
python text_analysis.ipynb



# Joke Generation using GPT-2

This project showcases the generation of jokes using the GPT-2 model. GPT-2, developed by OpenAI, is a powerful language model capable of generating human-like text. In this project, we fine-tune the GPT-2 model on a dataset of jokes to create new, humorous content.

## Code Overview

The provided code in `transormer.ipynb` performs the following tasks:

### 1. Imports and Setup

- Imports necessary libraries including PyTorch and Transformers by Hugging Face.
- Sets up logging to suppress unnecessary messages and warnings.
- Determines whether to use CPU or GPU for computation.

### 2. Data Loading

- Defines a custom dataset class `JokesDataset` for loading jokes from a CSV file.
- Reads jokes from the provided CSV file, preprocesses them, and adds a special token ("JOKE:") to mark the start of each joke.
- Implements methods to handle dataset length and individual joke retrieval.

### 3. Model Initialization

- Initializes the GPT2 tokenizer using the `GPT2Tokenizer` class from the Transformers library. This tokenizer is specifically trained to handle GPT-2 model input and output.
- Loads the GPT2 language model (`GPT2LMHeadModel`) with the `gpt2-medium` pre-trained weights. This model is capable of generating text based on input prompts.

### 4. Training Configuration

- Sets up parameters essential for training, including batch size, number of epochs, learning rate, and warm-up steps.
- Defines the optimizer (AdamW) and scheduler to adjust the learning rate during training.

### 5. Training Loop

- Iterates over the dataset of jokes and constructs sequences for training the GPT-2 model.
- Trains the GPT-2 model by predicting the next token in the sequence based on the input tokens.
- Updates model parameters using backpropagation and optimization techniques.

### 6. Model Saving

- Saves the trained GPT-2 model after each epoch to the specified directory. This allows us to reload the model for later use without retraining.

### 7. Model Loading for Generation

- Loads the trained GPT-2 model from the saved checkpoint for generating new jokes.

### 8. Joke Generation

- Generates new jokes using the trained GPT-2 model.
- Iteratively predicts the next token in the joke sequence until the end token is encountered.
- Writes the generated jokes to a file for further analysis or usage.

## Requirements

- Python 3.x
- PyTorch
- Transformers library by Hugging Face

## Usage

1. Ensure that you have prepared your jokes dataset. By default, the dataset should be in the `jokes_data/` directory, containing a CSV file named `shortjokes.csv`.

2. Run the provided Python script `transormer.ipynb` to train the GPT-2 model on the jokes dataset and generate new jokes.

```bash
python transormer.ipynb
