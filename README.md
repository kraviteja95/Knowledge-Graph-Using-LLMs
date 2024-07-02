# Knowledge Extraction and Graph Construction

This code extracts knowledge from the provided AWS S3 documentation URL, performs Named Entity Recognition (NER), identifies relationships between entities, generates insert queries, and constructs a knowledge graph. The code utilizes various natural language processing (NLP) tools and models.

## Approach Used to Solve

1. **Text Extraction:** The script extracts text content from a specified URL using BeautifulSoup and requests. Only the first 25 lines of text are considered.

2. **Text Preprocessing:** The extracted text is preprocessed using NLTK for sentence segmentation, tokenization, lowercasing, stopword removal, lemmatization, and custom cleaning. This results in a clean and processed text.

3. **Named Entity Recognition (NER):** The script uses a pre-trained BERT model for token classification to perform NER. Entities such as persons, organizations, and locations are extracted from the preprocessed text.

4. **Relationship Identification:** Relationships between entities are identified by analyzing the context within a specified window size. Relationships are established based on the proximity of entities within the text.

5. **Insert Query Generation:** GPT-2 is employed to generate insert queries based on the identified entities and relationships. The script utilizes a combination of entities and relationships to create meaningful queries.

6. **Knowledge Graph Construction:** A knowledge graph is constructed using NetworkX to represent entities as nodes, relationships as edges, and insert queries as node attributes. The graph is visualized using Matplotlib in a circular layout.

7. **Accuracy, Quality, and Relevance Calculation:** The script dynamically generates ground truth entities and relationships and calculates accuracy, quality, and relevance based on the NER and relationship identification results.

## Tools Used

- BeautifulSoup: For HTML parsing
- requests: For fetching content from URLs
- NLTK: For Natural Language Processing tasks
- Models used
  - **BERT (Bidirectional Encoder Representations from Transformers):**
    - **Model:** `BertForTokenClassification`
    - **Tokenizer:** `BertTokenizer`
    - **Usage:** Named Entity Recognition (NER)
    
  - **GPT-2 (Generative Pre-trained Transformer 2):**
    - **Model:** `GPT2LMHeadModel`
    - **Tokenizer:** `GPT2Tokenizer`
    - **Usage:** Insert query generation and language modeling
- NetworkX: For graph construction
- Matplotlib: For graph visualization

## Assumptions

- The relevant text on the webpage is enclosed in <p> tags.
- The maximum token length for BERT model input is set to 256.
- The script assumes a certain window size for relationship identification; users can adjust this based on their requirements.
- Considering the OpenAI API key constraints, I have used GPT-2 for insert query generation, and users can explore other variants of GPT-2 for experimentation as applicable.
- The script assumes a circular layout for visualizing the knowledge graph due to the external DB constraints. If required, users can consider any other layout available in networkx or can make use of external DBs.

## Achieved Accuracy, Quality and Relevance

1. Accuracy - 0.0
2. Quality - 1.0
3. Relevance - 1.0

## Usage

1. Install the required libraries
2. Import the corresponding modules from them
3. Download NLTK resources

```bash
pip install beautifulsoup4 requests nltk transformers networkx matplotlib
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer, BertForTokenClassification, GPT2LMHeadModel, GPT2Tokenizer
import torch
import gc
import networkx as nx
import matplotlib.pyplot as plt


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')