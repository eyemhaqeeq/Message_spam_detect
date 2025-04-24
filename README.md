- Message Spam Detection – NLP Project
- Overview
This project uses Natural Language Processing (NLP) techniques to classify SMS messages as Spam or Ham (Not Spam). The aim is to demonstrate practical NLP skills such as text preprocessing, feature extraction, and model training on real-world data.

- Features
Comprehensive text cleaning pipeline:

Lowercasing

Tokenization

Stopwords removal

Stemming

Lemmatization

Applied classic Bag of Words (BoW) model for vectorization (more to come!)

Trained using Naive Bayes classifier

Dataset split into training and test sets

Results preview and comparison (original vs cleaned text)

- NLP Techniques Used
Tokenization

Stopwords Removal

Stemming (PorterStemmer)

Lemmatization (WordNetLemmatizer)

Bag of Words

Text Classification with Multinomial Naive Bayes

-Dataset
SMS Spam Collection Dataset

UCI Machine Learning Repository

Contains 5,000+ labeled SMS messages

Two classes: spam or ham

-Tools & Libraries
Python

Pandas, NumPy

Scikit-learn (train_test_split, MultinomialNB)

NLTK (stopwords, tokenize, stem, lemmatize)

Matplotlib & Seaborn (for visualizations – optional)

Jupyter Notebook

Project Structure
bash
Copy
Edit
├── SMS_Spam_Detection_Enhanced.ipynb   # Main notebook
├── README.md                           # Project overview
└── requirements.txt                    # Libraries used (optional)
📈 What's Next?
Add TF-IDF and Word2Vec feature extraction

Compare model performance across different features

Add confusion matrix, precision, recall, and F1-score

Build a Streamlit web app for live demo

Export as .py script for automation

How to Run?
bash
Copy
Edit
# Clone this repo
git clone https://github.com/your-username/sms-spam-nlp.git

# Install requirements
pip install -r requirements.txt

# Run the notebook
jupyter notebook SMS_Spam_Detection_Enhanced.ipynb

Author
*Haqeeq Ahmed*
NLP Enthusiast
Masters in Computer Science & Technology with Business Development
LinkedIn: Haqeeq Ahmed(https://www.linkedin.com/in/haqeeq-ahmed-2b1412179/) | GitHub eyemhaqeeq(https://github.com/eyemhaqeeq)
