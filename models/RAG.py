import logging
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Okapi
from models.preprocess import preprocess_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def summarize_texts(texts, bm25, n=5):
    """
    Summarize the given texts using BM25.
    
    Args:
    texts (list of str): List of texts to summarize.
    bm25 (BM25Okapi): BM25 object for retrieval.
    n (int): Number of top sentences to include in the summary.
    
    Returns:
    list of str: Summarized texts.
    """
    summaries = []
    for text in texts:
        tokenized_sentences = [sentence.split(" ") for sentence in text.split(".")]
        bm25 = BM25Okapi(tokenized_sentences)
        summary = bm25.get_top_n(query.split(" "), tokenized_sentences, n=n)
        summaries.append(" ".join([" ".join(sent) for sent in summary]))
    return summaries

def retrieve_documents(query, bm25, texts, n=10):
    """
    Retrieve top n documents for the given query using BM25.
    
    Args:
    query (str): The query string.
    bm25 (BM25Okapi): BM25 object for retrieval.
    texts (list of str): List of texts from which to retrieve documents.
    n (int): Number of top documents to retrieve.
    
    Returns:
    tuple: Retrieved documents and their scores.
    """
    tokenized_query = query.split(" ")
    scores = bm25.get_scores(tokenized_query)
    top_n = scores.argsort()[-n:][::-1]
    return [texts[i] for i in top_n], scores[top_n]

def RAG_pipeline(query, texts, bm25, summarizer):
    """
    Perform the RAG (Retrieval-Augmented Generation) pipeline.
    
    Args:
    query (str): The query string.
    texts (list of str): List of texts to retrieve and summarize.
    bm25 (BM25Okapi): BM25 object for retrieval.
    summarizer (callable): Summarizer function.
    
    Returns:
    list of str: Summarized documents.
    """
    initial_docs, scores = retrieve_documents(query, bm25, texts)
    summarized_docs = summarizer(initial_docs, bm25)
    return summarized_docs

if __name__ == "__main__":
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    model = AutoModel.from_pretrained("google-bert/bert-base-uncased")
    
    try:
        # Load and preprocess data
        data = pd.read_csv("data/reviews_v1_hiring_task.csv", sep=",", encoding="latin-1").sample(500)
        df = data[["reviews.text", "reviews.rating", "reviews.date", "name"]]
        data = preprocess_text(data)
        texts = df["reviews.text"].tolist()
        
        # Tokenize the documents
        tokenized_corpus = [doc.split(" ") for doc in texts]
        bm25 = BM25Okapi(tokenized_corpus)
        
        # Define the query
        query = "What did users like about product AMAZON KINDLE"
        
        # Execute the RAG pipeline
        result = RAG_pipeline(query, texts, bm25, summarize_texts)
        logger.info(f"Summarized result: {result}")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
