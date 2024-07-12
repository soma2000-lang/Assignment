from transformers import AutoTokenizer, AutoModel
import torch
import faiss
from rank_bm25 import BM25Okapi
import logging

import pandas as pd
from models.preprocess import preprocess_text
def summarize_texts(texts, bm25, n=5):
    summaries = []
    for text in texts:
        tokenized_sentences = [sentence.split(" ") for sentence in text.split(".")]
        bm25 = BM25Okapi(tokenized_sentences)
        summary = bm25.get_top_n(query.split(" "), tokenized_sentences, n=n)
        summaries.append(" ".join([" ".join(sent) for sent in summary]))

    return summaries





# Function to retrieve documents
def retrieve_documents(query, bm25, n=10):
    tokenized_query = query.split(" ")
    scores = bm25.get_scores(tokenized_query)
    top_n = scores.argsort()[-n:][::-1]
    return [texts[i] for i in top_n], scores[top_n]


def RAG_pipeline(query, texts, bm25, summarizer):
    initial_docs, scores = retrieve_documents(query, bm25)
    summarized_docs = summarize_texts(texts, bm25)
    return summarized_docs


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    model = AutoModel.from_pretrained("google-bert/bert-base-uncased")
    data = pd.read_csv(
        "data/reviews_v1_hiring_task.csv", sep=",", encoding="latin-1"
    ).sample(500)
    df = data[
        ["reviews.text", "reviews.rating", "reviews.date", "name"]
    ]
    data = preprocess_text(data)
    texts = df["reviews.text"]
    # Tokenize your documents
    tokenized_corpus = [doc.split(" ") for doc in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    query = "What did users like about product AMAZON KINDLE"
    result = RAG_pipeline(query, texts, bm25, bm25)
    print(result)


  