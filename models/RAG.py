from transformers import AutoTokenizer, AutoModel
import torch
import faiss

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = AutoModel.from_pretrained("google-bert/bert-base-uncased")


texts= df['reviews.text']
def summarize_texts(texts, bm25, n=5):
    summaries = []
    for text in texts:
        tokenized_sentences = [sentence.split(" ") for sentence in text.split(".")]
        bm25 = BM25Okapi(tokenized_sentences)
        summary = bm25.get_top_n(query.split(" "), tokenized_sentences, n=n)
        summaries.append(" ".join([" ".join(sent) for sent in summary]))
    return summaries
