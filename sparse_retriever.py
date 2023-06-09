
from collections import Counter
import math
from tqdm import tqdm
import time
import pickle

import data_loader


def main(query, retrieval_amount, num_of_documents=1000):
    documents_dict = data_loader.load_documents(num_of_documents)
    documents = []
    for doc in documents_dict.values():
        documents.append(doc)

    # create a list of all tokens in documents
    docs_data = [(doc.title, doc.abstract.lower().split(), int(doc.doc_id)) for doc in documents]

    # create a list of all tokens in the query
    query_tokens = query.lower().split()

    # compute the document frequency of each token in the query
    df = Counter()
    for elem in docs_data:
        df.update(elem[1])

    # compute the inverse document frequency of each token in the query
    idf = {}
    for token in query_tokens:
        idf[token] = math.log(len(documents) / (df[token] + 1))
    for elem in docs_data:
        for token in elem[1]:
            idf[token] = math.log(len(documents) / (df[token] + 1))

    # compute the tf-idf of each token in the documents
    tf_idf = {}
    for elem in docs_data:
        tf = Counter(elem[1])
        tf_idf[elem[2]] = {}
        for term in tf.keys():
            if (term in tf.keys()) and (term in idf.keys()):
                tf_idf[elem[2]][term] = tf[term] * idf[term]

    # compute the tf-idf of each token in the query
    query_tf = Counter(query_tokens)
    query_tf_idf = {term: query_tf[term] * idf[term] for term in query_tf}

    # Compute the cosine similarity between the query vector and the document vectors
    scores = []
    for doc in tqdm(tf_idf, desc="Computing cosine similarities"):
        dot_product = 0
        doc_norm = 0
        query_norm = 0
        for term in query_tf_idf:
            if term in tf_idf[doc]:
                dot_product += tf_idf[doc][term] * query_tf_idf[term]
                doc_norm += tf_idf[doc][term] ** 2
            query_norm += query_tf_idf[term] ** 2
        score = dot_product / (math.sqrt(doc_norm + 1) * math.sqrt(query_norm + 1))
        scores.append(score)
        time.sleep(0.001)  # Simulate computation time

    # Get the top 5 documents with the highest similarity scores
    top_n_docs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:retrieval_amount]

    # Print the top 5 documents and their scores
    for i, doc_idx in enumerate(top_n_docs):
        doc = documents[doc_idx]
        print(f"Rank {i + 1}: {doc.title} (score: {scores[doc_idx]:.2f})")

    # print the first document abstract
    print(f"\nAbstract of the first document:\n{documents[top_n_docs[0]].abstract}")

