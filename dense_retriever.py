import torch
import pickle
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import data_loader


tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_Discharge_Summary_BERT", use_fast=True)
model = AutoModel.from_pretrained("emilyalsentzer/Bio_Discharge_Summary_BERT")
model.resize_token_embeddings(len(tokenizer))
model = model.cuda()

# set the device to cuda if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def encode_text_to_low_dimention_space(text):
    texts = tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
    texts = {x: y.cuda() for x, y in texts.items()}
    embed = model(**texts)
    embed = embed['last_hidden_state'][:, 0, :]
    return embed

def get_cosine_similarity(vec1, vec2):
    return torch.cosine_similarity(vec1, vec2).item()



def train_model():
    # train model with triplet margin loss function
    epoches = 10
    triplet_loss = torch.nn.TripletMarginLoss(margin=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    for epoch in range(epoches):
        model.train()
        anchor = encode_text_to_low_dimention_space()
        loss = triplet_loss()




def main(query, retrieval_amount, num_of_documents=1000):
    documents_iterator = data_loader.documents_generator(num_of_documents)
    # find top 5 documents by cosine similarity for a given query using dense retrieval and the document iterator
    best_score = 0
    top_n_docs = []
    query_vector = encode_text_to_low_dimention_space(query)
    for doc in tqdm(documents_iterator, desc="Computing cosine similarities"):
        doc_vector = encode_text_to_low_dimention_space(doc.abstract)
        score = get_cosine_similarity(query_vector, doc_vector)
        if score > best_score or len(top_n_docs)<retrieval_amount:
            top_n_docs.append((doc, score))
            top_n_docs = sorted(top_n_docs, key=lambda x: x[1], reverse=True)[:retrieval_amount]
            best_score = top_n_docs[-1][1]
    # Print the top 5 documents
    for i, doc in enumerate(top_n_docs):
        print("Rank ", i, ": ", doc[0].title, "(C-score: ", doc[1], ")")

    print("Top abstract:")
    print(top_n_docs[0][0].abstract)


