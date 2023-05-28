import torch
import pickle
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import data_loader

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_Discharge_Summary_BERT", use_fast=True)
model = AutoModel.from_pretrained("emilyalsentzer/Bio_Discharge_Summary_BERT")

query = ""

# Load the documents from the pickle file
documents_iterator = data_loader.documents_generator(10)
documents_dict = data_loader.load_documents(1000)

# set the device to cuda if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("computing results with:", device)
model.to(device)


def encode_text_to_low_dimention_space(text):
    input_ids = tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
    texts = {x: y for x, y in input_ids.items()}
    outputs = model(**texts)
    return outputs['last_hidden_state'][:, 0, :]


def generate_low_dimention_space(creat_new=False):
    # if the low dimention space is already generated, load it from the pickle file
    if not creat_new:
        try:
            with open('lib/low_dimention_space.pickle', 'rb') as handle:
                low_dimention_space = pickle.load(handle)
            return low_dimention_space
        except FileNotFoundError:
            pass
    low_dimention_space = {}
    for doc in tqdm(documents_dict.values(), desc="Generating low dimention space"):
        low_dimention_space[doc.doc_id] = encode_text_to_low_dimention_space(doc.abstract)
    # save the dictionary in a pickle file
    with open('lib/low_dimention_space.pickle', 'wb') as handle:
        pickle.dump(low_dimention_space, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return low_dimention_space


def get_cosine_similarity(vec1, vec2):
    return torch.cosine_similarity(vec1, vec2).item()


def train_model():
    pass


def __main__():
    low_dimention_space = generate_low_dimention_space()
    query_vector = encode_text_to_low_dimention_space(query)
    scores = {}
    for doc in tqdm(low_dimention_space, desc="Computing cosine similarities"):
        scores[doc] = torch.cosine_similarity(query_vector, low_dimention_space[doc]).item()

    # Sort the documents by their cosine similarity with the query
    top_k_articles = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]

    # Print the top 5 documents
    for doc in top_k_articles:
        print(documents_dict[doc[0]].title)
        print("Cosine similarity score: ", doc[1])
        print("---------------------------------------------------")

def main():
    # find top 5 documents by cosine similarity for a given query using dense retrieval and the document iterator
    best_score = 0
    top_5_docs = []
    query_vector = encode_text_to_low_dimention_space(query)
    for doc in tqdm(documents_iterator, desc="Computing cosine similarities"):
        doc_vector = encode_text_to_low_dimention_space(doc.abstract)
        score = get_cosine_similarity(query_vector, doc_vector)
        if score > best_score:
            top_5_docs.append((doc, score))
            top_5_docs = sorted(top_5_docs, key=lambda x: x[1], reverse=True)[:5]
            best_score = top_5_docs[-1][1]
    # Print the top 5 documents
    for doc in top_5_docs:
        print(doc[0].title)
        print("Cosine similarity score: ", doc[1])
        print("---------------------------------------------------")



main()
