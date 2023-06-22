import pickle
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from tqdm import tqdm
import data_loader
import statistics
import dense_data_loader

# =================== CONSTANTS ======================================
PATH_SECOND_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
PATH_FIRST_MODEL = "emilyalsentzer/Bio_Discharge_Summary_BERT"

BASE_DIR = "../lib/"

PATH_TO_TRAINING_TOPICS = BASE_DIR + "topics.xml"
PATH_TO_TRAINING_SAMPLES = BASE_DIR + "samples.txt"

# ===================================================================
# =========================== PARAMETERS ============================

CHOSEN_MODEL_PATH = PATH_FIRST_MODEL

# ===================================================================
label_vocab = {"RELEVANT": 1, "IRRELEVANT": 0}


def encode_text_to_low_dimention_space(model, tokenizer, text):
    texts = tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
    texts = {x: y.cuda() for x, y in texts.items()}
    embed = model(**texts)
    embed = embed['last_hidden_state'][:, 0, :]
    return embed


def get_cosine_similarity(vec1, vec2):
    return torch.cosine_similarity(vec1, vec2).item()


def get_training_data(generate_new_data=False):
    if generate_new_data:
        data_loader.generate_data(PATH_TO_TRAINING_SAMPLES, PATH_TO_TRAINING_TOPICS)
    try:
        training_data = pickle.load(open("../lib/train_data.p", 'rb'))
    except FileNotFoundError:
        data_loader.generate_data(PATH_TO_TRAINING_SAMPLES, PATH_TO_TRAINING_TOPICS)
        training_data = pickle.load(open("../lib/train_data.p", 'rb'))
    for data in training_data:
        if len(training_data[data].batches) == 0:
            training_data[data].generate_batches(num_of_batches=1)
    return training_data


def get_test_data(generate_new_data=False):
    if generate_new_data:
        data_loader.generate_data(PATH_TO_TRAINING_SAMPLES, PATH_TO_TRAINING_TOPICS)
    try:
        test_data = pickle.load(open("../lib/test_data.p", 'rb'))
    except FileNotFoundError:
        data_loader.generate_data(PATH_TO_TRAINING_SAMPLES, PATH_TO_TRAINING_TOPICS)
        test_data = pickle.load(open("../lib/test_data.p", 'rb'))
    for data in test_data:
        if len(test_data[data].batches) == 0:
            test_data[data].generate_batches(num_of_batches=1)
    return test_data


def load_model_checkpoint(model):
    checkpoint = torch.load("../lib/checkpoints.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    return epoch


def main(query, retrieval_amount=5, num_of_documents=1000):
    documents_iterator = data_loader.documents_generator(num_of_documents)
    # find top 5 documents by cosine similarity for a given query using dense retrieval and the document iterator
    best_score = 0
    top_n_docs = []
    query_vector = encode_text_to_low_dimention_space(query)
    for doc in tqdm(documents_iterator, desc="Computing cosine similarities"):
        doc_vector = encode_text_to_low_dimention_space(doc.abstract)
        score = get_cosine_similarity(query_vector, doc_vector)
        if score > best_score or len(top_n_docs) < retrieval_amount:
            top_n_docs.append((doc, score))
            top_n_docs = sorted(top_n_docs, key=lambda x: x[1], reverse=True)[:retrieval_amount]
            best_score = top_n_docs[-1][1]
    # Print the top 5 documents
    for i, doc in enumerate(top_n_docs):
        print("Rank ", i, ": ", doc[0].title, "(C-score: ", doc[1], ")")

    print("Top abstract:")
    print(top_n_docs[0][0].abstract)


# ==========================================================================================================

def test(model, tokenizer, test_data, id_to_query, epoch=0, k=10):
    model.eval()
    dist = nn.PairwiseDistance(p=2.0)
    model_predictions = {}
    for batch in test_data:
        queries_encoded = encode_text_to_low_dimention_space(model, tokenizer, [id_to_query[x.query_id] for x in batch])
        articles_encoded = encode_text_to_low_dimention_space(model, tokenizer,
                                                              [data_loader.get_document(x.document_id) for x in batch])
        # list of distances between pairs of query-abstract from the encodings
        distances = dist(queries_encoded, articles_encoded).detach().cpu().numpy().tolist()

        # add the model prediction (the distance between the query and article) to a dictionary
        for i, sample in enumerate(batch):
            qid, aid, rel = sample
            if qid not in model_predictions:
                model_predictions[qid] = []
            model_predictions[qid].append((aid, distances[i], rel))

    # compute top-k score for the predictions
    scores_per_topic = compute_precision_at_k(model_predictions, k=k)
    score_list = list(scores_per_topic.values())
    mean_preck = sum(score_list) / len(score_list)
    median_preck = statistics.median(score_list)
    print('------------------Precision@K Scores for Epoch {}-------------------'.format(epoch))
    print('Mean Precision@K: {}'.format(mean_preck))
    print('Median Precision@K: {}'.format(median_preck))

    return mean_preck


def compute_precision_at_k(preds, k=10):
    precision = {}
    for topic in preds.keys():
        cur_ordered_preds = list(sorted(preds[topic], key=lambda x: x[1]))
        cur_prec = 0.0
        # count how many of the top k rated documents are actually relevant
        for chosen_id in cur_ordered_preds[:k]:
            cur_prec += 1 if chosen_id[-1] >= 1 else 0
        cur_prec /= float(k)
        precision[topic] = cur_prec
    return precision


def train_model(model, tokenizer, training_data, id_to_query, epoch=0, epoches=10):
    # train model with triplet margin loss function
    triplet_loss = torch.nn.TripletMarginLoss(margin=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    step = 0
    running_loss = 0.0
    for epoch in tqdm(range(epoches), desc="Training"):
        model.train()
        running_loss = 0.0
        for batch in training_data:
            step += 1
            optimizer.zero_grad()
            anchor = encode_text_to_low_dimention_space([id_to_query[x.query_id] for x in batch])

            positive = encode_text_to_low_dimention_space(
                [data_loader.get_document(data.positive_id) for data in batch])

            negative = encode_text_to_low_dimention_space(
                [data_loader.get_document(data.negative_id) for data in batch])

            loss = triplet_loss(anchor, positive, negative)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            # visualize the learning process for validation
            if step % 10 == 9:
                last_loss = running_loss / 10
                # print(' batche {} loss: {}'.format(i, last_loss))
        running_loss /= len(training_data)
        print('Training loss after epoch {}: {}'.format(epoch, running_loss))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': running_loss
    }, "../lib/checkpoint.pt")


def initialize():
    config = AutoConfig.from_pretrained(
        CHOSEN_MODEL_PATH,
        num_labels=len(list(label_vocab.keys())),
        label2id=label_vocab,
        id2label={i: l for l, i in label_vocab.items()},
    )
    tokenizer = AutoTokenizer.from_pretrained(
        CHOSEN_MODEL_PATH,
        use_fast=True,
    )
    model = AutoModel.from_pretrained(
        CHOSEN_MODEL_PATH,
        config=config,
    )

    model.resize_token_embeddings(len(tokenizer))

    # set the device to cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    load_model_checkpoint(model)

    # generate data
    trec_data = dense_data_loader.TrecDataset(PATH_TO_TRAINING_TOPICS, PATH_TO_TRAINING_SAMPLES)

    return model, tokenizer, trec_data


if __name__ == "__main__":
    _model, _tokenizer, trec_data = initialize()

    epochs_g = 20
    train_model(_model, _tokenizer, trec_data.get_training_data(), trec_data.topics)
