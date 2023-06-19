import pickle
from matplotlib import pyplot as plt
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
from tqdm import tqdm
import data_loader

# =================== CONSTANTS ======================================
PATH_SECOND_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
PATH_FIRST_MODEL = "emilyalsentzer/Bio_Discharge_Summary_BERT"

PATH_TO_TRAINING_TOPICS = ".//lib//topics.xml"
PATH_TO_TRAINING_SAMPLES = ".//lib//samples.txt"

# ===================================================================
# =========================== PARAMETERS ============================

CHOSEN_MODEL_PATH = PATH_SECOND_MODEL

# ===================================================================
label_vocab = {"RELEVANT": 1, "IRRELEVANT": 0}

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


def get_training_data(generate_new_data=False):
    if generate_new_data:
        data_loader.generate_data(PATH_TO_TRAINING_SAMPLES, PATH_TO_TRAINING_TOPICS)
    try:
        training_data = pickle.load(open("lib/train_data.p", 'rb'))
    except FileNotFoundError:
        data_loader.generate_data(PATH_TO_TRAINING_SAMPLES, PATH_TO_TRAINING_TOPICS)
        training_data = pickle.load(open("lib/train_data.p", 'rb'))
    for data in training_data:
        if len(training_data[data].batches) == 0:
            training_data[data].generate_batches(num_of_batches=1)
    return training_data


def get_test_data(generate_new_data=False):
    if generate_new_data:
        data_loader.generate_data(PATH_TO_TRAINING_SAMPLES, PATH_TO_TRAINING_TOPICS)
    try:
        test_data = pickle.load(open("lib/test_data.p", 'rb'))
    except FileNotFoundError:
        data_loader.generate_data(PATH_TO_TRAINING_SAMPLES, PATH_TO_TRAINING_TOPICS)
        test_data = pickle.load(open("lib/test_data.p", 'rb'))
    for data in test_data:
        if len(test_data[data].batches) == 0:
            test_data[data].generate_batches(num_of_batches=1)
    return test_data


def train_model(training_data, epoches=10):
    # train model with triplet margin loss function
    triplet_loss = torch.nn.TripletMarginLoss(margin=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    for epoch in tqdm(range(epoches), desc="Training"):
        model.train()
        running_loss = 0
        i = 0
        for training_topic in training_data.values():
            i += 1
            optimizer.zero_grad()
            anchor = encode_text_to_low_dimention_space(training_topic.topic_query)
            positive = encode_text_to_low_dimention_space([data[1] for data in training_topic.batches])
            negative = encode_text_to_low_dimention_space([data[2] for data in training_topic.batches])

            loss = triplet_loss(anchor, positive, negative)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            # visualize the learning process for validation
            if i % 10 == 0:
                last_loss = running_loss / 10
                # print(' batche {} loss: {}'.format(i, last_loss))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }, "C:/Work/Ben/Retrieval_model/lib/checkpoints.pt")


def test(training_data, k=10):
    model.eval()
    top_k_scores = []
    inverse_scores = []
    for topic in training_data:
        top_k = []
        top_k_score = 0
        inverse_score = 0
        all_tested_abstracts_ids = training_data[topic].positive_ids + training_data[topic].negative_ids + \
                                   training_data[topic].natural_ids
        topic_code = encode_text_to_low_dimention_space(training_data[topic].topic_query)
        for abstract_id in all_tested_abstracts_ids:
            abstract_code = encode_text_to_low_dimention_space(data_loader.get_document(abstract_id).abstract)
            dist = get_cosine_similarity(topic_code, abstract_code)
            top_k.append((abstract_id, dist))
        top_k = sorted(top_k, key=lambda x: x[1], reverse=True)[:k]
        for abs_id, score in top_k:
            if abs_id in training_data[topic].positive_ids:
                top_k_score += 1
                continue
            if abs_id in training_data[topic].negative_ids:
                inverse_score += 1
                continue
        top_k_score /= float(len(top_k))
        inverse_score /= float(len(top_k))
        top_k_scores.append(top_k_score)
        inverse_scores.append(inverse_score)
        print(f"topic-{topic} acuracy {top_k_score}, inverse score {inverse_score}, length: {len(top_k)}")
    total_k_score = 100 * sum(top_k_scores) / len(top_k_scores)
    total_inverse_score = 100 * sum(inverse_scores) / len(top_k_scores)
    print(f"total top-k score = {total_k_score:.3f}%")
    print(f"total inverse top-k score = {total_inverse_score:.3f}%")
    with open("lib/result_archive.txt", "a") as f:
        f.write(f"{total_k_score},{total_inverse_score}\n")

    return total_k_score


def load_model_checkpoint():
    checkpoint = torch.load("C:/Work/Ben/Retrieval_model/lib/checkpoints.pt")
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


if __name__ == "__main__":
    load_model_checkpoint()
    query = "2-year-old boy with fever and irritability ,strawberry tongue, desquamation of the fingers"
    main(query,10,10000)
    """
    epochs_g = 20
    for i in range(50):
        train_model(get_training_data(), epoches=epochs_g)
        acr = test(get_test_data())
    """
