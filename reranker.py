import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from tqdm import tqdm
import data_loader
import dense_retriever

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
model = AutoModelForSequenceClassification.from_pretrained(
    CHOSEN_MODEL_PATH,
    config=config,
)

def tokenize_data(query, abstract,tokenizer_p):


def train(training_data, epochs=10):
    model.train()
    # train model with triplet margin loss function
    triplet_loss = torch.nn.TripletMarginLoss(margin=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        running_loss = 0
        i = 0
        for training_topic in training_data.values():
            i += 1
            optimizer.zero_grad()
            training_batch = [(training_topic.topic_query, pos_id) for pos_id in training_topic.positive_ids]
            outputs = model(**tokenizer(training_batch))
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # visualize the learning process for validation
            if i % 10 == 0:
                last_loss = running_loss / 10
                print(' batche {} loss: {}'.format(i, last_loss))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }, "C:/Work/Ben/Retrieval_model/lib/checkpoints_reranker.pt")





train(dense_retriever.get_training_data())
