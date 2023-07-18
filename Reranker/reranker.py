import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from tqdm import tqdm
from Dense_retriever import dense_retriever

# =================== CONSTANTS ======================================
PATH_SECOND_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
PATH_FIRST_MODEL = "emilyalsentzer/Bio_Discharge_Summary_BERT"

PATH_TO_TRAINING_TOPICS = "../lib/topics.xml"
PATH_TO_TRAINING_SAMPLES = "../lib/samples.txt"

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


def train_model(model, tokenizer, training_data, id_to_query, epoch=0, epoches=10):
    # train model with triplet margin loss function
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
            # move the batch tensors to GPU
            gpu_batch = {x: y.cuda() for x, y in batch.items()}
            output = model(**gpu_batch)  # unpack the batch
            # the learning - cross entropy loss function
            loss = output.loss
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step(epoch)  # update the learning rates
            # visualize the learning process for validation
            if step % 10 == 9:
                last_loss = running_loss / 10
                print(' batche {} loss: {}'.format(step, last_loss))
        running_loss /= len(training_data)
        print('Training loss after epoch {}: {}'.format(epoch, running_loss))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': running_loss
    }, "../lib/checkpoint_reranker.pt")


train(dense_retriever.get_training_data())
