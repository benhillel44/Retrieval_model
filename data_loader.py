import random
import tarfile
import ir_datasets
import xml.etree.ElementTree as ET
import tqdm
import pickle

dataset = ir_datasets.load('pmc/v2/trec-cds-2016')
datastore = dataset.docs_store()


def get_document(doc_id):
    return datastore.get(doc_id)


def load_documents(num_of_documents):
    # load num_of_documents documents from ir_datasets and save them in a dictionary with the document id as key
    data_iter = dataset.docs_iter()
    documents = {}
    i = 0
    for doc in data_iter:
        documents[doc.doc_id] = doc
        i += 1
        if i == num_of_documents:
            break
    return documents


def documents_generator(num_of_documents):
    # load num_of_documents documents from ir_datasets and save them in a dictionary with the document id as key
    dataset = ir_datasets.load('pmc/v2/trec-cds-2016')
    i = 0
    for doc in dataset.docs_iter():
        yield doc
        i += 1
        if i == num_of_documents:
            break


"""
class TrainData:
    def __init__(self, topic_id, topic_query):
        self.topic_id = topic_id
        self.topic_query = topic_query
        self.positive_ids = []
        self.negative_ids = []
        self.natural_ids = []
        self.batches = []

    def get_random_batch(self):
        anchor = self.topic_query
        positive = random.choice(self.positive_ids)
        negative = random.choice(self.negative_ids)
        natural = random.choice(self.natural_ids)
        return anchor, positive, negative, natural

    def generate_batches(self, num_of_batches=3):
        self.batches = []
        for i in range(num_of_batches):
            self.batches.append(self.get_random_batch())


def get_topics(topics_file_path):
    topics_file = open(topics_file_path, 'r')
    topics = {}
    # read the topic xml and match the number to the summery
    tree = ET.parse(topics_file)
    root = tree.getroot()
    raw_topics = root.findall('topic')
    for topic in raw_topics:
        topic_id = topic.get('number')
        summary = topic.find('summary').text.lstrip()
        topics[topic_id] = TrainData(topic_id, summary)
    topics_file.close()
    return topics


def get_train_data_for_triplet(labeled_data_file, topics_file):
    data = get_topics(topics_file)
    dataset = ir_datasets.load('pmc/v2/trec-cds-2016')
    datastore = dataset.docs_store()
    with open(labeled_data_file, 'r') as file:
        for line in file.readlines():
            label, q_zero, article_id, rank, relevance = line.split()
            try:
                if datastore.get(article_id).abstract <= "":
                    continue
            except(KeyError):
                continue
            if int(relevance) >= 1:
                data[label].positive_ids.append(article_id)
            if int(relevance) == -1:
                data[label].negative_ids.append(article_id)
            if int(relevance) == 0:
                data[label].natural_ids.append(article_id)
    return data


def generate_data(labeled_data_file, topics_file, test_data_percent=0.1):
    data = get_topics(topics_file)
    dataset = ir_datasets.load('pmc/v2/trec-cds-2016')
    datastore = dataset.docs_store()
    with open(labeled_data_file, 'r') as file:
        for line in file.readlines():
            label, q_zero, article_id, rank, relevance = line.split()
            try:
                if datastore.get(article_id).abstract <= "":
                    continue
            except(KeyError):
                continue
            if int(relevance) >= 1:
                data[label].positive_ids.append(article_id)
            if int(relevance) == -1:
                data[label].negative_ids.append(article_id)
            if int(relevance) == 0:
                data[label].natural_ids.append(article_id)

    # take some x% of the data to be the test data
    test_data = {}
    for topic_id in data.keys():
        natural = []
        positive = []
        negative = []
        for i in range((int)(len(data[topic_id].natural_ids) * test_data_percent)):
            natural.append(data[topic_id].natural_ids.pop(0))

        for i in range((int)(len(data[topic_id].positive_ids) * test_data_percent)):
            positive.append(data[topic_id].positive_ids.pop(0))

        for i in range((int)(len(data[topic_id].negative_ids) * test_data_percent)):
            negative.append(data[topic_id].negative_ids.pop(0))

        test_data[topic_id] = TrainData(topic_id, data[topic_id].topic_query)
        test_data[topic_id].natural_ids = natural
        test_data[topic_id].positive_ids = positive
        test_data[topic_id].negative_ids = negative
        with open("lib/train_data.p", 'wb') as f:
            pickle.dump(data, f)
        with open("lib/test_data.p", 'wb') as f:
            pickle.dump(test_data, f)
    return data, test_data
"""
