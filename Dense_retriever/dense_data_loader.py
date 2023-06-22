import random
import ir_datasets
import xml.etree.ElementTree as ET
import pickle
from copy import deepcopy
import data_loader

dataset = ir_datasets.load('pmc/v2/trec-cds-2016')
datastore = dataset.docs_store()

# constants
RELEVANT = '1'
IRRELEVANT = '0'


class TestDataTriplet:
    def __init__(self, query_id, document_id, relevance):
        self.query_id = query_id
        self.document_id = document_id
        self.relevance = relevance


class TrainDataTriplet:
    def __init__(self, query_id, pos_id, neg_id):
        self.query_id = query_id
        self.positive_id = pos_id
        self.negative_id = neg_id


class TrecDataset:
    PICKLE_PATH_TO_TOPICS = "../lib/topics_to_id.p"
    PICKLE_PATH_TO_SAMPLES = "../lib/trec_samples.p"

    # dataset to hold data from trec 2016
    # with intention of training and testing the model
    def __init__(self, topics_file_path, samples_file_path, test_p=0.1, batch_size=5):
        self.topics = self.__load_topics(topics_file_path)
        self.samples = self.__load_samples(samples_file_path)
        self.train_data, self.test_data = self.__generate_training_and_testing_data(test_p=test_p)
        self.training_batches = self.__generate_train_batches(batch_size=batch_size)
        self.testing_batches = self.__generate_test_batches(batch_size=batch_size)

    # load the topics (numerated queries to which the samples are linked)
    # for TRAINING the model
    # TODO: add mesh terms filter to the topics
    def __load_topics(self, topics_file_path):
        """
        :param topics_file_path: relative path to the file containing the topics
        """
        topics_ids_to_query = {}
        try:
            topics_ids_to_query = pickle.load(open(self.PICKLE_PATH_TO_TOPICS, 'rb'))
        except FileNotFoundError:
            topics_file = open(topics_file_path, 'r')
            tree = ET.parse(topics_file)
            root = tree.getroot()
            raw_topics = root.findall('topic')
            for topic in raw_topics:
                topic_id = topic.get('number')
                summary = topic.find('summary').text.lstrip()
                topics_ids_to_query[topic_id] = summary
            topics_file.close()
            with open(self.PICKLE_PATH_TO_TOPICS, 'wb') as f:
                pickle.dump(topics_ids_to_query, f)
        return topics_ids_to_query

    def __load_samples(self, samples_file_path):
        """
        :param samples_file_path:
        :return the returned dictionary is of the form
                 - dict[topic id: __]['1'] = [#all relevant article ids]
                 - dict[topic id: __]['0'] = [#all irrelevant article ids]
        """
        topic_id_to_samples = {}
        try:
            topic_id_to_samples = pickle.load(open(self.PICKLE_PATH_TO_SAMPLES, 'rb'))
        except FileNotFoundError:
            samples_file = open(samples_file_path, 'r')
            for line in samples_file.readlines():
                label_id, q_zero, article_id, rank, relevance = line.split()
                if label_id not in topic_id_to_samples.keys():
                    topic_id_to_samples[label_id] = {'1': [], '0': []}
                topic_id_to_samples[label_id]['1' if relevance >= 1 else '0'].append(article_id)

            with open(self.PICKLE_PATH_TO_SAMPLES, 'wb') as f:
                pickle.dump(topic_id_to_samples, f)
        return topic_id_to_samples

    def __generate_training_and_testing_data(self, test_p=0.1):
        """
        :param test_p: what part of all labeled data will go to testing (default 0.1 = 10%)
        :return: (train_data, test_data) in form of dict[topic_id:__]['1'] = [all relevant abstracts]
        """
        train_data = {}
        test_data = {}
        for topic_id in self.samples.keys():
            # run over all relevant ids and take the top %test_p to the test archive
            test_data[topic_id] = {'1': [], '0': [], 'topic': self.topics[topic_id]}
            train_data[topic_id] = {'1': [], '0': [], 'topic': self.topics[topic_id]}
            # all relevant ids
            for i, abstract_id in enumerate(self.samples[topic_id]['1']):
                if i < len(self.samples[topic_id]['1'])*test_p:
                    test_data[topic_id]['1'].append(abstract_id)
                else:
                    train_data[topic_id]['1'].append(abstract_id)
            # all irrelevant ids
            for i, abstract_id in enumerate(self.samples[topic_id]['0']):
                if i < len(self.samples[topic_id]['0'])*test_p:
                    test_data[topic_id]['0'].append(abstract_id)
                else:
                    train_data[topic_id]['0'].append(abstract_id)
        return train_data, test_data

    def __generate_train_batches(self, batch_size=5):
        """
        :return: a random set of batches of the form (query_id,positive_abstract_id,negative_abstract_id)
        """
        data_triplets = []
        batches = []
        for query_id in self.train_data:
            pos_ids = self.train_data[query_id]['1']
            neg_ids = self.train_data[query_id]['0']
            random.shuffle(pos_ids)
            random.shuffle(neg_ids)

            for i in range(min(len(pos_ids), len(neg_ids))):
                data_triplets.append(TrainDataTriplet(query_id, pos_ids[i], neg_ids[i]))
        batch = []
        for i, triplet in enumerate(data_triplets):
            batch.append(triplet)
            if i % batch_size == batch_size - 1:
                batches.append(deepcopy(batch))
                batch = []
        return batches

    def __generate_test_batches(self, batch_size=5):
        """
        :param batch_size:
        :returns list of batches of size 'batch_size', each containing triplets of the form (
                query_id, abstract_id,relevence)
        :return: note that relevance == '1' if the article is relevant, else '0'
                 batches always have the same query_id!
        """
        data = {query_id: [] for query_id in self.train_data}
        batches = []
        for query_id in self.train_data:
            for abstract_id in self.train_data[query_id]['1']:
                #            query id, abstract id , relevance ( '1' = relevant)
                data[query_id].append((query_id, abstract_id, '1'))
            for abstract_id in self.train_data[query_id]['0']:
                #            query id, abstract id , relevance ( '0' = irrelevant)
                data[query_id].append(TestDataTriplet(query_id, abstract_id, '0'))

        batch = []
        for topic in data.keys():
            for i, data_triplet in enumerate(data[topic]):
                batch.append(data_triplet)
                if i % batch_size == batch_size - 1:
                    batches.append(deepcopy(batch))
                    batch = []
            if len(batch) >= 1:
                batches.append(deepcopy(batch))
                batch = []
        return batches

    # getters and setters ===========================

    def get_query(self, query_id):
        return self.topics[query_id]

    def get_document(self, document_id):
        return data_loader.get_document(document_id)

    def get_training_data(self, batch_size, generate_new=False):
        if generate_new:
            self.training_batches = self.__generate_train_batches(batch_size=batch_size)
        return self.training_batches

    def get_testing_data(self, batch_size, generate_new=False):
        if generate_new:
            self.testing_batches = self.__generate_test_batches(batch_size=batch_size)
        return self.testing_batches
