import ir_datasets
import pickle

num_of_documents = 1000
# load num_of_documents documents from ir_datasets and save them in a dictionary with the document id as key
dataset = ir_datasets.load('pmc/v2/trec-cds-2016')
documents = {}
i = 0
for doc in dataset.docs_iter():
    documents[doc.doc_id] = doc
    i += 1
    if i == num_of_documents:
        break

# save the dictionary in a pickle file
with open('lib/raw_data.pickle', 'wb') as handle:
    pickle.dump(documents, handle, protocol=pickle.HIGHEST_PROTOCOL)

