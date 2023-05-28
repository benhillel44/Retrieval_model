import ir_datasets


def load_documents(num_of_documents):
    # load num_of_documents documents from ir_datasets and save them in a dictionary with the document id as key
    dataset = ir_datasets.load('pmc/v2/trec-cds-2016')
    documents = {}
    i = 0
    for doc in dataset.docs_iter():
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
