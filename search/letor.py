

import os
import pickle
import random

import lightgbm as lgb
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import LsiModel
from scipy.spatial.distance import cosine

from .bsbi import BSBIIndex
from .compression import VBEPostings

# ---------------------------
# Data Loading Functions
# ---------------------------

def load_documents(file_path):
    """Load documents from a file."""
    documents = {}
    # Construct absolute path
    abs_path = os.path.join(os.path.dirname(__file__), file_path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"The file {abs_path} does not exist.")
    with open(abs_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(" ", 1)
            if len(parts) != 2:
                print(f"Skipping invalid document line: {line.strip()}")
                continue
            doc_id, content = parts
            documents[doc_id] = content.split()
    return documents


def load_queries(file_path):
    """Load queries from a file."""
    queries = {}
    abs_path = os.path.join(os.path.dirname(__file__), file_path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"The file {abs_path} does not exist.")
    with open(abs_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(" ", 1)
            if len(parts) != 2:
                print(f"Skipping invalid query line: {line.strip()}")
                continue
            q_id, content = parts
            queries[q_id] = content.split()
    return queries

def load_relevance_judgments(file_path, queries, documents):
    """Load relevance judgments from a file."""
    q_docs_rel = {}  # Grouping by q_id
    abs_path = os.path.join(os.path.dirname(__file__), file_path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"The file {abs_path} does not exist.")
    with open(abs_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(" ")
            if len(parts) < 2:
                print(f"Skipping invalid line: {line.strip()}")
                continue
            q_id, doc_id = parts[:2]
            if len(parts) == 3:
                try:
                    rel = int(parts[2])
                except ValueError:
                    print(f"Invalid relevance score, defaulting to 1: {line.strip()}")
                    rel = 1
            else:
                rel = 1  # Default relevance
            if q_id in queries and doc_id in documents:
                if q_id not in q_docs_rel:
                    q_docs_rel[q_id] = []
                q_docs_rel[q_id].append((doc_id, rel))
    return q_docs_rel

def create_dataset(queries, documents, q_docs_rel):
    """Create dataset with positive and negative samples."""
    NUM_NEGATIVES = 1
    group_qid_count = []
    dataset = []

    for q_id, docs_rels in q_docs_rel.items():
        group_qid_count.append(len(docs_rels) + NUM_NEGATIVES)
        for doc_id, rel in docs_rels:
            dataset.append((queries[q_id], documents[doc_id], rel))
        for _ in range(NUM_NEGATIVES):
            negative_sample = random.choice(list(documents.values()))
            dataset.append((queries[q_id], negative_sample, 0))

    return dataset, group_qid_count

# ---------------------------
# Model and Vector Representation Functions
# ---------------------------

def create_lsi_model(documents, num_topics=200):
    """Create and train an LSI model."""
    dictionary = Dictionary(documents.values())
    bow_corpus = [dictionary.doc2bow(doc) for doc in documents.values()]
    lsi_model = LsiModel(bow_corpus, num_topics=num_topics)
    return lsi_model, dictionary

def vector_representation(text, lsi_model, dictionary):
    """Convert text to LSI vector representation."""
    bow = dictionary.doc2bow(text)
    lsi_vector = lsi_model[bow]
    dense_vector = [0.0] * lsi_model.num_topics
    for idx, score in lsi_vector:
        if idx < len(dense_vector):
            dense_vector[idx] = score
    return dense_vector

def calculate_features(query_vector, doc_vector, query, doc):
    """Calculate cosine distance and Jaccard similarity."""
    if len(query_vector) != len(doc_vector):
        raise ValueError(f"Vector lengths do not match: {len(query_vector)} vs {len(doc_vector)}")
    cosine_dist = cosine(query_vector, doc_vector)
    jaccard = len(set(query) & set(doc)) / len(set(query) | set(doc))
    return [cosine_dist, jaccard]

def create_feature_vectors(dataset, lsi_model, dictionary):
    """Create feature vectors for the dataset."""
    X = []
    Y = []
    for query, doc, rel in dataset:
        query_vector = vector_representation(query, lsi_model, dictionary)
        doc_vector = vector_representation(doc, lsi_model, dictionary)
        features = query_vector + doc_vector + calculate_features(query_vector, doc_vector, query, doc)
        X.append(features)
        Y.append(rel)
    return np.array(X), np.array(Y)

def predict_ranking(query, docs, ranker, lsi_model, dictionary):
    """Predict ranking scores for a set of documents given a query."""
    X_unseen = []
    for doc_id, doc in docs:
        query_vector = vector_representation(query.split(), lsi_model, dictionary)
        doc_vector = vector_representation(doc.split(), lsi_model, dictionary)
        features = query_vector + doc_vector + calculate_features(query_vector, doc_vector, query.split(), doc.split())
        X_unseen.append(features)
    X_unseen = np.array(X_unseen)
    scores = ranker.predict(X_unseen)
    return scores

def load_document_content(doc_path):
    """Load and split document content."""
    path = os.path.join(os.path.dirname(__file__), doc_path)
    with open(path, 'r', encoding='utf-8') as file:
        return file.read().split()

def prepare_docs(SERP):
    """Prepare documents for reranking."""
    docs = []
    for score, doc_path in SERP:
        doc_content = load_document_content(doc_path)
        docs.append((doc_path, ' '.join(doc_content)))
    return docs

# ---------------------------
# Evaluation Function
# ---------------------------

def calculate_ndcg(y_true, y_pred, group):
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG).
    y_true: list of relevance scores
    y_pred: list of predicted scores
    group: list of counts per query
    """
    ndcgs = []
    idx = 0
    for g in group:
        true_relevance = y_true[idx:idx + g]
        pred_scores = y_pred[idx:idx + g]
        if len(true_relevance) == 0:
            ndcgs.append(0.0)
            idx += g
            continue
        # Sort documents by predicted scores
        sorted_indices = np.argsort(pred_scores)[::-1]
        sorted_true_relevance = np.array(true_relevance)[sorted_indices]
        # Calculate DCG
        dcg = 0.0
        for i, rel in enumerate(sorted_true_relevance):
            dcg += (2 ** rel - 1) / np.log2(i + 2)
        # Calculate IDCG
        sorted_true_relevance_sorted = sorted(true_relevance, reverse=True)
        idcg = 0.0
        for i, rel in enumerate(sorted_true_relevance_sorted):
            idcg += (2 ** rel - 1) / np.log2(i + 2)
        # Calculate NDCG
        if idcg == 0.0:
            ndcgs.append(0.0)
        else:
            ndcgs.append(dcg / idcg)
        idx += g
    return np.mean(ndcgs)

# ---------------------------
# Hyperparameter Tuning
# ---------------------------

def tune_hyperparameters(train_dataset, train_group_qid_count, val_queries, val_q_docs_rel, documents, lsi_model, dictionary):
    """Tune hyperparameters using the validation set."""
    param_grid = {
        "num_leaves": [20, 40, 60],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [50, 100, 150],
    }
    best_ndcg = -1
    best_params = None
    print("Starting hyperparameter tuning...")
    for num_leaves in param_grid["num_leaves"]:
        for learning_rate in param_grid["learning_rate"]:
            for n_estimators in param_grid["n_estimators"]:
                print(f"\nTraining with params: num_leaves={num_leaves}, learning_rate={learning_rate}, n_estimators={n_estimators}")
                ranker = lgb.LGBMRanker(
                    objective="lambdarank",
                    boosting_type="gbdt",
                    num_leaves=num_leaves,
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                    importance_type="gain",
                    metric="ndcg"
                )
                X_train, Y_train = create_feature_vectors(train_dataset, lsi_model, dictionary)
                ranker.fit(X_train, Y_train, group=train_group_qid_count)
                
                # Prepare validation data
                val_dataset, val_group_qid_count = create_dataset(val_queries, documents, val_q_docs_rel)
                X_val, Y_val = create_feature_vectors(val_dataset, lsi_model, dictionary)
                Y_pred_val = ranker.predict(X_val)
                ndcg = calculate_ndcg(Y_val, Y_pred_val, val_group_qid_count)
                print(f"Validation NDCG: {ndcg:.4f}")
                
                if ndcg > best_ndcg:
                    best_ndcg = ndcg
                    best_params = {"num_leaves": num_leaves, "learning_rate": learning_rate, "n_estimators": n_estimators}
    print(f"\nBest Params: {best_params}, Best Validation NDCG: {best_ndcg:.4f}")
    return best_params

# ---------------------------
# Training and Hyperparameter Tuning
# ---------------------------

def train_models(train_dataset, train_group_qid_count, val_queries, val_q_docs_rel, documents, lsi_model, dictionary):
    """Train the ranker with hyperparameter tuning."""
    # Hyperparameter tuning
    best_params = tune_hyperparameters(train_dataset, train_group_qid_count, val_queries, val_q_docs_rel, documents, lsi_model, dictionary)
    
    # Train final model with best parameters
    print("\nTraining final model with best hyperparameters...")
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        boosting_type="gbdt",
        num_leaves=best_params["num_leaves"],
        learning_rate=best_params["learning_rate"],
        n_estimators=best_params["n_estimators"],
        importance_type="gain",
        metric="ndcg"
    )
    X_train, Y_train = create_feature_vectors(train_dataset, lsi_model, dictionary)
    ranker.fit(X_train, Y_train, group=train_group_qid_count)
    
    return ranker, best_params

# ---------------------------
# Reranking Function
# ---------------------------

# # NOn Absolute
# def rerank_search_results(search_query, top_k=100):
#     """Rerank search results using the trained ranker."""
#     # Paths for models
#     lsi_model_file = 'letor/lsi_model.pkl'
#     ranker_file = 'letor/lgb_ranker_lsi_model.pkl'

#     # Ensure model directory exists
#     if not os.path.exists('letor/'):
#         os.makedirs('letor/')

#     # Load documents
#     documents = load_documents('qrels-folder/train_docs.txt')

#     # Initialize and load BSBIIndex
#     BSBI_instance = BSBIIndex(
#         data_dir=os.path.join(os.path.dirname(__file__), "collections"),
#         postings_encoding=VBEPostings,
#         output_dir=os.path.join(os.path.dirname(__file__), "index")
#     )
#     BSBI_instance.load()

#     # Retrieve SERP using BM25
#     SERP = BSBI_instance.retrieve_bm25(search_query, k=top_k)
#     if SERP == []:
#         print("No search results found.")
#         return []

#     # Training phase
#     if not os.path.isfile(lsi_model_file) or not os.path.isfile(ranker_file):
#         print("\nTraining new models as saved models do not exist.")
        
#         # Load training data
#         print("Loading training data...")
#         queries_train = load_queries('qrels-folder/train_queries.txt')
#         q_docs_rel_train = load_relevance_judgments('qrels-folder/train_qrels.txt', queries_train, documents)
#         train_dataset, train_group_qid_count = create_dataset(queries_train, documents, q_docs_rel_train)
        
#         # Load validation data
#         print("Loading validation data...")
#         queries_val = load_queries('qrels-folder/val_queries.txt')
#         q_docs_rel_val = load_relevance_judgments('qrels-folder/val_qrels.txt', queries_val, documents)
        
#         # Create LSI model
#         NUM_LATENT_TOPICS = 200
#         print(f"Creating LSI model with {NUM_LATENT_TOPICS} latent topics...")
#         lsi_model, dictionary = create_lsi_model(documents, num_topics=NUM_LATENT_TOPICS)
        
#         # Train and tune models
#         print("Creating feature vectors for training data...")
#         ranker, best_params = train_models(train_dataset, train_group_qid_count, queries_val, q_docs_rel_val, documents, lsi_model, dictionary)
        
#         # Save models
#         print("\nSaving trained models...")
#         with open(lsi_model_file, 'wb') as f:
#             pickle.dump((lsi_model, dictionary), f)
#         with open(ranker_file, 'wb') as f:
#             pickle.dump(ranker, f)
#     else:
#         # Load pre-trained models
#         print("\nLoading pre-trained models...")
#         with open(lsi_model_file, 'rb') as f:
#             lsi_model, dictionary = pickle.load(f)
#         with open(ranker_file, 'rb') as f:
#             ranker = pickle.load(f)

#     # Testing phase with test_qrels
#     print("\nTesting with test_qrels...")
#     queries_test = load_queries('qrels-folder/test_queries.txt')
#     q_docs_rel_test = load_relevance_judgments('qrels-folder/test_qrels.txt', queries_test, documents)

#     test_dataset, test_group_qid_count = create_dataset(queries_test, documents, q_docs_rel_test)
#     X_test, Y_test = create_feature_vectors(test_dataset, lsi_model, dictionary)
#     Y_pred_test = ranker.predict(X_test)

#     test_ndcg = calculate_ndcg(Y_test, Y_pred_test, test_group_qid_count)
#     print(f"Test NDCG: {test_ndcg:.4f}")

#     # Prepare documents for reranking
#     print("\nPreparing documents for reranking...")
#     docs = prepare_docs(SERP)
#     scores = predict_ranking(search_query, docs, ranker, lsi_model, dictionary)

#     # Rerank SERP
#     reranked_SERP = sorted(zip(scores, [doc_path for doc_path, _ in docs]), key=lambda x: x[0], reverse=True)

#     return reranked_SERP

# Absolute path
def rerank_search_results(search_query, top_k=100):
    """Rerank search results using the trained ranker."""
    # Get absolute paths for models
    base_dir = os.path.dirname(os.path.abspath(__file__))
    lsi_model_file = os.path.join(base_dir, 'letor/lsi_model.pkl')
    ranker_file = os.path.join(base_dir, 'letor/lgb_ranker_lsi_model.pkl')
    letor_dir = os.path.join(base_dir, 'letor/')

    # Ensure model directory exists
    if not os.path.exists(letor_dir):
        os.makedirs(letor_dir)

    # Load documents with absolute path
    documents = load_documents(os.path.join(base_dir, 'qrels-folder/train_docs.txt'))

    # Initialize and load BSBIIndex with absolute paths
    BSBI_instance = BSBIIndex(
        data_dir=os.path.join(base_dir, "collections"),
        postings_encoding=VBEPostings,
        output_dir=os.path.join(base_dir, "index")
    )
    BSBI_instance.load()

    # Rest of the function remains the same, but update paths to be absolute
    SERP = BSBI_instance.retrieve_bm25(search_query, k=top_k)
    if SERP == []:
        print("No search results found.")
        return []

    if not os.path.isfile(lsi_model_file) or not os.path.isfile(ranker_file):
        print("\nTraining new models as saved models do not exist.")
        
        # Absolute Path
        queries_train = load_queries(os.path.join(base_dir, 'qrels-folder/train_queries.txt'))
        q_docs_rel_train = load_relevance_judgments(os.path.join(base_dir, 'qrels-folder/train_qrels.txt'), queries_train, documents)
        train_dataset, train_group_qid_count = create_dataset(queries_train, documents, q_docs_rel_train)
        
        queries_val = load_queries(os.path.join(base_dir, 'qrels-folder/val_queries.txt'))
        q_docs_rel_val = load_relevance_judgments(os.path.join(base_dir, 'qrels-folder/val_qrels.txt'), queries_val, documents)
        
        NUM_LATENT_TOPICS = 200
        lsi_model, dictionary = create_lsi_model(documents, num_topics=NUM_LATENT_TOPICS)
        
        ranker, best_params = train_models(train_dataset, train_group_qid_count, queries_val, q_docs_rel_val, documents, lsi_model, dictionary)
        
        with open(lsi_model_file, 'wb') as f:
            pickle.dump((lsi_model, dictionary), f)
        with open(ranker_file, 'wb') as f:
            pickle.dump(ranker, f)
    else:
        with open(lsi_model_file, 'rb') as f:
            lsi_model, dictionary = pickle.load(f)
        with open(ranker_file, 'rb') as f:
            ranker = pickle.load(f)

    queries_test = load_queries(os.path.join(base_dir, 'qrels-folder/test_queries.txt'))
    q_docs_rel_test = load_relevance_judgments(os.path.join(base_dir, 'qrels-folder/test_qrels.txt'), queries_test, documents)

    test_dataset, test_group_qid_count = create_dataset(queries_test, documents, q_docs_rel_test)
    X_test, Y_test = create_feature_vectors(test_dataset, lsi_model, dictionary)
    Y_pred_test = ranker.predict(X_test)

    test_ndcg = calculate_ndcg(Y_test, Y_pred_test, test_group_qid_count)
    print(f"Test NDCG: {test_ndcg:.4f}")

    docs = prepare_docs(SERP)
    scores = predict_ranking(search_query, docs, ranker, lsi_model, dictionary)
    reranked_SERP = sorted(zip(scores, [doc_path for doc_path, _ in docs]), key=lambda x: x[0], reverse=True)

    return reranked_SERP

# ---------------------------
# Main Function
# ---------------------------

def main():
    search_query = "Terletak sangat dekat dengan khatulistiwa"
    reranked_results = rerank_search_results(search_query, 10)

    print("\n--- Search Query Results ---")
    print("Query: ", search_query)
    
    # Initialize and load BSBIIndex for TF-IDF retrieval
    BSBI_instance = BSBIIndex(
        data_dir='collections',
        postings_encoding=VBEPostings,
        output_dir='index'
    )
    BSBI_instance.load()
    SERP = BSBI_instance.retrieve_tfidf(search_query, k=10)

    print("\n--- Initial SERP (TF-IDF) ---")
    for score, doc_id in SERP:
        print(f"{doc_id}: {score}")

    print("\n--- Reranked SERP ---")
    for rank, (score, doc_id) in enumerate(reranked_results, start=1):
        print(f"Rank {rank}: {doc_id} with score {score:.4f}")

if __name__ == "__main__":
    main()


