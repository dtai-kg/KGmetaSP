from evaluation_utils import *

from sentence_transformers import SentenceTransformer, CrossEncoder
import openml
import pprint
from itertools import islice

def get_dataset_doc(dataset_id, 
                    single_field = True):
    """Function that generates dataset document using dataset metadata"""

    dataset = openml.datasets.get_dataset(dataset_id)
    #print(dataset)

    #Get dataset semantic metadata from OpenML
    title = dataset.name

    description = dataset.description

    features = dataset.features
    feature_names = ""
    for feature_key in features:
        feature_names += features[feature_key].name + " "

    tags = dataset.tag
    if isinstance(tags, list):
        tags = " ".join(tags)

    if single_field:
        doc = f"{title} {description} {feature_names} {tags}"
        # doc = f"{title} {description} {feature_names}"

    else:
        doc = {"title": title, 
           "description": description,
           "features": feature_names,
           "tags": tags}
        
    
    return doc

def get_models_doc(dataset_id:str,  
                   valid_runs):
    """Function that generates dataset document using top pipelines' descriptions"""

    doc = ""

    for run_id in valid_runs[dataset_id]:
        try:
            run = openml.runs.get_run(run_id)
            doc += run.flow_name + " "
        except: continue
    
    return doc

def generate_dataset_documents(datasets, 
                               single_field = True):
    """Function that curates all dataset documents (dataset metadata) for benchmark datasets"""
    
    print("Generating dataset docs...")            
    docs = []
    doc_ids = []
    for dataset in datasets:
        doc = get_dataset_doc(int(dataset), single_field)
        docs.append(doc)
        doc_ids.append(int(dataset))

    return docs, doc_ids  

def generate_models_document(datasets, 
                             valid_runs):
    """Function that curates all dataset documents (pipeline metadata) for benchmark datasets"""

    print("Generating models docs...")            
    docs = []
    for dataset in datasets:
        doc = get_models_doc(dataset, valid_runs)
        docs.append(doc)

    return docs

def bi_encoder_sim(docs, 
                   bi_encoder_name):
    """Function that calculates single-field documentes similarity matrix using language model bi-encoder"""

    print("Calculating bi-encoder similarity...")
    bi_encoder = SentenceTransformer(bi_encoder_name)
    doc_embeddings = bi_encoder.encode(docs, normalize_embeddings=True)
    similarity_matrix = bi_encoder.similarity(doc_embeddings, doc_embeddings)
    #print(similarity_matrix)
    
    return similarity_matrix

def single_field_retrieval(dataset_ids, 
                            cand_dataset_ids, 
                            perf_metrics, 
                            perf_ks, 
                            perf_relevance_thresholds):
    """Function that evaluates performance-based retrieval with single-field language model approach."""

    print("\n\nCalculating single field retrieval metrics...")
    #Initialization
    ndatasets = len(dataset_ids)
    sem_relevance_thresholds = [0.9]
    datasets_with_clusters = 0
    single_field = True

    pipeline_evaluations_path = "data/pipeline_evaluations.json"
    pipeline_evaluations = read_json(pipeline_evaluations_path)

    sf_bi_enc_datasets_key = "sf_bi_encoder_dataset_metadata"
    sf_bi_enc_models_key = "sf_bi_encoder_models_metadata"
    retrieval_methods = [sf_bi_enc_datasets_key, sf_bi_enc_models_key]

    valid_runs_path = "data/valid_dataset_runs.json"
    valid_runs = read_json(valid_runs_path)

    bi_encoder_name = "all-MiniLM-L6-v2"
    #bi_encoder_name = "intfloat/multilingual-e5-large"

    perf_metrics_dict = get_metrics_dict(perf_metrics, 
                                         perf_ks, 
                                         perf_relevance_thresholds, 
                                         retrieval_methods)  

    sf_dataset_docs, doc_ids = generate_dataset_documents(cand_dataset_ids, single_field)
    bi_enc_dataset_sim_matrix = bi_encoder_sim(sf_dataset_docs, bi_encoder_name)

    sf_model_docs = generate_models_document(cand_dataset_ids, valid_runs)
    bi_enc_model_sim_matrix = bi_encoder_sim(sf_model_docs, bi_encoder_name)
    
    print("Similarity matrices calculated...")
    print("Evaluating retrieval for each dataset...")
    for dataset_id in dataset_ids:

        dataset_id = int(dataset_id)
        #print(f"Evaluating single field retrieval for dataset: {dataset_id}")
        performance_similarities = dataset_performance_similarities(str(dataset_id), 
                                                                    cand_dataset_ids,
                                                                    pipeline_evaluations)
        if str(dataset_id) in performance_similarities: del performance_similarities[str(dataset_id)]
        
        bi_encoder_dataset_scores = bi_enc_dataset_sim_matrix[doc_ids.index(dataset_id)]
        bi_encoder_dataset_score_dict = get_sim_score_dict(dataset_id, 
                                                   bi_encoder_dataset_scores, 
                                                   doc_ids, 
                                                   performance_similarities)
        
        bi_encoder_model_scores = bi_enc_model_sim_matrix[doc_ids.index(dataset_id)]
        bi_encoder_models_score_dict = get_sim_score_dict(dataset_id, 
                                                   bi_encoder_model_scores, 
                                                   doc_ids, 
                                                   performance_similarities)
        
        perf_metrics_dict = evaluate_retrieval(performance_similarities, 
                                          bi_encoder_dataset_score_dict, 
                                          sf_bi_enc_datasets_key, 
                                          perf_ks, 
                                          perf_relevance_thresholds, 
                                          perf_metrics_dict)
        
        perf_metrics_dict = evaluate_retrieval(performance_similarities, 
                                          bi_encoder_models_score_dict, 
                                          sf_bi_enc_models_key, 
                                          perf_ks, 
                                          perf_relevance_thresholds, 
                                          perf_metrics_dict)
        
    perf_metrics_dict = normalize_metrics(perf_metrics_dict, ndatasets)

    return perf_metrics_dict
        
        
def main():

    #Evaluation configuration
    # dataset_ids = [18]
    cand_dataset_ids = get_benchmark_datasets()
    # cand_dataset_ids = ['4', '18', '11']
    dataset_ids = cand_dataset_ids
    
    perf_metrics = ["hit@k", "ndcg"]
    perf_ks = [1, 2, 5, 10, 15, 20]
    perf_relevance_thresholds = [0.8, 0.9]

    perf_metrics_dict = single_field_retrieval(dataset_ids, 
                                                cand_dataset_ids, 
                                                perf_metrics, 
                                                perf_ks, 
                                                perf_relevance_thresholds)
    
    print("\n\n\nML-aware retrieval simulation:")
    pprint.pprint(perf_metrics_dict)
        
    return

if __name__ == "__main__":

    main()
