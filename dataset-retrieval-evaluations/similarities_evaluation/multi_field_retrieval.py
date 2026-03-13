from evaluation_utils import *
from single_field_retrieval import generate_dataset_documents

from sentence_transformers import SentenceTransformer, CrossEncoder
import openml
import pprint
from itertools import islice

def bi_encoder_sim_mf(docs, 
                      bi_encoder_name, 
                      field_weights):
    """Function that calculates multi-field documentes similarity matrix using language model bi-encoder"""

    print("Calculating bi-encoder similarity...")
    bi_encoder = SentenceTransformer(bi_encoder_name)

    doc_embeddings = {key: bi_encoder.encode([doc[key] for doc in docs],
                                            normalize_embeddings=True)
                                            for key in field_weights}
    
    weighted_embeddings = []
    for i in range(len(docs)):
        weighted_embeddings.append(field_weights["title"] * doc_embeddings["title"][i] + (
                                 field_weights["description"] * doc_embeddings["description"][i]) + (
                                 field_weights["features"] * doc_embeddings["features"][i])) #+ (
                                 #field_weights["tags"] * doc_embeddings["tags"][i]))
        
    similarity_matrix = bi_encoder.similarity(weighted_embeddings, weighted_embeddings)
    
    return similarity_matrix 

def multi_field_retrieval(dataset_ids, 
                          cand_dataset_ids, 
                          perf_metrics, 
                          perf_ks, 
                          perf_relevance_thresholds):
    
    """Function that evaluates performance-based retrieval with multi-field language model approach."""

    print("\n\nCalculating multi field retrieval metrics...")

    #Initialization
    ndatasets = len(dataset_ids)
    sem_relevance_thresholds = [0.9]
    datasets_with_clusters = 0
    single_field = False

    pipeline_evaluations_path = "data/pipeline_evaluations.json"
    pipeline_evaluations = read_json(pipeline_evaluations_path)

    field_weights = {
        "title": 0.5,  
        "description": 0.2,  
        "features": 0.2,
        "tags": 0.1  
    }

    mf_bi_enc_key = "multi_field_bi_encoder"
    retrieval_methods = [mf_bi_enc_key]

    bi_encoder_name = "all-MiniLM-L6-v2"
    #bi_encoder_name = "intfloat/multilingual-e5-large"

    perf_metrics_dict = get_metrics_dict(perf_metrics, 
                                         perf_ks, 
                                         perf_relevance_thresholds, 
                                         retrieval_methods)  
    
    mf_docs, doc_ids = generate_dataset_documents(cand_dataset_ids, single_field)
    bi_enc_sim_matrix = bi_encoder_sim_mf(mf_docs, bi_encoder_name, field_weights)
    
    print("Similarity matrices calculated...")
    print("Evaluating retrieval for each dataset...")
    for dataset_id in dataset_ids:

        dataset_id = int(dataset_id)
        #print(f"Evaluating single field retrieval for dataset: {dataset_id}")
        performance_similarities = dataset_performance_similarities(str(dataset_id), 
                                                                    cand_dataset_ids,
                                                                    pipeline_evaluations)
        if str(dataset_id) in performance_similarities: del performance_similarities[str(dataset_id)]
        
        bi_encoder_scores = bi_enc_sim_matrix[doc_ids.index(dataset_id)]
        bi_encoder_score_dict = get_sim_score_dict(dataset_id, 
                                                   bi_encoder_scores, 
                                                   doc_ids,
                                                   performance_similarities)
        
        perf_metrics_dict = evaluate_retrieval(performance_similarities, 
                                          bi_encoder_score_dict, 
                                          mf_bi_enc_key, 
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

    perf_metrics_dict = multi_field_retrieval(dataset_ids, 
                                                cand_dataset_ids, 
                                                perf_metrics, 
                                                perf_ks, 
                                                perf_relevance_thresholds)
    
    print("\n\n\n")
    pprint.pprint(perf_metrics_dict)

    return
        

if __name__ == "__main__":

    main()