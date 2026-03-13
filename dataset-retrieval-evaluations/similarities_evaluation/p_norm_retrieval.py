from evaluation_utils import *

import pprint
import math
import openml
import sys

def get_metafeatures(dataset_id):
    """Function that retrieves dataset metafeatures from OpenML. MLSea-KG metafeatures are kept.
    Experiments can be done with all OpenML metafeatures as well."""

    dataset = openml.datasets.get_dataset(dataset_id)

    # Filter-in only MLSea_KG qualities for fair comparison
    included_qualities = ["MajorityClassSize",
                          "MinorityClassSize",
                          "NumberOfClasses",
                          "NumberOfFeatures",
                          "NumberOfInstances",
                          "NumberOfInstancesWithMissingValues",
                          "NumberOfMissingValues",
                          "NumberOfNumericFeatures",
                          "NumberOfSymbolicFeatures"]
    
    dataset_qualities = dataset.qualities
    dataset_qualities_filtered = {quality: dataset_qualities[quality] 
                                  for quality in included_qualities}
    
    return dataset_qualities_filtered

def generate_metafeatures(datasets):
    """Function that curates metafeatures from all datasets of benchmark"""

    print("Generating dataset metafeatures...")            
    metafeature_dicts = []
    dict_ids = []
    for dataset in datasets:
        metafeature_dict = get_metafeatures(int(dataset))
        metafeature_dicts.append(metafeature_dict)
        dict_ids.append(int(dataset))

    return metafeature_dicts, dict_ids

def get_min_max_metafeatures(metafeatures_dict):
    """Function that calculates minimum and maximum for all metafeatures for use in normalization."""

    min_dict, max_dict = {}, {}
    metafeatures = set()

    for metafeature_dict in metafeatures_dict:
        #print(metafeature_dict)
        for metafeature in metafeature_dict:
            metafeatures.add(metafeature)

    metafeatures = list(metafeatures)
    for metafeature in metafeatures:
        min_dict[metafeature] = float('+inf')
        max_dict[metafeature] = float('-inf')

        for metafeature_dict in metafeatures_dict:
            if metafeature in metafeature_dict and (
                not math.isnan(metafeature_dict[metafeature])
            ):
                min_dict[metafeature] = min(min_dict[metafeature], metafeature_dict[metafeature])
                max_dict[metafeature] = max(max_dict[metafeature], metafeature_dict[metafeature])

    return min_dict, max_dict


def p_norm_distance(metafeatures1, 
                    metafeatures2, 
                    min_metafeature_dict, 
                    max_metafeature_dict):
    """Function that calculates dataset metafeature distance."""

    distance = 0
    scaled_distance = 0
    for quality in metafeatures1:

        quality_val1 = metafeatures1[quality]
        
        if not math.isnan(quality_val1) and (
            quality in metafeatures2) and (
            not math.isnan(metafeatures2[quality])):

            quality_val2 = metafeatures2[quality]

            quality_min_val = min_metafeature_dict[quality]
            quality_max_val = max_metafeature_dict[quality]

            if quality_min_val == quality_max_val: 
                scaled_quality_val1 = scaled_quality_val2 = 0

            else:
                scaled_quality_val1 = (quality_val1 - quality_min_val) / (
                        quality_max_val - quality_min_val)
                scaled_quality_val2 = (quality_val2 - quality_min_val) / (
                        quality_max_val - quality_min_val)
            

            distance += abs(quality_val1 - quality_val2)
            scaled_distance += abs(scaled_quality_val1 - scaled_quality_val2)

    return distance, scaled_distance


def get_distance_matrix(metafeature_dicts, dict_ids):
    """Function that creates dataset similarity matrix based on metafeature distance."""

    print("Calculating p-norm similarity between metafeatures...")

    metafeature_sim_matrix, scaled_metafeature_sim_matrix = [], []
    min_metafeature_dict, max_metafeature_dict = get_min_max_metafeatures(metafeature_dicts)

    for dataset1_qualities in metafeature_dicts:
        p_norm_sims = []
        scaled_p_norm_sims = []
        for dataset2_qualities in metafeature_dicts:

            distance, scaled_distance = p_norm_distance(dataset1_qualities, 
                                           dataset2_qualities,
                                           min_metafeature_dict,
                                           max_metafeature_dict)
            
            p_norm_sims.append(-distance)
            scaled_p_norm_sims.append(-scaled_distance)
        
        metafeature_sim_matrix.append(p_norm_sims)
        scaled_metafeature_sim_matrix.append(scaled_p_norm_sims)

    return metafeature_sim_matrix, scaled_metafeature_sim_matrix

def p_norm_retrieval(dataset_ids, 
                    cand_dataset_ids, 
                    perf_metrics, 
                    perf_ks, 
                    perf_relevance_thresholds):
    """Function that evaluates performance-based retrieval with metafeature distance approach."""

    print("\n\nCalculating p-norm retrieval metrics...")
    #Initialization
    ndatasets = len(dataset_ids)
    sem_relevance_thresholds = [0.9]
    datasets_with_clusters = 0

    pipeline_evaluations_path = "data/pipeline_evaluations.json"
    pipeline_evaluations = read_json(pipeline_evaluations_path)

    p_norm_key, scaled_p_norm_key = "p_norm", "scaled_p_norm"
    retrieval_methods = [p_norm_key, scaled_p_norm_key]

    perf_metrics_dict = get_metrics_dict(perf_metrics, 
                                         perf_ks, 
                                         perf_relevance_thresholds, 
                                         retrieval_methods)  

    metafeature_dicts, dict_ids = generate_metafeatures(cand_dataset_ids)
    metafeature_sim_matrix, scaled_metafeature_sim_matrix = get_distance_matrix(metafeature_dicts, dict_ids)

    print("Similarity matrices calculated...")
    print("Evaluating retrieval for each dataset...")
    for dataset_id in dataset_ids:

        dataset_id = int(dataset_id)
        #print(f"Evaluating single field retrieval for dataset: {dataset_id}")
        performance_similarities = dataset_performance_similarities(str(dataset_id), 
                                                                    cand_dataset_ids,
                                                                    pipeline_evaluations)
        if str(dataset_id) in performance_similarities: del performance_similarities[str(dataset_id)]
        
        p_norm_scores = metafeature_sim_matrix[dict_ids.index(dataset_id)]
        p_norm_scaled_scores = scaled_metafeature_sim_matrix[dict_ids.index(dataset_id)]

        p_norm_score_dict = get_sim_score_dict(dataset_id,
                                               p_norm_scores,
                                               dict_ids,
                                               performance_similarities,
                                               tensors=False)
        
        p_norm__scaled_score_dict = get_sim_score_dict(dataset_id,
                                               p_norm_scaled_scores,
                                               dict_ids,
                                               performance_similarities,
                                               tensors=False)
        
        perf_metrics_dict = evaluate_retrieval(performance_similarities, 
                                          p_norm_score_dict, 
                                          p_norm_key, 
                                          perf_ks, 
                                          perf_relevance_thresholds, 
                                          perf_metrics_dict)
        
        perf_metrics_dict = evaluate_retrieval(performance_similarities, 
                                          p_norm__scaled_score_dict, 
                                          scaled_p_norm_key, 
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

    perf_metrics_dict = p_norm_retrieval(dataset_ids, 
                                            cand_dataset_ids, 
                                            perf_metrics, 
                                            perf_ks, 
                                            perf_relevance_thresholds)
    
    print("\n\n\nML-aware retrieval simulation:")
    pprint.pprint(perf_metrics_dict)

    return
        

if __name__ == "__main__":

    main()