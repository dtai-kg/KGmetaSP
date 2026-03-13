from ranking_metrics import *

import pickle
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine
import math
import json

def get_performance_similarity_cross(dataset1, 
                                     dataset2, 
                                     valid_runs, 
                                     pipeline_evaluations):
    """Function that assesses performance similarity of datasets by cross-evaluating dataset pipelines"""

    #print(dataset1, dataset2)
    runs1 = valid_runs[dataset1]
    runs2 = valid_runs[dataset2]

    runs1_intersection = [str(run) for run in runs1 if (
    str(run) in pipeline_evaluations[dataset1] and str(run) in pipeline_evaluations[dataset2]) and (
    pipeline_evaluations[dataset1][str(run)]["run_success"] == True
    )]

    runs2_intersection = [str(run) for run in runs2 if (
    str(run) in pipeline_evaluations[dataset1] and str(run) in pipeline_evaluations[dataset2]) and (
    pipeline_evaluations[dataset2][str(run)]["run_success"] == True
    )]

    #Get 1st factor: Dataset 1 performance similarity penalty
    if runs1_intersection:

        #Classification score metric
        if "test_f1" in pipeline_evaluations[dataset1][runs1_intersection[0]]["run_metrics"]:
            score_metric = "test_f1"
        #Regression score metric
        elif "test_r2" in pipeline_evaluations[dataset1][runs1_intersection[0]]["run_metrics"]:
            score_metric = "test_r2"

        d1_similarity_penalty = get_performance_similarity_penalty(
        dataset1, dataset2, runs1_intersection, pipeline_evaluations, score_metric)
    #print(runs1_intersection)

    #Get 2nd factor: Dataset 2 performance similarity penalty
    if runs2_intersection:
        #Classification score metric
        if "test_f1" in pipeline_evaluations[dataset2][runs2_intersection[0]]["run_metrics"]:
            score_metric = "test_f1"
        #Regression score metric
        elif "test_r2" in pipeline_evaluations[dataset2][runs2_intersection[0]]["run_metrics"]:
            score_metric = "test_r2"

        d2_similarity_penalty = get_performance_similarity_penalty(
        dataset1, dataset2, runs2_intersection, pipeline_evaluations, score_metric)
    #print(runs2_intersection)

    if runs1_intersection and runs2_intersection:
        return max(0, (2 - d1_similarity_penalty - d2_similarity_penalty) / 2.0)
    
    elif runs1_intersection:
        return max(0, 1 - d1_similarity_penalty)
    
    elif runs2_intersection:
        return max(0, 1 - d2_similarity_penalty)
    
    else: return 0

def get_performance_similarity_full(dataset1:str, 
                                    dataset2:str, 
                                    pipeline_evaluations):
    """Function that assesses performance similarity of datasets by considering all executed pipelines accross datasets"""

    runs = [run for run in pipeline_evaluations[dataset1] if run in pipeline_evaluations[dataset2]]
    if len(runs) == 0: return 0

    classification_metric = "test_f1"
    regression_metric = "test_r2"

    evals1, evals2 = [], []
    for run in runs:
        d1_metric = pipeline_evaluations[dataset1][run]
        #print(dataset1, run, d1_metric)
        if d1_metric["run_success"] == True:
            if classification_metric in d1_metric["run_metrics"]:
                score_metric = classification_metric
            elif regression_metric in d1_metric["run_metrics"]:
                score_metric = regression_metric

            d1_score = d1_metric["run_metrics"][score_metric]
        else: d1_score = 0
        evals1.append(d1_score)
        
        d2_metric = pipeline_evaluations[dataset2][run]
        if d2_metric["run_success"] == True:
            if classification_metric in d2_metric["run_metrics"]:
                score_metric = classification_metric
            elif regression_metric in d2_metric["run_metrics"]:
                score_metric = regression_metric

            d2_score = d2_metric["run_metrics"][score_metric]
        else: d2_score = 0
        evals2.append(d2_score)

    nruns = len(runs)
    perf_diff = 0
    for i in range(nruns):
        perf_diff += min(abs(evals1[i] - evals2[i]), 1)
    perf_diff = perf_diff/nruns
    #perf_dict_w_rank_penalty = get_perf_diff_with_rank_penalty(evals1, evals2)

    if len(set(evals1)) == 1 or len(set(evals2)) == 1:
        spearman_corr = 0 #tag as no correlation
    else:
        spearman_corr, _ = spearmanr(evals1, evals2)
        spearman_corr = max(0, spearman_corr)

    if sum(evals1) == 0 or sum(evals2) == 0:
        cosine_sim = 0
    else:
        cosine_sim = max(0, 1 - cosine(evals1, evals2))

    # return spearman_corr
    # return 1 - perf_diff 
    return cosine_sim

def get_perf_diff_with_rank_penalty(evals1, evals2):
    """Function that calculates rank penalty between evaluation scores"""

    ninit = len(evals1)
    evals1_nz, evals2_nz = [], []
    for i in range(ninit):
        if evals1[i] != 0 and evals2[i] != 0:
            evals1_nz.append(evals1[i])
            evals2_nz.append(evals2[i])

    evals1_rank_dict = rank_evals(evals1_nz)
    evals2_rank_dict = rank_evals(evals2_nz) 
    nfinal = len(evals1_nz)

    if nfinal == 0: return 1
    
    perf_diff_w_rank_penalty = 0
    for i in range(nfinal):
        perf_diff_w_rank_penalty += min(abs(evals1_nz[i] - evals2_nz[i]), 1) + (
                                    abs(evals1_rank_dict[i] - evals2_rank_dict[i])/nfinal)
    perf_diff_w_rank_penalty = min(perf_diff_w_rank_penalty/nfinal, 1)

    return perf_diff_w_rank_penalty
    

def rank_evals(lst):
    """Function that ranks evaluations"""

    sorted_lst = sorted(lst, reverse=True)  
    rank_map = {num: rank + 1 for rank, num in enumerate(sorted_lst)}  
    return {index: rank_map[num] for index, num in enumerate(lst)}

def get_performance_similarity_penalty(dataset1, 
                                       dataset2, 
                                       runs, 
                                       pipeline_evaluations, 
                                       score_metric):
    """Function that calculates performance similarity penalty for cross evaluation method"""

    performance_similarity_penalty = 0
    nruns = float(len(runs))

    for run in runs:
        #print()
        d1_metric = pipeline_evaluations[dataset1][run]
        #print(dataset1, run, d1_metric)
        if d1_metric["run_success"] == True:
            d1_score = d1_metric["run_metrics"][score_metric]
        else: d1_score = 0
        #print(d1_score)

        d2_metric = pipeline_evaluations[dataset2][run]
        #print(dataset2, run, d2_metric)
        if d2_metric["run_success"] == True:
            d2_score = d2_metric["run_metrics"][score_metric]
        else: d2_score = 0
        #print(d2_score)

        performance_similarity_penalty += abs(d1_score - d2_score)

    performance_similarity_penalty /= nruns
    return performance_similarity_penalty


def dataset_performance_similarities(dataset_id, 
                                     candidate_datasets, 
                                     pipeline_evaluations):
    """Function that returns performance similarities based on the generated benchmark"""

    performance_similarities = {}
    for cand_dataset in candidate_datasets:
        # print()
        # print(dataset_id, cand_dataset)
        # performance_similarities[cand_dataset] = get_performance_similarity_cross(
        #                                             dataset_id, cand_dataset, 
        #                                             valid_runs, pipeline_evaluations)
        # print(performance_similarities[cand_dataset])
        performance_similarities[cand_dataset] = get_performance_similarity_full(
                                                    dataset_id, cand_dataset, 
                                                    pipeline_evaluations)    
        
    return performance_similarities

def get_benchmark_datasets():
    """Function that collects current datasets from benchmark"""
    pipeline_evaluations_path = "data/pipeline_evaluations.json"
    pipeline_evaluations = read_json(pipeline_evaluations_path)

    print("Retrieving candidate datasets...")
    datasets = []
    for dataset in pipeline_evaluations:
        
        for run in pipeline_evaluations[dataset]:
            if pipeline_evaluations[dataset][run]["run_success"] == True:
                datasets.append(dataset)
                break

    #excluded_datasets = ['187', '190', '196', '198', '204', '206', '217', '223', '227']
    #Exluded identical dataset pairs for evaluation setting 1
    excluded_datasets = ['187', "729","730","740","743","749","751","754","762","783","789","792","799","808","824","829","845","855","870","884","746","773","775"]
    # excluded_datasets = ['187']
    for dataset in excluded_datasets:
        if dataset in datasets:
            datasets.remove(dataset)
    return datasets

def evaluate_retrieval(ground_sims, 
                       retrieval_scores, 
                       retrieval_method,
                       ks, 
                       relevance_thresholds, 
                       metrics_dict):
    """Function that evaluates retrieved similarites using benchmark similarites"""

    #print("\nEvaluating embeddings similarity measures...")
    sorted_retrieval_scores = sorted(retrieval_scores.items(), key=lambda x: x[1], reverse=True)

    #print(sorted_data_emb_sims)
    
    #Perform different similarity evaluation metrics
    for relevance_threshold in relevance_thresholds:
        for k in ks:
            #P@K
            if "p@k" in metrics_dict[retrieval_method]:
                precision_at_k_score = precision_at_k(ground_sims, sorted_retrieval_scores, k, relevance_threshold)
                metrics_dict[retrieval_method]["p@k"][relevance_threshold][k] += precision_at_k_score

            #R@K (Not very meaningful when we have a lot of datasets to compare with)
            if "r@k" in metrics_dict[retrieval_method]:
                recall_at_k_score = recall_at_k(ground_sims, sorted_retrieval_scores, k ,relevance_threshold)                
                metrics_dict[retrieval_method]["r@k"][relevance_threshold][k] += recall_at_k_score

            #F1@K (Not very meaningful when we have a lot of datasets to compare with)
            if "f1@k" in metrics_dict[retrieval_method]:
                f1_at_k_score = f1_at_k(precision_at_k_score, recall_at_k_score)
                metrics_dict[retrieval_method]["f1@k"][relevance_threshold][k] += f1_at_k_score

            #Hit@k
            if "hit@k" in metrics_dict[retrieval_method]:
                hit_at_k_score = hit_at_k(ground_sims, sorted_retrieval_scores, k, relevance_threshold)
                metrics_dict[retrieval_method]["hit@k"][relevance_threshold][k] += hit_at_k_score
        
        #MRR
        if "mrr" in metrics_dict[retrieval_method]:
            mrr_score = mean_reciprocal_rank(ground_sims, sorted_retrieval_scores, relevance_threshold)
            metrics_dict[retrieval_method]["mrr"][relevance_threshold] += mrr_score

        #AP
        if "ap" in metrics_dict[retrieval_method]:
            ap_score = average_precision(ground_sims, sorted_retrieval_scores, relevance_threshold)
            metrics_dict[retrieval_method]["ap"][relevance_threshold] += ap_score
    

    for k in ks:
        
        #NDCG
        if "ndcg" in metrics_dict[retrieval_method]:
            ndcg_at_k_score = ndcg_at_k(ground_sims, sorted_retrieval_scores, k)
            metrics_dict[retrieval_method]["ndcg"][k] += ndcg_at_k_score

        #Clusters@K
        if "clusters@k" in metrics_dict[retrieval_method]:
            clusters_at_k_score = cluster_score_at_k(ground_sims, sorted_retrieval_scores, k)
            metrics_dict[retrieval_method]["clusters@k"][k] += clusters_at_k_score

    return metrics_dict

def normalize_metrics(metrics_dict, 
                      nrml_param):
    """Function that normalizes ranking metrics with the number of benchmark datasets"""


    for key in metrics_dict:

        if type(metrics_dict[key]) == dict:
            metrics_dict[key] = normalize_metrics(metrics_dict[key], nrml_param)
        else:
            metrics_dict[key] /= nrml_param

    return metrics_dict

def get_sim_score_dict(dataset_id, 
                       bi_encoder_scores, 
                       doc_ids, 
                       performance_similarities, 
                       tensors = True):
    """Function that generates a similarity scores dictionary, given model similarity scores"""

    score_dict = {}
    n = len(doc_ids)

    for i in range(n):
        if doc_ids[i] != dataset_id and str(doc_ids[i]) in performance_similarities:
            if tensors: 
                score_dict[str(doc_ids[i])] = bi_encoder_scores[i].item()
            else:
                score_dict[str(doc_ids[i])] = bi_encoder_scores[i]

    return score_dict


def filter_dict_by_dict(dict1, 
                        reference_dict):
    """Function that filters a dictionary given the keys of another"""

    for key in dict1:
        if key not in reference_dict:
            del dict1[key]

    return dict1


def load(filename):
    with open(filename, 'rb') as output:
        data = pickle.load(output)
    return data

def save(filename, data):
    with open(filename, 'wb') as output:
        pickle.dump(data, output)

def save_json(path, dict):

    with open(path, 'w') as file:
        json.dump(dict, file, indent=4)

    return

def read_json(path):
    """Function that reads JSON file"""

    with open(path, 'r') as file:
        data = json.load(file)

    return data


