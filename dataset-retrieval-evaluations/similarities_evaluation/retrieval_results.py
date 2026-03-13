from p_norm_retrieval import p_norm_retrieval
from single_field_retrieval import single_field_retrieval
from multi_field_retrieval import multi_field_retrieval
from embeddings_simulation_evaluation import embeddings_retrieval
from evaluation_utils import save, get_benchmark_datasets

import pandas as pd
import argparse

def get_final_perf_metrics_dict():
    """Function that defines performance-based similarity dictionary for curating metrics from different models"""

    metrics_dict = {
        "Hit@1 (0.8)": [],
        "Hit@2 (0.8)": [],
        "Hit@5 (0.8)": [],
        "Hit@10 (0.8)": [],
        "Hit@15 (0.8)": [],
        "Hit@20 (0.8)": [],
        "Hit@1 (0.9)": [],
        "Hit@2 (0.9)": [],
        "Hit@5 (0.9)": [],
        "Hit@10 (0.9)": [],
        "Hit@15 (0.9)": [],
        "Hit@20 (0.9)": [],
        "NDCG@1": [],
        "NDCG@2": [],
        "NDCG@5": [],
        "NDCG@10": [],
        "NDCG@15": [],
        "NDCG@20": [],
    }

    return metrics_dict

def populate_final_perf_metrics(final_dict, metric, metric_dict):
    
    final_dict["Hit@1 (0.8)"].append(metric_dict[metric]['hit@k'][0.8][1])
    final_dict["Hit@2 (0.8)"].append(metric_dict[metric]['hit@k'][0.8][2])
    final_dict["Hit@5 (0.8)"].append(metric_dict[metric]['hit@k'][0.8][5])
    final_dict["Hit@10 (0.8)"].append(metric_dict[metric]['hit@k'][0.8][10])
    final_dict["Hit@15 (0.8)"].append(metric_dict[metric]['hit@k'][0.8][15])
    final_dict["Hit@20 (0.8)"].append(metric_dict[metric]['hit@k'][0.8][20])

    final_dict["Hit@1 (0.9)"].append(metric_dict[metric]['hit@k'][0.9][1])
    final_dict["Hit@2 (0.9)"].append(metric_dict[metric]['hit@k'][0.9][2])
    final_dict["Hit@5 (0.9)"].append(metric_dict[metric]['hit@k'][0.9][5])
    final_dict["Hit@10 (0.9)"].append(metric_dict[metric]['hit@k'][0.9][10])
    final_dict["Hit@15 (0.9)"].append(metric_dict[metric]['hit@k'][0.9][15])
    final_dict["Hit@20 (0.9)"].append(metric_dict[metric]['hit@k'][0.9][20])

    final_dict["NDCG@1"].append(metric_dict[metric]['ndcg'][1])
    final_dict["NDCG@2"].append(metric_dict[metric]['ndcg'][2])
    final_dict["NDCG@5"].append(metric_dict[metric]['ndcg'][5])
    final_dict["NDCG@10"].append(metric_dict[metric]['ndcg'][10])
    final_dict["NDCG@15"].append(metric_dict[metric]['ndcg'][15])
    final_dict["NDCG@20"].append(metric_dict[metric]['ndcg'][20])

    return final_dict


def main():
    """Function that curates the evaluation of all embedding models and baselines."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--excl-performance-values", action="store_true", help="Exclude performance values")
    args = parser.parse_args()
    excl_perf = args.excl_performance_values

    #Evaluation configuration
    # dataset_ids = [18]
    cand_dataset_ids = get_benchmark_datasets()
    # cand_dataset_ids = ['4', '18', '11']
    dataset_ids = cand_dataset_ids
    
    perf_metrics = ["hit@k", "ndcg"]
    perf_ks = [1, 2, 5, 10, 15, 20]
    perf_relevance_thresholds = [0.8, 0.9]

    print("Calculating ranking metrics for all methods...")
    final_perf_metrics_dict = get_final_perf_metrics_dict()

    perf_metrics_sf = single_field_retrieval(dataset_ids, 
                                                cand_dataset_ids, 
                                                perf_metrics, 
                                                perf_ks, 
                                                perf_relevance_thresholds)
    
    perf_metrics_mf = multi_field_retrieval(dataset_ids, 
                                                cand_dataset_ids, 
                                                perf_metrics, 
                                                perf_ks, 
                                                perf_relevance_thresholds)
    
    perf_metrics_pnorm = p_norm_retrieval(dataset_ids, 
                                            cand_dataset_ids, 
                                            perf_metrics, 
                                            perf_ks, 
                                            perf_relevance_thresholds)
    
    perf_metrics_emb, _ = embeddings_retrieval(dataset_ids, 
                                                cand_dataset_ids, 
                                                perf_metrics, 
                                                perf_ks, 
                                                perf_relevance_thresholds,
                                                excl_perf)

    perf_metric_dicts = [perf_metrics_sf, perf_metrics_mf, perf_metrics_pnorm, perf_metrics_emb]

    methods = []
    for metric_dict in perf_metric_dicts:
        for metric in metric_dict:
            methods.append(metric)
            final_perf_metrics_dict = populate_final_perf_metrics(final_perf_metrics_dict, metric, metric_dict)

    df = pd.DataFrame(final_perf_metrics_dict, index = methods)
    print("Ranking metrics calculated!")
    print(df)
    
    save("data/retrieval_results/perf_results_cosine_sim_excl_perf.pkl", df) 
    df.to_csv("data/retrieval_results/perf_results_cosine_sim_excl_perf.csv")   
    print("Results succesfully saved!")

if __name__ == "__main__":
    main()