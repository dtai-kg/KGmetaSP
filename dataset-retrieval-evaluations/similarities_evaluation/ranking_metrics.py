import math 

def cluster_score_at_k(ground_truth, predicted_scores, k):
    """Function that calculates custom clusters @ K metric for semantic similarity based on semantic cluster co-occurence."""

    # Step 1: Compute Score@K (sum of true scores in top-K predicted items)
    top_k_items = [item[0] for item in predicted_scores[:k]]
    score_k = sum(ground_truth.get(item, 0) for item in top_k_items)
    
    # Step 2: Compute Perfect Score@K (sum of the top-K highest ground truth scores)
    sorted_ground_truth = sorted(ground_truth.values(), reverse=True)
    perfect_score_k = sum(sorted_ground_truth[:k])
    
    # Avoid division by zero
    if perfect_score_k == 0:
        return 0.0
    
    # Step 3: Compute Normalized Score@K
    return score_k / perfect_score_k

def precision_at_k(ground_truth, predicted_scores, k ,relevance_threshold):
    """Function that calculates precision@K metric given model retrieval similarity and benchmark similarity."""

    #Top k items
    top_k_items = [item[0] for item in predicted_scores[:k]]

    # Check relevance in ground truth
    relevant_count = sum(1 for item in top_k_items if ground_truth[item] > relevance_threshold)
    
    # Precision at K
    precision = relevant_count / k
    return precision

def recall_at_k(ground_truth, predicted_scores, k ,relevance_threshold):
    """Function that calculates recall@K metric given model retrieval similarity and benchmark similarity."""

    #Top k items
    top_k_items = [item[0] for item in predicted_scores[:k]]

    # Total relevant items in ground truth How many ground truth scores are above threshold
    total_relevant = sum(1 for score in ground_truth.values() if score > relevance_threshold)
    
    # Relevant items in top K
    relevant_in_top_k = sum(1 for item in top_k_items if ground_truth[item] > relevance_threshold)

    if total_relevant == 0: return 0

    recall = relevant_in_top_k / total_relevant
    return recall

def f1_at_k(p_at_k, r_at_k):
    """Function that calculates F1@K metric given model retrieval similarity and benchmark similarity."""

    if p_at_k == r_at_k == 0: return 0
    return 2 * (p_at_k * r_at_k) / (p_at_k + r_at_k)

def mean_reciprocal_rank(ground_truth, predicted_scores, relevance_threshold):
    #MRR: The average reciprocal rank of the first relevant item across all queries

    for rank, (item, _) in enumerate(predicted_scores, start=1):
        if ground_truth[item] > relevance_threshold:
            return 1/rank
        
    return 0

def dcg(relevance_scores, k):
    """Compute Discounted Cumulative Gain at K."""
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevance_scores[:k]))

def ndcg_at_k(ground_truth, predicted_scores, k):
    """Function that calculates NDCG@K metric given model retrieval similarity and benchmark similarity."""
    #NDCG@K: evaluates the quality of predictions by measuring the relevance of
    # items at each position in the ranking, 
    # with higher-ranked items given exponentially more weight.

    #CG: Measures the relevance of results, regardless the position
    #DCG: Adds a logarithmic discount factor to reduce the impact of lower-ranked results.
    #IDG: Ideal DG --> The DCG of the ideal ranking, where the most relevant items are ranked highest
    #NDCG: DCG / IDCG.

    #Get best predictions in descending order
    predicted_order = [item[0] for item in predicted_scores]

    # Get relevance scores in predicted order
    relevance_scores = [ground_truth[item] if item in ground_truth else 0 for item in predicted_order]
    
    # Compute DCG
    dcg_score = dcg(relevance_scores, k)

    # Compute IDCG
    ideal_relevance_scores = sorted(ground_truth.values(), reverse=True)
    idcg_score = dcg(ideal_relevance_scores, k)

    if idcg_score == 0: return 0.0
    
    return dcg_score / idcg_score

def average_precision(ground_truth, predicted_scores, relevance_threshold):
    """Function that calculates average precision metric given model retrieval similarity, benchmark similarity and a relevance threshold."""

    # Sort predicted items by score in descending order
    predicted_order = [item[0] for item in predicted_scores]
    
    # Get number of relevant items according to ground truth
    num_relevant = sum(1 for score in ground_truth.values() if score > relevance_threshold)
    if num_relevant == 0: return 0.0  
    
    precision_sum = 0.0
    relevant_count = 0
    
    # Compute precision at each rank where a relevant item is found
    for k, item in enumerate(predicted_order, start=1):
        if ground_truth.get(item, 0) > relevance_threshold:  # Check if item is relevant
            relevant_count += 1
            precision_sum += relevant_count / k  # P@k

    return precision_sum / num_relevant


def hit_at_k(ground_truth, predicted_scores, k, relevance_threshold):
    """Function that calculates Hit@K metric given model retrieval similarity, benchmark similarity and a relevance threshold."""

    top_k_items = [item[0] for item in predicted_scores[:k]]
    
    # Check if at least one relevant item exists in the top K
    for item in top_k_items:
        if ground_truth.get(item, 0) > relevance_threshold:
            return 1  # Hit
    
    return 0


def get_metrics_dict(metrics, ks, relevance_thresholds, relevance_methods):
    """Function that initiates a metrics dictionary, using set relevance thresholds, Ks, retrieval models and retrieval metrics"""

    metrics_dict = {rel_method: {} for rel_method in relevance_methods}

    k_dependent_metrics = ["ndcg", "clusters@k", "sota_perf"]
    thresh_dependent_metrics = ["mrr", "ap"]
    k_and_thresh_dependent_metrics = ["p@k", "r@k", "f1@k", "hit@k"]
    difference_metrics = ["sota_diff"]
    measures = ["f1", "r2"]
    
    for key in metrics_dict:
        for metric in metrics:

            metrics_dict[key][metric] = {}

            if metric in k_dependent_metrics:
                for k in ks:
                    metrics_dict[key][metric][k] = 0

            elif metric in thresh_dependent_metrics:
                for thresh in relevance_thresholds:
                    metrics_dict[key][metric][thresh] = 0

            elif metric in k_and_thresh_dependent_metrics:
                for thresh in relevance_thresholds:
                    metrics_dict[key][metric][thresh] = {}
                    for k in ks:
                        metrics_dict[key][metric][thresh][k] = 0

            elif metric in difference_metrics:
                for measure in measures:
                    metrics_dict[key][metric][measure] = {}
                    for k in ks:
                        metrics_dict[key][metric][measure][k] = {}
                        metrics_dict[key][metric][measure][k]["min"] = float('+inf')
                        metrics_dict[key][metric][measure][k]["max"] = float('-inf')
                        metrics_dict[key][metric][measure][k]["avg"] = 0
                        metrics_dict[key][metric][measure][k]["n_success"] = 0
    
    return metrics_dict
