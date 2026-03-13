# ML Dataset Discovery

This repository is a comprehensive project for exploring machine learning dataset similarity and retrieval.

**Table of Contents:**
1. [Knowledge Graph Embedding-Based Dataset Similarity](#1-knowledge-graph-embedding-based-dataset-similarity)
2. [Dataset Retrieval Evaluation](#2-dataset-retrieval-evaluation)
3. [Pipeline-specific meta-model analysis](#3-pipeline-specific-meta-model-analysis)
4. [Indicative Results: Link Prediction Models](#4-indicative-results-link-prediction-models-for-pipeline-performance-prediction)

## 1. Knowledge Graph Embedding-Based Dataset Similarity
Located in the [`kge-based-dataset-similarity`](kge-based-dataset-similarity/) folder, this component focuses on:

- Preparing data.
- Training RDF2Vec-based Knowledge Graph Embedding (KGE) models.
- Calculating KGE-based similarities.
- Computing Graph Edit Distances (GEDs).
- Predicting ML pipeline performance using embeddings.
- Analyzing retrieval evaluation results and generating figures and tables.

Refer to the [README.md](kge-based-dataset-similarity/README.md) in this folder for detailed instructions on setup, usage, and commands.

## 2. Dataset Retrieval Evaluation
Located in the [`dataset-retrieval-evaluations`](dataset-retrieval-evaluations/) folder, this component focuses on:

- Calculation of ground truth based on the performance of ML pipelines after applied on the available datasets.
- Calculating rank-based metrics for both baseline methods and KGE-based similarity.
- Implementation of baseline methods.

Refer to the [README.md](dataset-retrieval-evaluations/README.md) in this folder for detailed instructions on setup, usage, and commands.

## 3. Pipeline-specific meta-model analysis
Located in the [`pipeline-specific-meta-model-analysis`](pipeline-specific-meta-model-analysis/) folder, this component focuses on:

- Meta-feature retrieval from OpenML for the investigated datasets
- Training of pipeline-specific meta-models for performance prediction
- Aggregation and analysis of pipeline-specific meta-model evaluations

Refer to the [README.md](pipeline-specific-meta-model-analysis/README.md) in this folder for detailed instructions on setup, usage, and commands.

## 4. Indicative Results: Link Prediction Models for Pipeline Performance Prediction

We conducted additional experiments using **link prediction (LP)-based KGE models within KGmetaSP** for the pipeline performance prediction (PPE) task. Across settings, **KGmetaSP (RDF2Vec)** consistently outperforms **KGmetaSP (LP)**, aligning with prior findings that walk-based methods are better suited for sparse and structurally complex KGs such as MetaExe-KG. In settings where KG-based approaches are beneficial, **KGmetaSP (LP)** improves over baselines, supporting the value of KG-based modeling in KGmetaSP.

### Experimental Setup

We trained and evaluated three LP models: **TransE, DistMult, and ComplEx**. We used the PyKEEN library with the following configuration:

**Model Configuration**
- Training: Self-adversarial negative sampling, 1500 epochs  
- Embedding dimension: 128  
- Batch size: 2048  
- Learning rate: 0.0005  
- Negatives per positive: 3  
- Loss margin: 50  

### Pipeline Performance Prediction Results

#### Scenario 1: Unseen Datasets

**Meta-Classification (Target: Accuracy)**

| Dataset Emb. | Pipeline Strategy           | Acc.   | F1     |
|---|---|---:|---:|
| MF All | Conf.-specific               | 0.7363 | 0.7358 |
| MF All | KGmetaSP (LP)               | 0.7351 | 0.7368 |
| MF All | **KGmetaSP (RDF2Vec)**      | **0.7413** | **0.7427** |

**Meta-Regression (Target: Accuracy)**

| Dataset Emb. | Pipeline Strategy           | MSE    | R²     |
|---|---|---:|---:|
| MF All | **Conf.-specific**               | **0.0081** | **0.6748** |
| MF All | KGmetaSP (LP)               | 0.0105 | 0.6032 |
| MF All | KGmetaSP (RDF2Vec)          | 0.0101 | 0.6181 |

**Key Findings**
- **KGmetaSP (RDF2Vec)** outperforms **KGmetaSP (LP)** in both tasks (F1: 0.7427 vs. 0.7368; R²: 0.6181 vs. 0.6032).
- In meta-classification, **KGmetaSP (LP)** underperforms **KGmetaSP (RDF2Vec)** but outperforms the non-KG baseline.

#### Scenario 2: Unseen Pipelines

**Meta-Classification (Target: Accuracy)**

| Method                    | Acc.   | F1     |
|---|---:|---:|
| Avg. Performance (Base)   | 0.3303 | 0.1640 |
| Closest Embedding (Base)  | 0.7748 | 0.7747 |
| MF All + KGmetaSP (LP)             | 0.8055 | 0.8045 |
| **MF All + KGmetaSP (RDF2Vec)**    | **0.8250** | **0.8244** |

**Meta-Regression (Target: Accuracy)**

| Method                    | MSE | R² |
|---|---:|---:|
| Avg. Performance (Base)   | 0.0267 | -0.0005 |
| Closest Embedding (Base)  | 0.0127 | 0.5241 | 
| MF All + KGmetaSP (LP)             | 0.0081 | 0.6976 |
| **MF All + KGmetaSP (RDF2Vec)**    | **0.0070** | **0.7361** |

**Key Findings**
- Both KGmetaSP variants substantially outperform baselines across both tasks.
- **KGmetaSP (LP)** improves over the closest-embedding baseline (e.g., +3.9% F1 in meta-classification), while **KGmetaSP (RDF2Vec)** further improves over **KGmetaSP (LP)** (e.g., +2.4% F1).
- Across both tasks, the ordering is consistent: **KGmetaSP (RDF2Vec) > KGmetaSP (LP) > baselines**.

### Conclusions

These indicative results show that **KGmetaSP (RDF2Vec)** achieves the strongest KG-based PPE performance. Also, **KGmetaSP (LP)** provides improvements over baselines in settings where KGmetaSP outperforms baselines. Together, the findings indicate that walk-based embeddings offer an advantage for sparse and complex KGs such as MetaExe-KG, and that the improvements are driven by the KG-based modeling in KGmetaSP.
