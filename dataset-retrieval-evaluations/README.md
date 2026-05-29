# Dataset Retrieval Evaluation

This directory contains the codebase used for dataset retrieval evaluations in the paper **"Beyond Meta-Features: Predicting Performance and Dataset Similarity using Knowledge Graph Embeddings"**. This directory focuses on:

- Generating Sklern Piepeline objects from OpenML pipelines.
- Assessing the correctness of OpenML pipelines.
- Generating a performance-based benchmark vased on OpenML dataset performance on Sklearn pipelines mined from OpenML.
- Implementing baselines for estimating performance-based similarity dataset retrieval based on benchmark. 
- Evaluating embeddings and baselines in performance-based similarity dataset retrieval based on benchmark.
- Evaluating the probability of finding a better OpenML pipeline amongst top retrieved datasets, for a query dataset.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Performance Similarity Benchmark](#performance-similarity-benchmark)
- [Baselines and Embeddings Retrieval Evaluation](#baselines-and-embeddings-retrieval-evaluation)
- [Evaluation of SOTA Pipeline Discovery Probability](#evaluation-of-sota-pipeline-discovery-probability)

## Prerequisites

- Python 3.11.9
- Poetry (for dependency management)

Install dependencies using Poetry:  
```bash
poetry install
```

## MetaExe-Bench: Performance Similarity Benchmark

The existing performance similarity benchmark can be found at:
```bash
data/pipeline_evaluations.json
```

The benchmark can either be extended or created from scratch (after removing the existing file from the directory) with the following command:
```bash
python performance_benchmark/inter_pipeline_simulation.py
```

## Baselines and Embeddings Retrieval Evaluation

The embedding and similarities produced by RDF2Vec model can be found at the `openml_exekgs` folder. To evaluate dataset performance-based similarity for embeddings and baselines (for the two evaluation settings mentioned in the paper), run one of the following commands:

- **Evaluation Setting 1**:  
     Exclude performance values.  
     ```bash
     python similarities_evaluation/retrieval_results.py --excl-performance-values
     ```

- **Evaluation Setting 2**:  
     Include performance values.  
     ```bash
     python similarities_evaluation/retrieval_results.py
     ```

## Evaluation of SOTA Pipeline Discovery Probability 

For each query dataset, the retrieved datasets, are evaluated based on the probability of being associated with an OpenML run that improves state-of-the-art for the query dataset. To estimate this probability, as well as the potential performance increase, run:

```bash
python similarities_evaluation/embeddings_simulation_evaluation.py
```