# Pipeline-specific models analysis

This directory contains the codebase used for pipeline-specific performance predicticting meta-models in the paper **"Beyond Meta-Features: Predicting Performance and Dataset Similarity using Knowledge Graph Embeddings"**. This directory focuses on:

- Meta-feature retrieval from OpenML for the investigated datasets
- Training of pipeline-specific meta-models for performance prediction
- Aggregation and analysis of pipeline-specific meta-model evaluations

## Table of Contents
- [Prerequisites](#prerequisites)
- [Pipeline-specific models meta-training and evaluation](#pipeline-specific-models-meta-training-and-evaluation)
- [Aggregate and analyse evaluation results](#aggregate-and-analyse-evaluation-results) 

## Prerequisites

- Python 3.11.11
- Poetry (for dependency management)

Install dependencies using Poetry:  
```bash
poetry install --no-root
```

## Pipeline-specific models meta-training and evaluation
Train and evaluate pipeline-specific meta-models using OpenML meta-features:
```bash
poetry run python -m baselines.model_specific.experiment_simulator
```

## Aggregate and analyse evaluation results
Aggregate and analyse evaluation results:
```bash
poetry run python -m baselines.model_specific.evaluations_analysis
```