# siRNADiscovery: A Graph Neural Network for siRNA Efficacy Prediction via Deep RNA Sequence Analysis

This repository contains the source code for **siRNADiscovery**.

Rongzhuo Long†, Ziyu Guo†, Da Han, Boxiang Liu, Xudong Yuan*, Guangyong Chen*, Pheng-Ann Heng, Liang Zhang*<sup>#</sup>, siRNADiscovery: a graph neural network for siRNA efficacy prediction via deep RNA sequence analysis, *Briefings in Bioinformatics*, Volume 25, Issue 6, November 2024, bbae563, https://doi.org/10.1093/bib/bbae563

† contributed equally, 
\* corresponding authors, 
<sup>#</sup> lead corresponding author

For questions or further information, please contact the lead corresponding author, Liang Zhang, at [zhangliang@him.cas.cn](mailto:zhangliang@him.cas.cn).

## Table of Contents

- [siRNADiscovery: A Graph Neural Network for siRNA Efficacy Prediction via Deep RNA Sequence Analysis](#sirnadiscovery-a-graph-neural-network-for-sirna-efficacy-prediction-via-deep-rna-sequence-analysis)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Data Preprocessing](#data-preprocessing)
    - [RNA-Protein Interaction Probabilities](#rna-protein-interaction-probabilities)
    - [siRNA-mRNA Base-Pairing Probabilities](#sirna-mrna-base-pairing-probabilities)
  - [Dependencies](#dependencies)
  - [Running the Code](#running-the-code)
  - [Prediction (Inference)](#prediction-inference)

## Overview

siRNADiscovery leverages a graph neural network (GNN) to predict the efficacy of small interfering RNA (siRNA) sequences in gene silencing. This model incorporates RNA-protein interaction probabilities and siRNA-mRNA base-pairing probabilities as part of its predictive framework.

## Data Preprocessing

### RNA-Protein Interaction Probabilities

To generate RNA-protein interaction probabilities, we utilize tools provided by the [RPISeq website](http://pridb.gdcb.iastate.edu/RPISeq/). These probabilities are essential for building RNA-protein interaction features used in siRNADiscovery.

### siRNA-mRNA Base-Pairing Probabilities

For siRNA-mRNA base-pairing probabilities, we use the [ViennaRNA package](https://www.tbi.univie.ac.at/RNA/). Specifically, tools like **RNAcofold** and **RNAfold** are employed for calculating these pairing probabilities.

Refer to the **Data_preprocess** folder for detailed instructions on executing RNAcofold and RNAfold.


## Dependencies

Ensure that you have the following installed to run the code smoothly:

- Python 3.x
- Required Python packages (list of dependencies in `requirements.txt`)
- ViennaRNA (for siRNA-mRNA base-pairing calculations)

To install the required Python packages:

```bash
# Using pixi (recommended — see pixi.toml)
pixi install

# Or using pip
pip install -r requirements.txt
```

## Running the Code

You can reproduce the results by executing the relevant scripts located in the `siRNA_split` and `mRNA_split` folders.

```bash
# Train the siRNA-split model (10-fold cross-validation)
cd siRNA_split && python siRNADiscovery.py

# Train the mRNA-split model
cd mRNA_split && python siRNADiscovery.py
```

Both training scripts automatically save model weights after each fold to `saved_models/fold{N}_weights.h5`.

## Prediction (Inference)

After training, use the `predict.py` script in each split directory to make predictions on new siRNA-mRNA sequence pairs.

### Input

The model requires **both siRNA and mRNA sequences** — the HinSAGE graph neural network has three node types (siRNA, mRNA, interaction) and all are needed.

Sequence features (one-hot, GC%, k-mers, thermodynamics, positional encoding, rules scores) are computed on the fly. External-tool-dependent features (RNAfold self-fold, RNAcofold co-fold, RPISeq AGO2 probabilities) are zero-filled by default; for best accuracy, precompute them and load via CSV.

### Usage

```bash
# Single pair
python siRNA_split/predict.py \
  --sirna-seq UUGCUAGAGAGUUUGGUGUU \
  --mrna-seq "CAGCAGUGGCAGUUCAGAUGCG..." \
  --model-weights siRNA_split/saved_models/fold0_weights.h5

# Batch prediction from CSV (columns: siRNA_seq, mRNA_seq)
python siRNA_split/predict.py \
  --csv input_pairs.csv \
  --output predictions.csv \
  --model-weights siRNA_split/saved_models/fold0_weights.h5

# Ensemble prediction (average across all 10 folds)
python siRNA_split/predict.py \
  --csv input_pairs.csv \
  --output ensemble_preds.csv \
  --model-weights "siRNA_split/saved_models/fold*.h5"

# Via stdin (one sirna_seq,mrna_seq per line)
echo "UUGCU...,CAGCA..." | python siRNA_split/predict.py \
  --model-weights siRNA_split/saved_models/fold0_weights.h5

# mRNA-split model (same interface)
python mRNA_split/predict.py \
  --sirna-seq UUGCUAGAGAGUUUGGUGUU \
  --mrna-seq "CAGCAG..." \
  --model-weights mRNA_split/saved_models/fold0_weights.h5
```
