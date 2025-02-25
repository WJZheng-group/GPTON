# GPTON

**GPTON – Generative Pre-trained Transformers enhanced with Ontology Narration for accurate annotation of biological data**

# Framework of GPTON
![GPTON for gene set annotation using GPT-4 verbalized ontology.](./example_data/fig1.png)

# Implementation
## Overview
Here we provide an implementation of GPTON in trl. The repository is organized as follows:
- `example_data/` contains example files of the biological process terms, the training set and the testing set, and other necessary dataset files;
- `GPT_rephrase.py` is for GO term verbalization using GPT;
- `evaluation.py` predicts gene set summaries using fine-tuned model;
- `gene_desc_match_topk.py` maps the generated summaries back to GO terms;
- `literature_search.py` identifies the related PubMed abstracts for the generated summary;
- `output/` contains all models and prediction results;
- `plot_eval_results.ipynb` is for plotting the figures in the paper.


## Step 1: Prepare data
- Download and extract *biological process* ontology branch from Gene Ontology http://purl.obolibrary.org/obo/go/go-basic.obo;
- Download Gene information from https://ftp.ncbi.nih.gov/gene/DATA/;
- Prepare your gene set of interest including labels and corresponding gene lists. Please see `example_data/example_terms_for_rephrase.csv` for example data.
## Step 2: GO Verbalization by GPT-4
- Run `./GPT_rephrase.py`, and replace the information in *AzureOpenAI* and *GPT_rephrase* functions with your own credentials. Output is stored in `output/rephrased_biological_terms.csv`;
- Organize the training data and testing data into the format in `example_data/train_human.json` and `example_data/test_human.json`, respectively. Note that verbalized ontology terms in training and testing files all start with "This process involves...".
## Step 3: Model training
- You can refer to `trl` (https://github.com/huggingface/trl) to fine-tune the model. Fine-tuned model is stored in `output/fine_tuned_model/`.
## Step 4: Model prediction
- You can run `./evaluation.py` to generate gene set summaries. Predictions are stored in `output/eval_table.csv`.
## Step 5 Map back to GO terms
- You can run `./gene_desc_match_topk.py` to get top k GO terms for each gene set. Results are stored in `output/eval_table_match_back.csv`.
## Step 6 Map back to GO terms
- You can run `./literature_search.py` to extract the most related PubMed abstracts for the generated summary of a given gene set.

# Contact
Please contact zhao.li@uth.tmc.edu or wenjin.j.zheng@uth.tmc.edu if you have any questions.
