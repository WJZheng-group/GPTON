# GPTON example data

This folder contains the required files to facilitate this repository and the running of the code for a simple demonstration of GPTON's workflow. The files are shortened compared to the full data that was used during our study to ensure the compactness of the sample code repository. 

## The detail of the files are listed below: 

- **example_terms_for_rephrase.csv**: The raw file that contains the true GO term label for the gene sets contained in the sample training set. 
- **fig1.png**: Figure file that showcases the workflow of GPTON. 
- **gene2pubmed_10_29_24**: A sample table that contains the relationship between genes and PubMed articles. 
- **go_terms_bp_all.csv**: Table that contains detailed information of all GO terms under the biological process (BP) branch. 
- **Homo_sapiens.gene_info**: A sample table that contains detailed information of genes that belongs to the human species. 
- **hsgene_pubmed_embed_dict_filterGreater50.pkl**: A saved dictionary file that stores the encoded titles and abstracts of the PubMed articles to facilitate verctor space search. 
- **PubMed_id_abstract_dictionary.csv**: A table that contains the PubMed IDs, title, and abstracts of PubMed articles. 
- **test_human.json**: JSON file that contains a sampled test set of gene sets for GPTON to annotate. 
- **train_human.json**: JSON file that contains a sampled training set for the fine-tuning of GPTON. 