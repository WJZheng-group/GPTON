import os
import csv
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

model_name = 'ncbi/MedCPT-Query-Encoder'
medcpt_model = AutoModel.from_pretrained(model_name)
medcpt_tokenizer = AutoTokenizer.from_pretrained(model_name)
medcpt_model.eval()


GENE2PUBMED = pd.read_csv('example_data/gene2pubmed_10_29_24', delimiter='\t')
GENE2PUBMED = GENE2PUBMED.groupby('PubMed_ID').filter(lambda x: len(x) <= 50)

GENE_INFO_HS = pd.read_csv('example_data/Homo_sapiens.gene_info', delimiter='\t')

PMID2ABSTRACT = pd.read_csv('example_data/PubMed_id_abstract_dictionary.csv') # PMID vs abstract table

def find_relevant_pmid(gene_info_df, gene2pubmed_df, pm_gene_dict, embed_dict, genelist, gpton_summ):
    gen_emb = get_term_embeddings(gpton_summ).squeeze()
    filtered_pmidlist, pmid_count = get_pmidlist(genelist, gene_info_df, gene2pubmed_df, pm_gene_dict)
    retrieved_pmids, retrieved_scores, retrieved_titles = retrieve_top_k(filtered_pmidlist, gen_emb, embed_dict)

    result_df = pd.DataFrame(columns=['pmid', 'score', 'title'])
    result_df['pmid'] = retrieved_pmids
    result_df['score'] = retrieved_scores
    result_df['title'] = retrieved_titles

    return result_df.values.tolist()

def get_term_embeddings(term, max_length=64, batch_size=32):
    max_length=64
    with torch.no_grad():
        # Process terms in batches
        encoded = medcpt_tokenizer(
            term, 
            truncation=True, 
            padding=True, 
            return_tensors='pt', 
            max_length=max_length,
        )
        
        # Encode the terms (use the [CLS] last hidden states as the representations)
        embeds = medcpt_model(**encoded).last_hidden_state[:, 0, :]

    return embeds.numpy()


def get_pmidlist(genelist, gene_info_df, gene2pubmed_df, pub2gene_dict, filter_threshold=0.3):
    gene2id_dict = dict(zip(gene_info_df['Symbol'], gene_info_df['GeneID']))
    pmidlist = []
    geneid_list = [gene2id_dict[gene_sym] for gene_sym in genelist if gene_sym in gene2id_dict.keys()]

    for gene_id in geneid_list:
        pm =  list(gene2pubmed_df[gene2pubmed_df['GeneID'] == gene_id]['PubMed_ID'])
        pm = [pmid for pmid in pm if len(set(pub2gene_dict[pmid]) & set(geneid_list))/len(set(pub2gene_dict[pmid])) >= filter_threshold]
        pmidlist.extend(pm)
        
    pmidlist = list(set(pmidlist))

    return pmidlist, len(pmidlist)

def retrieve_top_k(pmidlist, gen_emb, embed_dict):
    cos_sims = [(ref,cosine_similarity(embed_dict[ref], gen_emb)) for ref in pmidlist if ref in embed_dict.keys()]
    if len(cos_sims) == 0:
        print(pmidlist[:5])
        return [], []
    pmids, cos_sims = zip(*cos_sims)

    k = 3
    top_k_idx = np.argsort(cos_sims)[-k:][::-1]

    top_k_pmids = [pmids[idx] for idx in top_k_idx]
    top_k_sim_scores = [cos_sims[idx] for idx in top_k_idx]

    top_k_titles = []
    for top_pmid in top_k_pmids:
        top_title_abs = PMID2ABSTRACT[PMID2ABSTRACT['PubMed_ID']==top_pmid]['Title_abstract'].iloc[0]
        top_k_titles.append(top_title_abs.split('..')[0])

    return top_k_pmids, top_k_sim_scores, top_k_titles

def cosine_similarity(ref_embed, gen_embed):
    # Check if inputs are tensors
    ref_embed = torch.tensor(ref_embed) if not isinstance(ref_embed, torch.Tensor) else ref_embed
    gen_embed = torch.tensor(gen_embed) if not isinstance(gen_embed, torch.Tensor) else gen_embed

    # Normalize the embeddings
    ref_embed = F.normalize(ref_embed, p=2, dim=0)
    gen_embed = F.normalize(gen_embed, p=2, dim=0)

    # Compute the cosine similarity (dot product of the normalized vectors)
    cos_sim = torch.dot(ref_embed, gen_embed).item()

    return cos_sim


def matching_pubmed(genelist, gpton_summ, taxid):
    
    genelist = genelist.split(',')

    gene_info_df = GENE_INFO_HS
    embed_dict_dir = 'example_data/hsgene_pubmed_embed_dict_filterGreater50.pkl'

    gene_info_df = gene_info_df[gene_info_df['#tax_id'] == taxid]
    print('Finished loading gene_info')

    
    specie_gene2pubmed_df = GENE2PUBMED[GENE2PUBMED['#tax_id'] == taxid]
    

    pm_gene_dict = {}
    for id, df in specie_gene2pubmed_df.groupby('PubMed_ID')['GeneID']:
        pm_gene_dict[id] = df.to_list()

    with open(embed_dict_dir, 'rb') as file:
        # Load the dictionary from the pickle file
        embed_dict = pickle.load(file)
    print('Finished loading saved embedding dictionary')

    result_list = find_relevant_pmid(gene_info_df, specie_gene2pubmed_df, pm_gene_dict, embed_dict, genelist, gpton_summ)

    return result_list

if __name__ == '__main__':

    taxid = 9606 # Taxonomy ID of species

    genelist = 'MEGF10,SDC1,WNT10B,SOX15' # A sample input gene list
    # A generated narrative for the input genelist
    gpton_summ = 'the formation and maturation of myoblasts, which are precursor cells that eventually develop into muscle fibers.'
    
    result_list = matching_pubmed(genelist, gpton_summ, taxid)

    cluster_dir = 'example_data/' # Output file directory

    #with open('example_data/output/pubmed_search_result.tsv', 'w') as f:
    #    for line in result_list:
    #        f.write('\t'.join([str(x) for x in line]))
    #        f.write('\n')
    print(result_list)



