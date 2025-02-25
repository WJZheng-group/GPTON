import pandas as pd
import re
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

df_ref = pd.read_csv('example_data/go_terms_bp_all.csv')
eval_table = pd.read_csv('example_data/output/eval_table.csv')

def sub_ref(ref):
    pattern1 = re.compile(r"^This term refers to a .* which involves")
    pattern2 = re.compile(r"^This term refers to a .* which is")

    new_ref = []

    for rephrase in ref:
        description = rephrase
        description = pattern1.sub("", description).strip()
        description = pattern2.sub("", description).strip()
        new_ref.append(description)

    return new_ref

def sub_gen(gen):
    pattern = re.compile(r"^This process involves ")

    new_gen = []

    for desc in gen:
        description = desc
        description = pattern.sub("", description).strip()
        new_gen.append(description)

    return new_gen

def get_term_embeddings(terms, model_name="ncbi/MedCPT-Query-Encoder", max_length=64, batch_size=32):
    # Load the model and tokenizer
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    all_embeds = []

    with torch.no_grad():
        # Process terms in batches
        for i in tqdm(range(0, len(terms), batch_size), desc="Get term embeddings"):
            batch_terms = terms[i:i + batch_size]
            # Tokenize the terms
            encoded = tokenizer(
                batch_terms, 
                truncation=True, 
                padding=True, 
                return_tensors='pt', 
                max_length=max_length,
            )
            
            # Encode the terms (use the [CLS] last hidden states as the representations)
            embeds = model(**encoded).last_hidden_state[:, 0, :]
            all_embeds.append(embeds)

    # Concatenate all embeddings
    all_embeds = torch.cat(all_embeds, dim=0)
    return all_embeds


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


def get_top_k_indices(gen_emb, ref_emb, k):
    top_k_indices = []
    top_k_sim_scores = []

    for gen in tqdm(gen_emb, desc="Mapping back", total=len(gen_emb)):
        cos_sims = [cosine_similarity(gen, ref) for ref in ref_emb]

        top_k_idx = np.argsort(cos_sims)[-k:][::-1]
        top_k_indices.append(top_k_idx.tolist())

        top_k_score = np.sort(cos_sims)[-k:][::-1]
        top_k_sim_scores.append(top_k_score.tolist())

    return top_k_indices, top_k_sim_scores


def main(gene_descs, k):
    # Generate the embeddings for all the reference terms
    ref = df_ref['gpt_4o_default description'].tolist()
    ref = sub_ref(ref)
    ref_emb = get_term_embeddings(ref)
    torch.save(ref_emb, 'example_data/output/all_bp_embeddings.pt')

    # After the first run, you can load the embeddings from the file
    # ref_emb = torch.load('example_data/output/all_bp_embeddings.pt')
    gen = sub_gen(gene_descs)
    gen_emb = get_term_embeddings(gen)
    top_k_indices, top_k_sim_scores = get_top_k_indices(gen_emb, ref_emb, k)
    top_k_GO_id = []
    top_k_Term = []
    for idx_list in top_k_indices:
        top_k_GO_id.append([df_ref.iloc[idx]["GO"] for idx in idx_list])
        top_k_Term.append([df_ref.iloc[idx]["Term_Description"] for idx in idx_list])
    
    return top_k_GO_id, top_k_Term, top_k_sim_scores

gene_descs = eval_table['generated_summ'].tolist()

if __name__ == "__main__":
    k = 3
    top_k_GO_id, top_k_term, top_k_sim_scores = main(gene_descs, k)
    eval_table[f'top_{k}_GO_id'] = top_k_GO_id
    eval_table[f'top_{k}_term'] = top_k_term
    eval_table[f'top_{k}_sim_scores'] = top_k_sim_scores
    eval_table.to_csv('example_data/output/eval_table_match_back.csv', index=False)

# gene_descs = ["This process involves the formation and maturation of the small subunit of the ribosome, essential for protein synthesis in cells.", "This process involves the chemical reactions and pathways by which the amino acid citrulline is synthesized, utilized, and broken down within the body.", "This process involves the control of the release of glucagon, a hormone crucial for maintaining blood sugar levels by stimulating the liver to convert stored glycogen into glucose."]