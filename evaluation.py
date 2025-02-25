from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd
from rouge import Rouge
from bert_score import score
import time
import torch
from peft import PeftModel
import os
import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)



start_time = time.time()

# Model directory
model_dir =  "example_data/output/fine_tuned_model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
# Model name
model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
# Output evaluation table directory
eval_table_dir = 'example_data/output/eval_table.csv'
# Test data directory
test_data = pd.read_json('example_data/test_human.json', lines=True)
print(eval_table_dir)

def create_model_and_tokenizer():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_safetensors=True,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer

model, tokenizer = create_model_and_tokenizer()
model = PeftModel.from_pretrained(model, model_dir)

# Evaluation metrics
# ROUGE
rouge = Rouge()
def rouge_score(generated_summary, reference_summary):
    scores = rouge.get_scores(generated_summary, reference_summary)
    return scores[0]['rouge-1']['f'], scores[0]['rouge-2']['f'], scores[0]['rouge-l']['f']

# BertScore
def bert_score(generated_summary, reference_summary):
    P, R, F1 = score(cands=generated_summary, refs=reference_summary, lang="en")
    return F1.tolist()[0]

# Evaluation
metrics_table = []
def eval_table(output_text, reference):
    rouge_1, rouge_2, rouge_l = rouge_score([output_text], [reference])
    bert = bert_score([output_text], [reference])
    return rouge_1, rouge_2, rouge_l, bert


for index, rows in test_data.iterrows():
    prompt = rows['input']
    reference = rows['output']
    input_tokens = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt").to('cuda')
    tokens_len = len(input_tokens)
    attention_mask = (input_tokens != tokenizer.pad_token_id).long()

    if tokens_len <= 4096:
        output_tokens = model.generate(
            input_tokens,
            attention_mask=attention_mask,
            max_new_tokens = 100, 
            num_return_sequences=1, 
            num_beams=5, 
            do_sample = True)
        output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        ## for llama3
        output_text = output_text.split('\n')[6]
        #output_text = output_text.replace(template[4:], "").strip().split("\n")[0]
        print('----------------------Output_text------------------------')
        print(output_text)
        print(index)


        if output_text is not None and output_text != "":
            rouge_1, rouge_2, rouge_l, bert = eval_table(output_text, reference)
            print(rouge_1, rouge_2, rouge_l, bert)

            metrics_table.append(
                    {
                        'reference_summ': reference,
                        'generated_summ': output_text,
                        'rouge_1_f': rouge_1,
                        'rouge_2_f': rouge_2,
                        'rouge_l_f': rouge_l,
                        'bertscore': bert,
                    }
                )
            
metrics = pd.DataFrame(metrics_table)
print(f"Number of predictions: {metrics.shape[0]}")
print(f"Average rouge_1 F1 score: {metrics['rouge_1_f'].mean()}")
print(f"Average rouge_2 F1 score: {metrics['rouge_2_f'].mean()}")
print(f"Average rouge_l F1 score: {metrics['rouge_l_f'].mean()}")
print(f"Average Bert F1 score: {metrics['bertscore'].mean()}")

metrics.to_csv(eval_table_dir, index=None)

end_time = time.time() 
elapsed_time = end_time - start_time 

print(f"The code ran for {elapsed_time} seconds")
