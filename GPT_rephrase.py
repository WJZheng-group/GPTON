import pandas as pd
from openai import AzureOpenAI
from tqdm import tqdm

# Replace "Your Azure Endpoint", "Your API Key", "Your API" with your own credentials

client = AzureOpenAI(
  azure_endpoint = "Your Azure Endpoint", 
  api_key='Your API Key',  
  api_version="Your API Version"
)


def make_user_prompt_rephrase(term):
    instructions = """Rephrase this biological term in natural language using one sentence. Make sure the generated description is centered around this term and within 30-50 words.
The rephrased sentence should first identify which category, out of biological process, cellular component, and molecular function, the given term is from, and then provide an explanation of this term.
Be concise, do not use unnecessary words.
Don't include gene symbol information in your rephrased description.
 
 
"""
   
    examples = """Here are several examples:
 
 
Biological Term: pole plasm
Rephrased Description: This term refers to a cellular component which is a specialized area in the egg cytoplasm that contains critical materials needed for the early stages of embryonic development and the formation of germ cells.
 
 
Biological Term: mitochondrial fragmentation involved in apoptotic process
Rephrased Description: This term refers to a biological process which is the splitting of mitochondria into smaller parts, which is essential for the programmed death of a cell.
 
 
Biological Term: guanylate cyclase regulator activity
Rephrased Description: This term refers to a molecular function which is the control of the enzyme guanylate cyclase, which is vital for converting GTP to cGMP, a key signaling molecule in various cellular processes.
 
 
"""
 
    prompt_text = instructions
    prompt_text += examples
       
    prompt_text += "\n\nThe biological term is: "
    prompt_text += term + "\n\n"
   
    return prompt_text

def GPT_rephrase(prompt_text):
  message_text = [{"role":"system","content":"You are a senior biologist."},
                {"role":"user","content": prompt_text}]
  completion = client.chat.completions.create(
    # Replace "Your model" with your model deployment name
    model="Your model",
    messages = message_text,
    temperature=0.7, # Higher values will make the output more random
    max_tokens=1024,
    top_p=0.95,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None
  )
  rephrase = completion.choices[0].message.content
  return rephrase

def main():
    # Replace your dataset here and load the dataset
    df = pd.read_csv('example_data/example_terms_for_rephrase.csv')
    # Replace 'Term' with the column name of the biological terms in your dataset
    terms = df['term']
    rephrased_terms = []
    for term in tqdm(terms, desc="Rephrasing terms", total=len(terms)):
        prompt_text = make_user_prompt_rephrase(term)
        rephrased_term = GPT_rephrase(prompt_text)
        rephrased_terms.append(rephrased_term)
    df['Rephrased_Term'] = rephrased_terms
    df.to_csv('example_data/output/rephrased_terms.csv', index=False)

if __name__ == "__main__":
    main()