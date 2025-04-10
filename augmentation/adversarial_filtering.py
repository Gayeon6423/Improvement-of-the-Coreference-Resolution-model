# Setting up the environment
import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.chdir(os.path.join(os.getcwd(), '..', 'model'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Libraries
import pandas as pd
import numpy as np
import ast
import re
import os
import json
from tqdm import tqdm
from openai import OpenAI
from langchain.prompts.chat import ChatPromptTemplate
from utils_filtering import *
from maverick import Maverick

# Configuration
config_path = "../config.json"
with open(config_path, 'r') as f:
    config = json.load(f)
    
open_api_key = config['openai_api']
folder_path = "../data/LitBank_Case/"
save_path = "../data/NewLitBank/"
case_idx = config["filtering_case_index"]

# Load Model
model = Maverick(hf_name_or_path = config["discriminate_model"],  device = "cuda:0") 
client = OpenAI(api_key=open_api_key)

# Load Data
csv_files = [file for file in os.listdir(folder_path) if file.endswith(".csv")]

# Load Prompt
with open("../prompt/prompt_text.txt", encoding='utf-8') as f:
    prompt_text = f.read()
with open("../prompt/request_prompt_text.txt", encoding='utf-8') as f:
    request_prompt_text = f.read()
with open("../prompt/next_prompt_text.txt", encoding='utf-8') as f:
    next_prompt_text = f.read()
with open("../prompt/request_next_prompt_text.txt", encoding='utf-8') as f:
    request_next_prompt_text = f.read()

def model_inference(sentence):
    result = model.predict(sentence)
    clusters_token_offsets_list = clusters_token_offsets_to_list(result['clusters_token_offsets'])
    return clusters_token_offsets_list

progress_bar = tqdm(total=(len(csv_files) - case_idx))
while case_idx<len(csv_files):
    step = 0
    coref_case = csv_files[case_idx]
    df_case = pd.read_csv(folder_path + coref_case)
    df_case['extracted_sentence'] = [ast.literal_eval(data) for data in df_case['extracted_sentence']]
    df_case['extracted_sentence'] = df_case['extracted_sentence'].apply(ontonote_to_list)
    df_case['text'] = [ast.literal_eval(data) for data in df_case['text']]
    df_case['coref'] = [ast.literal_eval(data) for data in df_case['coref']]
    df_case['adjusted_offsets'] = [ast.literal_eval(data) for data in df_case['adjusted_offsets']]
    
    New_LitBank_df = pd.DataFrame()
    
    while True:
        #Terminate
        step += 1
        if df_case.empty:
            # Maverick doesn't predict all case at first -> we need to add each col to fit other form
            for col in ['update_sentence', 'update_coref', 'update_text']:
                if col not in New_LitBank_df.columns:
                    New_LitBank_df[col] = np.nan
            break
        
        #Prevent too much augmentation. Actually, if the step is bigger than 10, it must be some problem in the extracting update words.
        if step>100:
            New_LitBank_df = pd.concat([New_LitBank_df, df_case]).reset_index(drop=True)
            break
        
        #First step
        if step == 1:
            #model inference
            df_case['inference_offsets'] = df_case['extracted_sentence'].apply(model_inference)
            df_case['labels'] = [1 if data.adjusted_offsets in data.inference_offsets else 0 for data in df_case.itertuples()]
            
            #New LitBank
            New_LitBank_df = pd.concat([New_LitBank_df, df_case[df_case['labels']==0]]).reset_index(drop=True)
            df_case = df_case[df_case['labels']==1].reset_index(drop=True)
            
            update_text_list = []
            prompt = ChatPromptTemplate.from_template(prompt_text)

            for data in df_case.itertuples():
                ontonotes_sentence = data.extracted_sentence
                offsets = data.adjusted_offsets
                words = data.text
                while True:
                    response = client.chat.completions.create(
                        model=config["generate_model"],
                        messages=[{"role": "user",
                                "content": prompt.format(
                                            ontonotes_sentence=ontonotes_sentence,
                                            offsets=offsets,
                                            words=words
                                        )}])
                    output_text = response.choices[0].message.content
                    updated_coreference_words_match = re.search(r"Updated Coreference Words\s*:\s*(\[\[.*?\]\])", output_text, re.DOTALL)

                    if updated_coreference_words_match:
                        updated_coreference_words = updated_coreference_words_match.group(1) if updated_coreference_words_match else None
                        updated_coreference_words = ast.literal_eval(updated_coreference_words) if updated_coreference_words else None
                    else:
                        prompt = ChatPromptTemplate.from_template(request_prompt_text)
                        continue
                    
                    if updated_coreference_words==data.text or len(updated_coreference_words)!=len(offsets):
                        prompt = ChatPromptTemplate.from_template(request_prompt_text)
                    else:
                        prompt = ChatPromptTemplate.from_template(prompt_text)
                        break
                            
                update_text_list.append(updated_coreference_words)
            
            df_case['update_text'] = update_text_list
            df_case['update_sentence'] = df_case.apply(lambda x: replace_words_by_dynamic_indices(x['extracted_sentence'], x['adjusted_offsets'], x['update_text']), axis=1) if df_case.empty==False else []
            df_case['update_coref'] = df_case.apply(lambda x: offset_modify(x['adjusted_offsets'], x['update_text']), axis=1) if df_case.empty==False else []
            
        #After first
        else:
            #model inference
            df_case['inference_offsets'] = df_case['update_sentence'].apply(model_inference)
            df_case['labels'] = [1 if data.update_coref in data.inference_offsets else 0 for data in df_case.itertuples()]
            #New LitBank
            New_LitBank_df = pd.concat([New_LitBank_df, df_case[df_case['labels']==0]]).reset_index(drop=True)
            df_case = df_case[df_case['labels']==1].reset_index(drop=True)
            
            
            update_text_list = []
            prompt = ChatPromptTemplate.from_template(next_prompt_text)

            for data in df_case.itertuples():
                modified_ontonotes_sentence = data.update_sentence
                original_coreference_words = data.adjusted_offsets
                updated_coreference_words = data.update_text
                updated_coreference_offsets = data.update_coref
                while True:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user",
                                "content": prompt.format(
                                        modified_ontonotes_sentence=modified_ontonotes_sentence,
                                        original_coreference_words=original_coreference_words,
                                        updated_coreference_words=updated_coreference_words,
                                        updated_coreference_offsets=updated_coreference_offsets
                                    )}])
                    output_text = response.choices[0].message.content
                    updated_coreference_words_match = re.search(r"Further Updated Coreference Words\s*:\s*(\[\[.*?\]\])", output_text, re.DOTALL)
    

                    if updated_coreference_words_match:
                        next_updated_coreference_words = updated_coreference_words_match.group(1) if updated_coreference_words_match else None
                        next_updated_coreference_words = ast.literal_eval(next_updated_coreference_words) if next_updated_coreference_words else None
                    else:
                        prompt = ChatPromptTemplate.from_template(request_next_prompt_text)
                        continue
                    
                    if next_updated_coreference_words==data.text or len(next_updated_coreference_words)!=len(updated_coreference_offsets):
                        prompt = ChatPromptTemplate.from_template(next_prompt_text)
                    else:
                        prompt = ChatPromptTemplate.from_template(request_next_prompt_text)
                        break
                update_text_list.append(next_updated_coreference_words)
                
            df_case['update_text'] = update_text_list
            df_case['update_sentence'] = df_case.apply(lambda x: replace_words_by_dynamic_indices(x['update_sentence'], x['update_coref'], x['update_text']), axis=1) if df_case.empty==False else []
            df_case['update_coref'] = df_case.apply(lambda x: offset_modify(x['update_coref'], x['update_text']), axis=1) if df_case.empty==False else []

        df_case = df_case[df_case['labels']==1].reset_index(drop=True)
    New_LitBank_df.to_csv(save_path + f'New_litbank_case_{coref_case}')
    progress_bar.update(1)
    case_idx += 1
progress_bar.close()

# if __name__ == "__main__":
#     main()