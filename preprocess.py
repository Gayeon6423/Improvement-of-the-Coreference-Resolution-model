# Setting up the environment
import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Libraries
import jsonlines
import pandas as pd
import ast
import warnings
warnings.filterwarnings('ignore')

# Configuration
litbank_case_folder_path  = os.makedirs("data/LitBank_Case/", exist_ok=True)

# Load Data
LitBank_train=[]
with jsonlines.open("data/LitBank/train.english.jsonlines") as read_file:
    for line in read_file.iter():
        LitBank_train.append(line)

# Change to datafrmae
df_LBtrain = pd.DataFrame(LitBank_train)[['doc_key', 'sentences', 'clusters']]
# Read the sentences and clusters
df_LBtrain['sentences'] = [ast.literal_eval(data) for data in df_LBtrain['sentences']]
df_LBtrain['clusters'] = [ast.literal_eval(data) for data in df_LBtrain['clusters']]

# Process each individual case -> Process one row into one case
def calculate_new_offsets(ontonotes_sentence, coreference_offsets):
    """
    Extracts sentences containing the words at specified coreference offsets,
    while preserving the order of occurrence, and adjusts offsets based on the new subset of sentences.

    Args:
    ontonotes_sentence (list of lists): The sentence in OntoNotes format.
    coreference_offsets (list of lists): Coreference offsets indicating word positions.

    Returns:
    tuple: Ordered sentences containing the coreference words and adjusted offsets.
    """
    unique_sentences = []
    adjusted_offsets = []

    # Track cumulative sentence length as we build the subset of sentences
    cumulative_length = 0
    previous_length = 0
    for offset in coreference_offsets:
        for sentence_idx, sentence in enumerate(ontonotes_sentence):
            # Calculate start and end indices for the sentence
            sentence_start = sum(len(s) for s in ontonotes_sentence[:sentence_idx])
            sentence_end = sentence_start + len(sentence) - 1

            # Check if the offset is within the sentence's range
            if sentence_start <= offset[0] <= sentence_end:
                if sentence not in unique_sentences:
                    unique_sentences.append(sentence)
                    cumulative_length += previous_length

                # Adjust offset relative to the current sentence in the subset
                new_offset_start = offset[0] - sentence_start
                new_offset_end = new_offset_start + (offset[1] - offset[0])

                # Add cumulative length of previous sentences (if any)
                adjusted_offset = [
                    new_offset_start + cumulative_length,
                    new_offset_end + cumulative_length
                ]
                adjusted_offsets.append(adjusted_offset)
                previous_length = len(unique_sentences[-1])
                break

    return unique_sentences, adjusted_offsets

def find_coreference_terms(ontonotes_format, clusters_token_offsets):
    result = []
    sentence_lengths = [len(sentence) for sentence in ontonotes_format]

    for cluster in clusters_token_offsets:
        if len(cluster)==1:
            continue
        cluster_terms = []
        for start, end in cluster:
            # Determining Sentence Numbers and Indexes
            sentence_idx = 0
            for i, length in enumerate(sentence_lengths):
                if start < length:
                    sentence_idx = i
                    break
                start -= length
                end -= length

            # Extract words from the corresponding sentence
            terms = ontonotes_format[sentence_idx][start:end+1]
            cluster_terms.append(terms)
        result.append(cluster_terms)

    return result

def sort_by_first_value(data):
    return sorted(data, key=lambda x: x[0])

for index in range(len(df_LBtrain)):  # Looping through indices 0 to 80
    coref_list = []
    litbank_case = pd.DataFrame()
    for col in litbank_case.columns:
        litbank_case[col] = litbank_case[col].apply(ast.literal_eval) # String Form List -> Convert to Real List
    # Extract coreference clusters for the current index
    for data in df_LBtrain['clusters'][index]:
        coref = data
        if len(coref) == 1:
            continue
        coref_list.append(coref)

    litbank_case['coref'] = coref_list

    # Calculate new offsets and extract sentences
    extracted_sentence_list, adjusted_offsets_list = [], []
    for data in litbank_case['coref']:
        extracted_sentence, adjusted_offsets = calculate_new_offsets(df_LBtrain['sentences'][index], data)
        extracted_sentence_list.append(extracted_sentence)
        adjusted_offsets_list.append(adjusted_offsets)

    litbank_case['extracted_sentence'] = extracted_sentence_list
    litbank_case['adjusted_offsets'] = adjusted_offsets_list

    # sorting 추가
    litbank_case['adjusted_offsets'].apply(lambda x: sort_by_first_value(x))

    litbank_case['text'] = find_coreference_terms(df_LBtrain['sentences'][index], litbank_case['coref'])

    # Save each case to a separate CSV file
    litbank_case.to_csv(f'{litbank_case_folder_path}litbank_case_{index}.csv', index=False)