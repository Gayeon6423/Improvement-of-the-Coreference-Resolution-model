import spacy
from nltk import sent_tokenize

nlp = spacy.load("en_core_web_sm", exclude=["tagger", "parser", "lemmatizer", "ner", "textcat"])

def clusters_token_offsets_to_list(clusters_token_offsets):
    clusters_token_offsets_list = []
    for clusters_token in clusters_token_offsets:
        clusters_token_offsets_list.append([list(item) for item in clusters_token])
    return clusters_token_offsets_list

def check_increase(list1, list2):
    for l1, l2 in zip(list1, list2):
        if l1[0] == l2[0]:  # 첫 번째 인덱스는 변하지 않음
            if l2[1] >= l1[1] + 2:  # 두 번째 인덱스가 2 이상 커졌는지 확인
                return 0
        else:
            return 0
    return 1

# Function to adjust offsets based on added words
def adjust_offsets(coreference_offsets, original_words, updated_words):
    """
    Adjusts the coreference offsets based on additional words in the updated words list.
    
    Args:
    coreference_offsets (list of lists): Original offsets for each coreference word cluster.
    original_words (list of lists): Original coreference word clusters before modification.
    updated_words (list of lists): Updated coreference word clusters after modification.

    Returns:
    list of lists: Adjusted offsets.
    """
    adjusted_offsets = []
    cumulative_offset = 0

    for i, (original, updated) in enumerate(zip(original_words, updated_words)):
        # Calculate the difference in length due to added words
        added_words_count = len(updated) - len(original)

        # Adjust the starting and ending offsets by the cumulative offset
        start_offset, end_offset = coreference_offsets[i]
        start_offset += cumulative_offset
        end_offset += cumulative_offset + added_words_count

        # Append the adjusted offsets
        adjusted_offsets.append([start_offset, end_offset])

        # Update the cumulative offset
        cumulative_offset += added_words_count

    return adjusted_offsets

def ontonote_to_list(ontonotes_format):
    list_format = []
    for ontonote in ontonotes_format:
        list_format.extend(ontonote)
    return list_format

def word_to_ontonotes_format(clusters_token_list):
    nlp = spacy.load("en_core_web_sm", exclude=["tagger", "parser", "lemmatizer", "ner", "textcat"])
    ontonotes_format_clusters_token_list = []
    for clusters_token in clusters_token_list:
        ontonotes_format_clusters_token = []
        for token in clusters_token:
            tokenized_token = sent_tokenize(token)
            for sent in nlp.pipe(tokenized_token):
                ontonotes_format_clusters_token.extend([tok.text for tok in sent])
        ontonotes_format_clusters_token_list.append(ontonotes_format_clusters_token)
    return ontonotes_format_clusters_token_list

def replace_words_by_dynamic_indices(sentence, adjusted_offsets, update_texts):
    """
    sentence: 리스트 형태의 문장 (예: ['PART', 'ONE', 'CHAPTER', ...])
    adjusted_offsets: 각 단어를 교체할 위치의 리스트 (예: [[18, 20], [23, 25], [43, 43], [60, 63], [121, 126]])
    update_texts: 각 위치에 삽입할 새로운 단어의 리스트 (예: [['notorious', 'Hell', 'Row'], ['notorious', 'Hell', 'Row'], ...])
    """
    # 수정된 문장을 저장할 리스트 생성
    modified_sentence = []
    current_index = 0

    # 모든 오프셋에 대해 순회
    for i in range(len(adjusted_offsets)):
        start, end = adjusted_offsets[i]
        
        # 현재 인덱스부터 교체할 부분의 시작 전까지 원본 문장을 추가
        while current_index < start:
            modified_sentence.append(sentence[current_index])
            current_index += 1
        
        # 교체할 부분을 새로운 텍스트로 대체
        modified_sentence.extend(update_texts[i])
        
        # 교체된 부분을 건너뜀
        current_index = end + 1

    # 남은 부분 추가
    while current_index < len(sentence):
        modified_sentence.append(sentence[current_index])
        current_index += 1

    return modified_sentence

def offset_modify(adjusted_offsets, update_text):
    cumulative = 0
    updated_coreference_offsets_list = []
    for idx in range(len(update_text)):
        updated_coreference_offsets = [0, 0]
        updated_coreference_offsets[0] = cumulative + adjusted_offsets[idx][0]
        updated_coreference_offsets[1] = cumulative + adjusted_offsets[idx][0] + len(update_text[idx]) - 1
        cumulative = updated_coreference_offsets[1] - adjusted_offsets[idx][1]
        updated_coreference_offsets_list.append(updated_coreference_offsets)
    return updated_coreference_offsets_list


def find_coreference_terms(ontonotes_format, clusters_token_offsets):
    result = []
    sentence_lengths = [len(sentence) for sentence in ontonotes_format]
    clusters_token_offsets_list = []
    clusters_token_offsets_list.append(clusters_token_offsets)

    cluster_terms = []
    for start, end in clusters_token_offsets:
        # 문장 번호와 인덱스를 결정
        sentence_idx = 0
        for i, length in enumerate(sentence_lengths):
            if start < length:
                sentence_idx = i
                break
            start -= length
            end -= length
        # 해당 문장에서 단어를 추출
        terms = ontonotes_format[sentence_idx][start:end+1]
        cluster_terms.append(terms)
    result.append(cluster_terms)

    return result


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
                print(cumulative_length, offset[0], new_offset_start,sentence_start, adjusted_offset)
                break            
            
    return unique_sentences, adjusted_offsets


def extract_coreference_sentences_ordered(ontonotes_sentence, coreference_offsets):
    """
    Extracts sentences containing the words at specified coreference offsets
    while preserving the order of occurrence.

    Args:
    ontonotes_sentence (list of lists): The sentence in OntoNotes format.
    coreference_offsets (list of lists): Coreference offsets indicating word positions.

    Returns:
    list of lists: Ordered sentences containing the coreference words.
    """
    # 전체 단어 위치를 저장하는 리스트
    unique_sentences = []
    
    # 각 오프셋에 해당하는 문장을 추출
    for offset in coreference_offsets:
        for sentence_idx, sentence in enumerate(ontonotes_sentence):
            # 각 문장의 시작과 끝 인덱스를 계산
            sentence_start = sum(len(s) for s in ontonotes_sentence[:sentence_idx])
            sentence_end = sentence_start + len(sentence) - 1

            # 오프셋이 해당 문장의 범위 내에 있을 경우 추가 (이미 추가된 문장은 건너뜀)
            if sentence_start <= offset[0] <= sentence_end and sentence not in unique_sentences:
                unique_sentences.append(sentence)
                break

    return unique_sentences

def find_changed_indices(original_list, modified_list):
    # 변경된 인덱스를 저장할 리스트
    changed_indices = []
    
    # 두 리스트의 길이가 같다고 가정
    for i in range(len(original_list)):
        if original_list[i] != modified_list[i]:
            changed_indices.append(i)
    
    return changed_indices