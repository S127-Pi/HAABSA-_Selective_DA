from transformers import BertTokenizer, AutoTokenizer
from transformers import pipeline
import re
import spacy
import string
import random as rd

# Load the spaCy English model
nlp = spacy.load('en_core_web_sm')
################################################################
# Initialize BERT model
################################################################
BERT_MODEL = 'bert-base-uncased'
unmasker = pipeline(task='fill-mask', model="data/programGeneratedData/finetuning_data/_finetune_model/BERT/bert",
                     tokenizer='bert-base-uncased', top_k = 2)
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

def file_maker(in_file, out_file, strategy):
    
    if strategy == "adverbs":
        augment_func = augment_aspect_adj_adv
    elif strategy == "nouns":
        augment_func = augment_sentence_nouns
    elif strategy == "nouns_adverbs":
        augment_func = augment_all_noun_adj_adv
    elif strategy == "aspect":
        augment_func = augment_sentence_aspect
    elif strategy == "aspect_adverbs":
        augment_func = augment_aspect_adj_adv
    else:
        raise ValueError("Not valid strategy")
    
    rd.seed(546297)
    print('Starting BERT-augmentation')
    with open(in_file, 'r') as in_f, open(out_file, 'w+', encoding='utf-8') as out_f:
        lines = in_f.readlines()
        for i in range(0, len(lines) - 1, 3):
            print(i)
            old_sentence = lines[i].strip()
            target = lines[i + 1].strip()
            sentiment = lines[i + 2].strip()
            new_sentence = augment_func(old_sentence, target, unmasker)
            out_f.writelines([old_sentence + '\n', target + '\n', sentiment + '\n'])
            out_f.writelines([new_sentence + '\n', target + '\n', sentiment + '\n'])
    return out_file

def augment_sentence_aspect(in_sentence, in_target):
    """
    This function selective substitute all aspects occuring in a sentence
    """
    masked_word = in_target
    sentence_mask_target = re.sub(r'\$t\$', "[MASK]", in_sentence)

    results = unmasker(sentence_mask_target)
    predicted_words = []
    target = ""
    for result in results: # decode predicted tokens
        token_id = result['token']
        token_str = tokenizer.decode([token_id])
        predicted_words.append(token_str)
    # print(f"{predicted_words=}")
    if predicted_words[0] == masked_word: # skip to the next predicted word
        sentence_aug_target = re.sub(r'\$t\$', predicted_words[1], in_sentence)
        target = predicted_words[1]
    else:
        sentence_aug_target = re.sub(r'\$t\$', predicted_words[0], in_sentence)
        target = predicted_words[0]

    return sentence_aug_target, target




def augment_sentence_nouns(in_sentence, in_target):
    """
    This function selective substitute all nouns occuring in a sentence
    """
    tar = re.findall(r'\w+|[^\s\w]+', in_target)
    sentence_w_target = re.sub(r'\$t\$', in_target, in_sentence) # replace $t$ with actual target

    # Tokenize the sequence using spaCy
    doc = nlp(sentence_w_target)
    doc_tokens = [token.text for token in doc] # list of tokens
    tar_idx = [i for i, token in enumerate(doc_tokens) if token in tar] # obtain target indices 

    noun_idx = []
    j = 0
    number_not_words = 0
    number_nouns = 0
    while j < len(doc_tokens):
        if doc[j].pos_ in ['NOUN','PRON']:
            noun_idx.append(j)
            j += 1
            number_nouns += 1
        elif doc_tokens[j] in string.punctuation:
            j += 1
            number_not_words += 1
        else:
            j += 1
    # print(F"{number_nouns=}")
    

    i = 0
    augmented_sentence = []
    amount_masked = 0
    cur_sent = doc_tokens.copy()

    while i < len(doc_tokens):
        if doc_tokens[i] in string.punctuation:
            augmented_sentence.append(doc_tokens[i])
            i += 1
        else:
            if doc[i].pos_ in ['NOUN', 'PRON']:
                amount_masked += 1
                masked_word = doc_tokens[i]
                cur_sent[i] = '[MASK]'
                results = unmasker(' '.join(cur_sent))
                predicted_words = []
                for result in results: # decode predicted tokens
                    token_id = result['token']
                    token_str = tokenizer.decode([token_id])
                    predicted_words.append(token_str)
                # print(f"{predicted_words=}")
                if predicted_words[0] == masked_word: # skip to the next predicted word
                    augmented_sentence.append(predicted_words[1])
                    cur_sent[i] = predicted_words[1]
                    i += 1
                else:
                    augmented_sentence.append(predicted_words[0])
                    cur_sent[i] = predicted_words[0]
                    i += 1
            else:
                augmented_sentence.append(doc_tokens[i])
                i += 1

    # Extract the modified_aspect based on in_target_idx in the new augmented sentence
    modified_target = tar
    modified_target = [augmented_sentence[idx] for idx in tar_idx]

    # Replace the target words with '$t$'
    start_index = tar_idx[0]
    end_index = tar_idx[-1] + 1  # +1 because list slicing is exclusive of the end index
    augmentend_sentence = augmented_sentence[:start_index] + ['$t$'] + augmented_sentence[end_index:]

    # Join the masked tokens to form the masked sequence
    augmented_sentence_str = re.sub(r'\s([,.:;!])', r'\1', " ".join(augmented_sentence))
    modified_target_str = ' '.join(modified_target)
    return augmented_sentence_str, modified_target_str


    

    

def augment_sentence_adjective_adverbs(in_sentence, in_target):
    """
    This function selective substitute 15% of adverbs or adjectives occuring in a sentence
    """

    tar = re.findall(r'\w+|[^\s\w]+', in_target) # extract target
    sentence_w_target = re.sub(r'\$t\$', in_target, in_sentence) # substitute $t$ with autual target

    # Tokenize the sequence using spaCy
    doc = nlp(sentence_w_target)
    doc_tokens = [token.text for token in doc] # list of tokens
    tar_idx = [i for i, token in enumerate(doc_tokens) if token in tar]

    j = 0
    number_not_words = 0
    number_adj_adv = 0
    adj_adv_ind = []
    while j < len(doc_tokens):
        if doc[j].pos_ in ['ADJ', 'ADV']:
            print(doc[j].text)
            adj_adv_ind.append(j)
            j += 1
            number_adj_adv += 1
        elif doc_tokens[j] in string.punctuation:
            j += 1
            number_not_words += 1
        else:
            j += 1

    # Mask tokens tagged as ADJ or ADV
    masked_sequence = []
    mask_prob = 0.2
    max_total_mask = 0.15
    num_to_mask = max(1, int(0.15 * number_adj_adv)) #maximum of 15% of adjectives and adverbs can be masked in the sentence

    mask_indices = rd.sample(adj_adv_ind, num_to_mask)
    # print(f"{num_to_mask=}")

    i = 0
    amount_masked = 0
    augmented_sentence = []
    cur_sent = doc_tokens.copy()

    while i < len(doc_tokens):
        if doc_tokens[i] in string.punctuation:
            augmented_sentence.append(doc_tokens[i])
            i += 1
        else:
            if i in mask_indices:
                amount_masked += 1
                masked_word = doc_tokens[i]
                cur_sent[i] = '[MASK]'
                amount_masked += 1
                results = unmasker(' '.join(cur_sent))
                predicted_words = []
                for result in results: # decode predicted tokens
                    token_id = result['token']
                    token_str = tokenizer.decode([token_id])
                    cur_sent[i] = token_str 
                    predicted_words.append(token_str)
                # print(f"{predicted_words=}")
                if predicted_words[0] == masked_word: # skip to the next predicted word
                    augmented_sentence.append(predicted_words[1])
                    cur_sent[i] = predicted_words[1]
                    i += 1
                else:
                    augmented_sentence.append(predicted_words[0])
                    cur_sent[i] = predicted_words[0]
                    i += 1
            else:
                augmented_sentence.append(doc_tokens[i])
                i += 1

    # Extract the modified_aspect based on in_target_idx
    modified_target = tar
    modified_target = [augmented_sentence[idx] for idx in tar_idx]

    # Replace the target words with '$t$'
    start_index = tar_idx[0]
    end_index = tar_idx[-1] + 1  # +1 because list slicing is exclusive of the end index
    augmented_sentence = augmented_sentence[:start_index] + ['$t$'] + augmented_sentence[end_index:]

    # Join the masked tokens to form the masked sequence
    augmented_sentence_str = re.sub(r'\s([,.:;!])', r'\1', " ".join(augmented_sentence))
    modified_target_str = ' '.join(modified_target)
    return augmented_sentence_str, modified_target_str

def augment_aspect_adj_adv(in_sentence, in_target):
    """
    This function selective substitute all aspect, adjectives and adverbs (15%) occuring in a sentence
    """
    aug, aspect = augment_sentence_adjective_adverbs(in_sentence, in_target)
    aug, aspect = augment_sentence_aspect(aug, aspect)

    return aug, aspect

def augment_all_noun_adj_adv(in_sentence, in_target):
    """
    This function selective substitute all nouns, adjectives and adverbs (15%) occuring in a sentence
    """
    aug, aspect = augment_sentence_adjective_adverbs(in_sentence, in_target)
    aug, aspect = augment_sentence_nouns(aug, aspect)

    return aug, aspect


in_sentence = "The $t$ is too dry, but the salmon compensates it all."
in_target = "french fries"
aug, aspect = augment_sentence_adjective_adverbs(in_sentence, in_target)
print(in_sentence)
print(in_target)
print(aug) 
print(aspect) 


# in_sentence = "The $t$ is too dirty, but the salmon compensates it all."
# in_target = "mens bathroom"
# aug, aspect = augment_sentence_nouns(in_sentence, in_target)
# print(in_sentence)
# print(in_target)
# print(aug) 
# print(aspect)


# in_sentence = "The $t$ is too dirty, but the salmon compensates it all."
# in_target = "mens bathroom"
# aug, aspect = augment_sentence_aspect(in_sentence, in_target)
# print(aug)
# print(aspect)



# in_sentence = "The $t$ is too dirty, but the salmon compensates it all."
# in_target = "mens bathroom"
# aug, aspect = augment_all_noun_adj_adv(in_sentence, in_target)
# print(aug)
# print(aspect)

# in_sentence = "The $t$ is too dirty, but the salmon compensates it all."
# in_target = "mens bathroom"
# aug, aspect = augment_aspect_adj_adv(in_sentence, in_target)
# print(aug)
# print(aspect)