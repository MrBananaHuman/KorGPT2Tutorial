from transformers import GPT2LMHeadModel, GPT2Config
from new_tokenizer import MyTokenizer
import torch
import kss
import json

data = open('qa_data/KorQuAD_v1.0_dev.json', 'r', encoding='utf-8')
data_json = json.load(data)

output = open('qa_data/korquad_result.json', 'w', encoding='utf-8')

vocab_file_path = '../tokenizer/vocab.json'
merge_file_path = '../tokenizer/merges.txt'

korquad_tokenizer = MyTokenizer(vocab_file_path, merge_file_path)
korquad_config = GPT2Config(vocab_size=52005, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0, summary_first_dropout=0)
korquad_model = GPT2LMHeadModel(korquad_config)
korquad_model_dir = '../KorGPT-2SampleModel/korquad_model.bin'
korquad_model.load_state_dict(torch.load(korquad_model_dir), strict=False)
korquad_model.to('cpu')

def add_special_tokens_(model, tokenizer, added_tokens):
    orig_num_tokens = tokenizer.get_vocab_size()
    tokenizer.add_special_tokens(added_tokens)

added_korquad_tokens = ['<answer>', '</answer>', '<question>', '</question>']

add_special_tokens_(korquad_model, korquad_tokenizer, added_korquad_tokens)

unk = korquad_tokenizer.convert_tokens_to_ids('<unk>')
pad = korquad_tokenizer.convert_tokens_to_ids('<pad>')
s_answer = korquad_tokenizer.convert_tokens_to_ids('<answer>')
e_answer = korquad_tokenizer.convert_tokens_to_ids('</answer>')
s_question = korquad_tokenizer.convert_tokens_to_ids('<question>')
e_question = korquad_tokenizer.convert_tokens_to_ids('</question>')


def context_tokenizer(text, tokenizer):
    sent_list = kss.split_sentences(text)
    tokens = []
    for sent in sent_list:
        tokenized_sentence = tokenizer.tokenize(text)
        if len(tokens) + len(tokenized_sentence) < 912:
            tokens += ['<s>'] + tokenized_sentence + ['</s>']
        else:
            break
    return tokens

def decoding(ids, tokenizer):
    return tokenizer.convert_ids_to_tokens(ids)

def get_answer(model, context, question, tokenizer):
    tokens = context_tokenizer(context, tokenizer)
    tokens += ['<question>'] + tokenizer.tokenize(question) + ['</question>'] + ['<answer>']
    input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens)).unsqueeze(0)
    answer_output = model.generate(
        input_ids,
        max_length=1024, 
        eos_token_id=e_answer,
        pad_token_id=pad,
        early_stopping=True,
        bad_words_ids=[[unk]]
    )
    answer_start_idx = len(input_ids.tolist()[0])
    decoded_answer = decoding(answer_output.tolist()[0][answer_start_idx:-1], tokenizer)
    return decoded_answer


result_map = {}

qa_datas = data_json['data']
for qa_data in qa_datas:
    paras = qa_data['paragraphs']
    for para in paras:
        context = para['context'].replace('\n', ' ').strip()
        qas = para['qas']
        for qa in qas:
            question_text = qa['question'].replace('\n', ' ').strip()
            question_id = qa['id']
            result_map[question_id] = get_answer(korquad_model, context, question_text, korquad_tokenizer)
            
output.write(json.dumps(result_map))
output.close()
