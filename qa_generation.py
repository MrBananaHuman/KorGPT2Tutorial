from transformers import GPT2LMHeadModel, GPT2Config
from new_tokenizer import MyTokenizer
import torch
import kss

vocab_file_path = '../tokenizer/vocab.json'
merge_file_path = '../tokenizer/merges.txt'

answer_tokenizer = MyTokenizer(vocab_file_path, merge_file_path)
question_tokenizer = MyTokenizer(vocab_file_path, merge_file_path)

answer_config = GPT2Config(vocab_size=52004)
question_config = GPT2Config(vocab_size=52005)

answer_model = GPT2LMHeadModel(answer_config)
question_model = GPT2LMHeadModel(question_config)

answer_model_dir = '../KorGPT-2SampleModel/answer_model.bin'
question_model_dir = '../KorGPT-2SampleModel/question_model.bin'

answer_model.load_state_dict(torch.load(answer_model_dir), strict=False)
question_model.load_state_dict(torch.load(question_model_dir), strict=False)

answer_model.to('cpu')
question_model.to('cpu')

def add_special_tokens_(model, tokenizer, added_tokens):
    orig_num_tokens = tokenizer.get_vocab_size()
    tokenizer.add_special_tokens(added_tokens)

added_answer_tokens = ['<answer>', '<sep>', '</answer>']
added_question_tokens = ['<answer>', '</answer>', '<question>', '</question>']

add_special_tokens_(answer_model, answer_tokenizer, added_answer_tokens)
add_special_tokens_(question_model, question_tokenizer, added_question_tokens)

unk = answer_tokenizer.convert_tokens_to_ids('<unk>')
s_answer = answer_tokenizer.convert_tokens_to_ids('<answer>')
e_answer = answer_tokenizer.convert_tokens_to_ids('</answer>')
s_question = question_tokenizer.convert_tokens_to_ids('<question>')
e_question = question_tokenizer.convert_tokens_to_ids('</question>')


def context_tokenizer(text, tokenizer):
    sent_list = kss.split_sentences(text)
    tokens = []
    for sent in sent_list:
        tokenized_sentence = tokenizer.tokenize(sent)
        if len(tokens) + len(tokenized_sentence) < 912:
            tokens += ['<s>'] + tokenized_sentence + ['</s>']
        else:
            break
    return tokens

def decoding(ids, tokenizer):
    return tokenizer.convert_ids_to_tokens(ids)

def get_answer_list(text):
    tokens = context_tokenizer(text, answer_tokenizer) + ['<answer>']
    input_ids = torch.tensor(answer_tokenizer.convert_tokens_to_ids(tokens)).unsqueeze(0)
    answer_outputs = answer_model.generate(
        input_ids,
        do_sample=True, 
        max_length=1024, 
        top_k=50, 
        top_p=0.95, 
        eos_token_id=e_answer,
        early_stopping=True,
        bad_words_ids=[[unk]]
    )
    answer_start_idx = answer_outputs.tolist()[0].index(s_answer) + 1
    decoded_answers = decoding(answer_outputs.tolist()[0][answer_start_idx:-1], answer_tokenizer)
    answers = decoded_answers.split('<sep>')
    
    return answers

def get_question_list(text, answers):
    questions = []
    for answer in answers:
        tokens = context_tokenizer(text, answer_tokenizer) + ['<answer>']
        tokens += question_tokenizer.tokenize(answer) + ['<question>']
        input_ids = torch.tensor(question_tokenizer.convert_tokens_to_ids(tokens)).unsqueeze(0)
        question_outputs = question_model.generate(
            input_ids,
            do_sample=True, 
            max_length=1024, 
            top_k=50, 
            top_p=0.95, 
            eos_token_id=e_question,
            early_stopping=True,
            bad_words_ids=[[unk], [s_question], [s_answer]]
        )
        question_start_idx = question_outputs.tolist()[0].index(s_question) + 1
        decoded_question = decoding(question_outputs.tolist()[0][question_start_idx:-1], question_tokenizer)
        questions.append(decoded_question)
    
    return questions
    

input_text = '지미 카터는 조지아주 섬터 카운티 플레인스 마을에서 태어났다. 조지아 공과대학교를 졸업하였다. 그 후 해군에 들어가 전함·원자력·잠수함의 승무원으로 일하였다. 1953년 미국 해군 대위로 예편하였고 이후 땅콩·면화 등을 가꿔 많은 돈을 벌었다. 그의 별명이 "땅콩 농부" (Peanut Farmer)로 알려졌다.'

answer_list = get_answer_list(input_text)
question_list = get_question_list(input_text, answer_list)

for i in range(len(answer_list)):
    print(question_list[i] + '\t->\t' + answer_list[i])
