from transformers import GPT2LMHeadModel, GPT2Config
from new_tokenizer import MyTokenizer
import torch
import kss

ATTR_TO_SPECIAL_TOKEN = ['<social>', '<economy>', '<world>', '<science>', '<sports>', '<politics>', '<entertainment>', '<it>', '<title>', '</title>']
category_map = {'사회':'<social>', '경제':'<economy>', '세계':'<world>', 'IT/과학':'<science>', '스포츠':'<sports>', '정치':'<politics>', '연예':'<entertainment>', 'IT':'<it>'}

vocab_file_path = '../tokenizer/vocab.json'
merge_file_path = '../tokenizer/merges.txt'

tokenizer = MyTokenizer(vocab_file_path, merge_file_path)
bos = tokenizer.convert_tokens_to_ids('<s>')
eos = tokenizer.convert_tokens_to_ids('</s>')
pad = tokenizer.convert_tokens_to_ids('<pad>')
unk = tokenizer.convert_tokens_to_ids('<unk>')

config = GPT2Config(vocab_size=52011, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0, summary_first_dropout=0)
model = GPT2LMHeadModel(config)

# model_dir = '../KorGPT-2SampleModel/lyric_model.bin'
model_dir = '../model/summary_model.bin'

model.load_state_dict(torch.load(model_dir), strict=False)
model.to('cpu')

def add_special_tokens_(model, tokenizer):
    orig_num_tokens = tokenizer.get_vocab_size()
    tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
    num_added_tokens = len(ATTR_TO_SPECIAL_TOKEN)
    model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens + 1)

add_special_tokens_(model, tokenizer)
b_title = tokenizer.convert_tokens_to_ids('<title>')
e_title = tokenizer.convert_tokens_to_ids('</title>')

def encoding(category, text):
    sent_list = kss.split_sentences(text)
    tokens = []
    for sent in sent_list:
        tokenized_sentence = tokenizer.tokenize(sent)
        if len(tokens) + len(tokenized_sentence) < 912:
            tokens += ['<s>'] + tokenized_sentence + ['</s>']
        else:
            break
    tokens += [category]
    tokens += ['<title>']
    return torch.tensor(tokenizer.convert_tokens_to_ids(tokens)).unsqueeze(0)

def decoding(input_ids, ids):
    print(len(input_ids[0]))
    return tokenizer.convert_ids_to_tokens(ids[0][len(input_ids[0]):])

input_ids = encoding('<social>', '16일 코로나19 신규 확진자가 34명으로 집계됐다. 그중 국내 발생 사례는 21명이고 나머지 13명은 해외유입 사례다.\n질병관리본부 중앙방역대책본부는 이날 오전 0시 기준 코로나19 확진자가 전날 대비 34명 늘어난 1만2155명이라고 밝혔다. 국내 신규 감염 사례는 지난 10일 이후 40~50명선을 유지했으나 지난 14일부터 20명대로 내려왔다.\n국내 발생 사례를 지역별로 보면 서울 11명, 인천 2명, 경기 4명으로 수도권에서 17명이 나왔다. 그밖에 대전에서 3명, 경남에서 1명이 추가 확진됐다.\n13건의 해외 유입 사례의 경우 검역소에서 9명이 새로 확진을 받았다. 서울, 부산, 경기, 경남에서도 1명씩 신규 확진자가 나왔다.\n수도권의 경우 ‘국내 발생’과 ‘해외 유입’을 모두 합하면 19명의 신규 확진자가 나왔다.\n코로나19로 인한 사망자는 하루새 1명 늘어나 총 278명이다.\n완지로 격리 해제된 사람은 30명이 추가돼 총 1만760명으로 집계됐다. 격리 중인 이는 전날보다 3명 늘어난 1117명이다.')

sample_outputs = model.generate(
    input_ids,
    do_sample=True, 
    max_length=1024, 
    top_k=50, 
    top_p=0.95, 
    eos_token_id=e_title,
    pad_token_id=pad,
    early_stopping=True,
    bad_words_ids=[[unk]]
)
print(decoding(input_ids, sample_outputs.tolist()))

