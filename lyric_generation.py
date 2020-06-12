from transformers import GPT2LMHeadModel, GPT2Config
from new_tokenizer import MyTokenizer
import torch

ATTR_TO_SPECIAL_TOKEN = ['<song>', '</song>']

vocab_file_path = '../tokenizer/vocab.json'
merge_file_path = '../tokenizer/merges.txt'

tokenizer = MyTokenizer(vocab_file_path, merge_file_path)
bos = tokenizer.convert_tokens_to_ids('<s>')
eos = tokenizer.convert_tokens_to_ids('</s>')
pad = tokenizer.convert_tokens_to_ids('<pad>')
unk = tokenizer.convert_tokens_to_ids('<unk>')

config = GPT2Config(vocab_size=52003, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0, summary_first_dropout=0)
model = GPT2LMHeadModel(config)

model_dir = '../KorGPT-2SampleModel/lyric_model.bin'

model.load_state_dict(torch.load(model_dir), strict=False)
model.to('cpu')

def add_special_tokens_(model, tokenizer):
    orig_num_tokens = tokenizer.get_vocab_size()
    tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
    num_added_tokens = len(ATTR_TO_SPECIAL_TOKEN)
    model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens + 1)

add_special_tokens_(model, tokenizer)
b_song = tokenizer.convert_tokens_to_ids('<song>')
e_song = tokenizer.convert_tokens_to_ids('</song>')

def encoding(text):
    tokens = ['<song>', '<s>'] + tokenizer.tokenize(text)
    return torch.tensor(tokenizer.convert_tokens_to_ids(tokens)).unsqueeze(0)

def decoding(ids):
    return tokenizer.convert_ids_to_tokens(ids[0])

input_ids = encoding('하늘을 날아')

sample_outputs = model.generate(
    input_ids,
    do_sample=True, 
    max_length=1024, 
    top_k=50, 
    top_p=0.95, 
    eos_token_id=e_song,
    early_stopping=True,
    bad_words_ids=[[unk]]
)
print(decoding(sample_outputs.tolist()))

