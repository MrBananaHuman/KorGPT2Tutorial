from transformers import GPT2LMHeadModel, GPT2Config, AdamW
from new_tokenizer import MyTokenizer
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json

vocab_file_path = '../tokenizer/vocab.json'
merge_file_path = '../tokenizer/merges.txt'

tokenizer = MyTokenizer(vocab_file_path, merge_file_path)
config = GPT2Config(vocab_size=52000)
model = GPT2LMHeadModel(config)

model_dir = '../model/pytorch_model.bin'

model.load_state_dict(torch.load(model_dir), strict=False)
model.to('cuda')

ATTR_TO_SPECIAL_TOKEN = ['<social>', '<economy>', '<world>', '<science>', '<sports>', '<politics>', '<entertainment>', '<it>', '<title>', '</title>']
category_map = {'사회':'<social>', '경제':'<economy>', '세계':'<world>', 'IT/과학':'<science>', '스포츠':'<sports>', '정치':'<politics>', '연예':'<entertainment>', 'IT':'<it>'}

def add_special_tokens_(model, tokenizer):
    orig_num_tokens = tokenizer.get_vocab_size()
    tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
    num_added_tokens = len(ATTR_TO_SPECIAL_TOKEN)
    model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens + 1)

add_special_tokens_(model, tokenizer)
b_title = tokenizer.convert_tokens_to_ids('<title>')
e_title = tokenizer.convert_tokens_to_ids('</title>')

class NewsDataSet(Dataset):
    def __init__(self, file_path):
        self.data = []
        self.file_path = file_path
    
    def load_data(self):
        news_file = open(self.file_path, 'r', encoding='utf-8')
        news_lines = news_file.readlines()
        news_file.close()
        
        for line in news_lines:
            line = line.strip()
            article_data = json.loads(line)
            category = category_map[article_data['category']]
            article = article_data['article']
            # summary = article_data['contents']
            title = article_data['title']
            splited_article = article.split('\n')
            # splited_summary = summary.split('\n')
            article_data = []
            # summary_data = [category, '<summary>']
            title_data = [category, '<title>']
            tokenized_line = ['<s>'] + tokenizer.tokenize(title) + ['</s>']
            title_data += tokenized_line
            title_data += ['</title>']
            # for single_line in splited_summary:
            #     tokenized_line = ['<s>'] + tokenizer.tokenize(title) + ['</s>']
            #     summary_data += tokenized_line
            # summary_data += ['</summary>']
            for single_line in splited_article:
                tokenized_line = tokenizer.tokenize(single_line)
                if len(article_data) + len(tokenized_line) < 1024 - len(title_data):
                    article_data += tokenized_line
                else:
                    break
            article_data += title_data
            padded_article_data = article_data + ['<pad>'] * (1024 - len(article_data))
            self.data.append(torch.tensor(tokenizer.convert_tokens_to_ids(padded_article_data)).unsqueeze(0))
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        return item

news_file_path = 'news_data/sample_news_data.txt'
news_data = NewsDataSet(news_file_path)
news_data.load_data()
print(len(news_data.data))
news_data_loader = DataLoader(news_data, batch_size=4, shuffle=True)

optimizer = AdamW(model.parameters(), lr=1e-4, correct_bias=True)

epochs = 5
count = 0

avg_loss = (0.0, 0.0)
for epoch in range(epochs):
	for data in news_data_loader:
		optimizer.zero_grad()
		data = data.transpose(1,0)
		data = data.to('cuda')

		outputs = model(data, labels=data)
		loss, logits = outputs[:2]
		loss = loss.to('cuda')
		loss.backward()
		avg_loss = (avg_loss[0] * 0.99 + loss, avg_loss[1] * 0.99 + 1.0)
		optimizer.step()

		if count % 200 == 0:
			print('epoch no.{0} train no.{1}  loss = {2:.5f} avg_loss = {3:.5f}' . format(epoch, count, loss, avg_loss[0] / avg_loss[1]))
		count += 1

torch.save(model.state_dict(), '/'.join(model_dir.split('/')[:-1]) + '/lyric_model.bin' )