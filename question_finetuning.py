from transformers import GPT2LMHeadModel, GPT2Config, AdamW
from new_tokenizer import MyTokenizer
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import kss

vocab_file_path = '../tokenizer/vocab.json'
merge_file_path = '../tokenizer/merges.txt'

tokenizer = MyTokenizer(vocab_file_path, merge_file_path)
config = GPT2Config(vocab_size=52000)
model = GPT2LMHeadModel(config)

model_dir = '../KorGPT-2SampleModel/pytorch_model.bin'

model.load_state_dict(torch.load(model_dir), strict=False)
model.to('cuda')

ATTR_TO_SPECIAL_TOKEN = ['<answer>', '</answer>', '<question>', '</question>']

def add_special_tokens_(model, tokenizer):
    orig_num_tokens = tokenizer.get_vocab_size()
    tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
    num_added_tokens = len(ATTR_TO_SPECIAL_TOKEN)
    model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens + 1)

add_special_tokens_(model, tokenizer)
b_answer = tokenizer.convert_tokens_to_ids('<answer>')
e_answer = tokenizer.convert_tokens_to_ids('</answer>')
b_question = tokenizer.convert_tokens_to_ids('<question>')
e_question = tokenizer.convert_tokens_to_ids('</question>')

class QuestionDataSet(Dataset):
    def __init__(self, file_path):
        self.data = []
        self.file_path = file_path
    
    def load_data(self):
        question_file = open(self.file_path, 'r', encoding='utf-8')
        question_lines = question_file.readlines()
        question_file.close()
        
        for line in question_lines:
            line = line.strip()
            context = line.split('\t')[0]
            answer = line.split('\t')[1]
            question = line.split('\t')[2]
            
            total_tokens = []
            
            qa_tokens = ['<answer>'] + tokenizer.tokenize(answer) + ['</answer>'] + ['<question>'] + tokenizer.tokenize(question) + ['</question>']
            
            for single_line in kss.split_sentences(context):
                tokenized_single_line = ['<s>'] + tokenizer.tokenize(single_line) + ['</s>']
                if len(total_tokens) + len(qa_tokens) + len(tokenized_single_line) < 1024:
                    total_tokens += tokenized_single_line
                else:
                    break
            total_tokens += qa_tokens
            padded_total_tokens = total_tokens + ['<pad>'] * (1024 - len(total_tokens))
            self.data.append(torch.tensor(tokenizer.convert_tokens_to_ids(padded_total_tokens)).unsqueeze(0))
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        return item

question_file_path = 'qa_data/question_train_data.txt'
question_data = QuestionDataSet(question_file_path)
question_data.load_data()
print(len(question_data.data))
question_data_loader = DataLoader(question_data, batch_size=4, shuffle=True)

optimizer = AdamW(model.parameters(), lr=1e-4, correct_bias=True)

epochs = 5
count = 0

avg_loss = (0.0, 0.0)
for epoch in range(epochs):
	for data in question_data_loader:
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

torch.save(model.state_dict(), '/'.join(model_dir.split('/')[:-1]) + '/question_model.bin' )
