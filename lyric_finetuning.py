from transformers import GPT2LMHeadModel, GPT2Config, AdamW
from new_tokenizer import MyTokenizer
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

vocab_file_path = '../tokenizer/vocab.json'
merge_file_path = '../tokenizer/merges.txt'

tokenizer = MyTokenizer(vocab_file_path, merge_file_path)
config = GPT2Config(vocab_size=52000)
model = GPT2LMHeadModel(config)

model_dir = '../KorGPT-2SampleModel/pytorch_model.bin'

model.load_state_dict(torch.load(model_dir), strict=False)
model.to('cpu')

ATTR_TO_SPECIAL_TOKEN = ['<song>', '</song>']

def add_special_tokens_(model, tokenizer):
    orig_num_tokens = tokenizer.get_vocab_size()
    tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
    num_added_tokens = len(ATTR_TO_SPECIAL_TOKEN)
    model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens + 1)

add_special_tokens_(model, tokenizer)
b_song = tokenizer.convert_tokens_to_ids('<song>')
e_song = tokenizer.convert_tokens_to_ids('</song>')

class LyricDataSet(Dataset):
    def __init__(self, file_path):
        self.data = []
        self.file_path = file_path
        
    def split_songs(self, lines):
        songs = []
        single_song = []
        for line in lines:
            line = line.strip()
            if line == '':
                if len(single_song) > 5:
                    songs.append(single_song)
                single_song = []
            else:
                single_song.append(line)
        return songs
    
    def load_data(self):
        lyric_file = open(self.file_path, 'r', encoding='utf-8')
        lyric_lines = lyric_file.readlines()
        lyric_file.close()
        
        song_list = self.split_songs(lyric_lines)
        for song in song_list:
            song_data = ['<song>']
            for line in song:
                tokenized_line = ['<s>'] + tokenizer.tokenize(line) + ['</s>']
                if len(song_data) + len(tokenized_line) < 1024:
                    song_data += tokenized_line
                else:
                    break
            song_data += ['</song>']
            padded_song_data = song_data + ['<pad>'] * (1024 - len(song_data))
            self.data.append(torch.tensor(tokenizer.convert_tokens_to_ids(padded_song_data)).unsqueeze(0))
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        return item

lyric_file_path = 'lyric_data/preprocessed_data.txt'
lyric_data = LyricDataSet(lyric_file_path)
lyric_data.load_data()
lyric_data_loader = DataLoader(lyric_data, batch_size=4, shuffle=True)

optimizer = AdamW(model.parameters(), lr=1e-4, correct_bias=True)

epochs = 5
for epoch in range(epochs):
    for data in lyric_data_loader:
        optimizer.zero_grad()

        data= data.transpose(1,0)
        
        outputs = model(data, labels=data)
        loss, logits = outputs[:2]
        loss.backward()
        optimizer.step()
        print('epoch = {} loss = {}' . format(epoch, loss))

torch.save(model.state_dict(), '/'.join(model_dir.split('/')[:-1]) + '/lyric_model.bin' )
        



    
