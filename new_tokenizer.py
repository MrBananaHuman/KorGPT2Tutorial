from tokenizers.implementations import SentencePieceBPETokenizer
from tokenizers.processors import BertProcessing

from transformers.tokenization_utils import PreTrainedTokenizer, PreTrainedTokenizerFast

import json

class MyTokenizer():

    def __init__(self, vocab_file_path, merge_file_path):
        self.tokenizer = SentencePieceBPETokenizer(vocab_file_path, merge_file_path)
        self.unknown_token = self.tokenizer.token_to_id("<unk>")
        self._pad_token = "<pad>"
        self.pad_token_id = self.tokenizer.token_to_id("<pad>")
        self.max_len = 1024
        self.max_len_single_sentence = 1024
        self.init_kwargs = {}
        self.added_tokens_encoder = {}
        self.unique_added_tokens_encoder = set()
        self.added_tokens_decoder = {}
        self.unexpected_sep_token = ['<pad>', '<unk>', '<eos>', '<sos>']

        self.encoder = self.tokenizer.get_vocab()
        self.decoder = dict(map(reversed, self.encoder.items()))

    
    def tokenize(self, text):
        if text in self.unexpected_sep_token:
            return text
        return self.tokenizer.encode(text).tokens
    
    def convert_tokens_to_ids(self, tokens):
        ids = []
        if isinstance(tokens, str):
            if tokens in self.encoder:
                return self.encoder[tokens]
            else:
                return self.unknown_token
        for token in tokens:
            if token in self.encoder:
                ids.append(self.encoder[token])
            else:
                ids.append(self.unknown_token)
        return ids
    
    def convert_ids_to_tokens(self, ids):
        sentence = ''
        for id_ in ids:
            sentence += self.decoder[id_]
        sentence = sentence.replace('▁', ' ')
        return sentence.strip()
            
    
    def build_inputs_with_special_tokens(self, ids):
        return ids

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def add_special_tokens(self, new_tokens):
        cnt = 0
        self.tokenizer.add_special_tokens(new_tokens)
        return cnt
    # def save_pretrained(self, save_directory)
        # self.tokenizer.

if __name__ == '__main__':
    vocab_file_path = 'salt_tokenizer/vocab.json'
    merge_file_path = 'salt_tokenizer/merges.txt'
    tokenizer = MyTokenizer(vocab_file_path, merge_file_path)
    sentence = "이순신은 조선 중기의 무신이다."
    tokens = tokenizer.tokenize(sentence)
    print(tokens)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    print(ids)
    ids2 = tokenizer.build_inputs_with_special_tokens(ids)
    print(ids2)
    print(tokenizer.convert_ids_to_tokens(ids))
    