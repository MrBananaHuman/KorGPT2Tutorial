import tokenizers as tk

path = 'pretrain_data/book_sentences.txt'
tokenizer = tk.SentencePieceBPETokenizer()
tokenizer.train(files=path, vocab_size=52000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

tokenizer.save(".", "saltgpt2")
