from tokenizers.implementations import SentencePieceBPETokenizer
from tokenizers.processors import BertProcessing

tokenizer = SentencePieceBPETokenizer(
    "tokenizer/vocab.json",
    "tokenizer/merges.txt",
)
tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)

result = tokenizer.encode("이순신은 조선 중기의 무신이다.")
print(result.ids)
print(result.tokens)

result = tokenizer.encode("이순신은")
print(result.ids)
print(result.tokens)

