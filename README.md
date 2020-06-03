# KoGPT2PretraingTutorial
Tutorial for pretraining Korean GPT-2 model


1. make vocab
- python make_tokenizer

2. GPT-2 training
- python pretrain_gpt2.py --do_train --do_eval --eval_data_file=pretrain_data/datas/sample_text.txt --model_type=gpt2 --train_data_file=pretrain_data/total_pretrain_data.txt --num_train_epochs=1
