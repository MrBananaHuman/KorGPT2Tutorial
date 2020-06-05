# KoGPT2PretraingTutorial
Tutorial for pretraining Korean GPT-2 model

Sample model download: https://drive.google.com/drive/folders/124Uux07pym2YaCeQKQWNhzhLNeIlLm7r?usp=sharing


1. make vocab
- python make_tokenizer

2. GPT-2 training from scratch (if you want to train GPT-2 from existing model, add the argument '--init_model')
- python pretrain_gpt2.py --do_train --do_eval --eval_data_file=pretrain_data/datas/sample_text.txt --model_type=gpt2 --train_data_file=pretrain_data/total_pretrain_data.txt --num_train_epochs=1

3. Text generation test

4. Paraphrasing finetuning

5. Machine reading comprehension (MRC) finetuning








