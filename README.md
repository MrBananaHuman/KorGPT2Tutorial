# KorGPT2PretraingTutorial
Tutorial for pretraining Korean GPT-2 model

Sample model download: https://drive.google.com/drive/folders/124Uux07pym2YaCeQKQWNhzhLNeIlLm7r?usp=sharing
(100,000 sentences, 1 epoch) 

1. make vocab from corpus
- python make_tokenizer

2. GPT-2 training from scratch (if you want to train GPT-2 from existing model, add the argument '--init_model')
- python pretrain_gpt2.py --do_train --do_eval --eval_data_file=pretrain_data/datas/sample_text.txt --model_type=gpt2 --train_data_file=pretrain_data/total_pretrain_data.txt --num_train_epochs=1

3. Text generation test
- python generation_text.py
- 이순신은 조선왕조의 충신 최창원(<unk><unk><unk>)이 조선왕조를 따라 영현으로 진격해오는 것을 시작으로 한성을 침공, 한성을 수복하고 한성을 점령하는 과정에서 이순신의 공세(<unk><unk>)를 저지하려 했다.
- 이순신은 조선 최고의 명승지로 평가받는다.
- 이순신은 조선 태조 이성계의 충신으로 조선 왕조의 사신과 사신의 아들이다.
- 이순신은 조선 태조 때의 학자.

4. Paraphrasing finetuning

5. Machine reading comprehension (MRC) finetuning








