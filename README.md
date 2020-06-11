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

input: 이순신은 조선
- 이순신은 조선왕조의 충신 최창원(<unk><unk><unk>)이 조선왕조를 따라 영현으로 진격해오는 것을 시작으로 한성을 침공, 한성을 수복하고 한성을 점령하는 과정에서 이순신의 공세(<unk><unk>)를 저지하려 했다.
- 이순신은 조선 최고의 명승지로 평가받는다.
- 이순신은 조선 태조 이성계의 충신으로 조선 왕조의 사신과 사신의 아들이다.
- 이순신은 조선 태조 때의 학자.

input: 
-  밤하늘의 별자리를 지나 서쪽으로 동쪽으로 이동하면서 북쪽으로 방향을 바<unk>던 천혜의 형상은 천<unk>으로 <unk>어냈다.</s><s> 동쪽으로 가면서 황하와 서쪽으로 이동하면서 서쪽으로 <unk>아지른 <unk>아지른 <unk>한 <unk>이 흩어져 있다.
 북쪽은 북류하며 북서쪽으로 가면서 서북쪽으로 방향을 <unk>다.
 이 천혜의 형상은 남북 
 36<unk>30년이고 북쪽은 
 북류하며 남서쪽으로 가면서 남북 길이가 27<unk>350년이다.


4. Paraphrasing finetuning

5. Machine reading comprehension (MRC) finetuning








