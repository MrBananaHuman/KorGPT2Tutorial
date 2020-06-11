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
-  밤하늘의 별과 비슷한 것인데도 불구하고 하늘엔 태양이나 어두운 기운이 스며 있고, 눈이 하늘보다 높지만 그마저도 태양이나 어두운 색이었다. 빛의 별이나 태양의 별과 마찬가지로 하늘보다 훨씬 높고 밝은 별이 더 많았을 것이다. 하늘엔 태양이나 어두운 기운이 가득 고여 있고, 하늘이 더 밝게 빛났으며, 이 별을 관측하는 것도 하늘의 별과 같은 것이라고 생각했다. 그 별들은 태양을 처럼, 밝은 별들이 더 밝게 빛났었다. 그 별이 빛난 별과 달리 태양이 빛난 별이 더 더 밝게 빛났기 때문이다. 별의 별들은 하늘의 별과 별에 대해 별과 별 사이의 차이점을 구분하였다.


4. Paraphrasing finetuning

5. Machine reading comprehension (MRC) finetuning








