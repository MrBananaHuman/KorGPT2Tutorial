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

input: 그 노인은
- 그 노인은 <unk>장히 큰 소리로 말했다. "<unk><unk>. " 노인의 말에 노인은 고개를 끄덕이며 말했다."<unk>, 정말 고맙네. " 노인은 노인의 말을 듣고 고개를 끄덕였다."아니, 그 노인?" 노인의 말이 끝나기도 전에 노인은 다시 물었다."그렇지 않나?" 노인의 물음에 노인은 대답하지 않았다."아, 이 노인??" 노인은 노인이 고개를 저었다."그럼, 이 노인은 어떻게 된 거야??" 노인이 대답했다."어떻게 된 거야?" 노인의 대답은 노인의 대답이었다."그러면 어떻게 될까?" 노인의 말은 노인의 대답에 노인의 대답이 없었다."그렇다면 어떻게 되는 거야?" 노인이 물었다. 노인은 노인에게 물었다. "그 노인이 어떻게 되나. " 노인이 말했다. "그 노인은 무슨 말을 하는가?" 노인의 말에는 노인이 대답할 말이 없었다. 노인은 그 노인의 대답을 들을 수 없었다. 노인의 말대로 노인은 노인에게 물었다. "이 노인이 얼마나 큰 소리를 하고 있는지 알 수 있을 것 같나?" 노인은 노인은 대답했다. "그 노인을 어떻게 할 수 있겠나?" 노인이 노인의 질문에 노인은 대답하지 않았다. 노인은 이 노인이 노인에게 대답했다. "이 노인은 어떤 말을 하고 있는지는 알 수가 없어. " 노인도 노인은 말을 하지 않았다. 노인의 말과는 달리 노인은 노인을

4. Paraphrasing finetuning

5. Machine reading comprehension (MRC) finetuning








