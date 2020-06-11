# KorGPT2Tutorial
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
-  이순신은 조선 시대 이순신은 무솔한 이순신의 모습을 생생하게 재현했다.
 이순신은 일제강점기 조선의 한 축을 책임지고 있다.
 임진왜란 당시 이순신은 무솔한 이순신을 살려내 이순신의 모습을 담은 전시회를 가졌다.
 이순신은 '역사'로서 이순신을 대표하는 이순신답게, 이순신은 일제강점기 조선 조선 조선 시대의 대표적인 작품 중 하나다.
 이순신은 조선의 역사와 전통을 담아내기 위해 선조들의 관심을 끌었고, 조선 시대 조선총독부에서 이순신이 보여준 이순신의 모습은 이순신과 이순신을 가장 가까이에서 볼 수 있는 대목이다.
 이순신은 1900년 전란 중인 조선총독부 경무국과 조선군사정관을 지냈다.
 이순신은 일제강점기 조선의 역사를 한 획을 그었다.
 이순신 시대 이순신은 조선총독부 경무국장으로 근무했다.
 그 동안 이순신은 조선총독부 경무국과 조선군의 역사를 다년간 지냈다.
 그 동안 이순신의 업적을 재현하기 위해 이순신의 발자취를


input: 밤하늘의 별
-  밤하늘의 별과 비슷한 것인데도 불구하고 하늘엔 태양이나 어두운 기운이 스며 있고, 눈이 하늘보다 높지만 그마저도 태양이나 어두운 색이었다. 빛의 별이나 태양의 별과 마찬가지로 하늘보다 훨씬 높고 밝은 별이 더 많았을 것이다. 하늘엔 태양이나 어두운 기운이 가득 고여 있고, 하늘이 더 밝게 빛났으며, 이 별을 관측하는 것도 하늘의 별과 같은 것이라고 생각했다. 그 별들은 태양을 처럼, 밝은 별들이 더 밝게 빛났었다. 그 별이 빛난 별과 달리 태양이 빛난 별이 더 더 밝게 빛났기 때문이다. 별의 별들은 하늘의 별과 별에 대해 별과 별 사이의 차이점을 구분하였다.


4. lyric finetuning

5. Machine reading comprehension (MRC) finetuning

6. Paraphrasing finetuning
- https://github.com/MrBananaHuman/KoGPT2ForParaphrasing









