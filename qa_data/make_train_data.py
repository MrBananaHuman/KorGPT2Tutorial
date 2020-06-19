import json

data = open('KorQuAD_v1.0_train.json', 'r', encoding='utf-8')
question_output = open('qustion_train_data.txt', 'w', encoding='utf-8')
answer_output = open('answer_train_data.txt', 'w', encoding='utf-8')

data_json = json.load(data)

qa_datas = data_json['data']
for qa_data in qa_datas:
    paras = qa_data['paragraphs']
    for para in paras:
        context = para['context']
        answer_output.write(context + '\t')
        qas = para['qas']
        answers = set()
        for qa in qas:
            answer = qa['answers'][0]['text']
            question = qa['question']
            if answer not in answers:
                answer_output.write(answer + '|')
                answers.add(answer) 
            question_output.write(context + '\t' + answer + '\t' + question + '\n')
        answer_output.write('\n')

question_output.close()
answer_output.close()    
