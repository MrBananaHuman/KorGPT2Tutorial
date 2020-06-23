import json

data = open('github-typo-corpus.v1.0.0.jsonl', 'r', encoding='utf-8')
output = open('kor_typo_data.txt', 'w', encoding='utf-8')

def is_contain_kor(json_text):
    try:
        json_data = json.loads(json_text)
    except:
        print(json_text)
        return False
    edits = json_data['edits']
    for edit in edits:
        if edit['src']['lang'] == 'kor':
            return True
        if edit['tgt']['lang'] == 'kor':
            return True
    return False

lines = data.readlines()

def get_data(json_text):
    srcs = []
    tgts = []
    try:
        json_data = json.loads(json_text)
    except:
        return srcs, tgts
    edits = json_data['edits']
    for edit in edits:
        src = edit['src']['text']
        src_lang = edit['src']['lang']
        tgt = edit['tgt']['text']
        tgt_lang = edit['tgt']['lang']
        #if src_lang == 'kor' and tgt_lang == 'kor':
        srcs.append(src)
        tgts.append(tgt)
    return srcs, tgts

for line in lines:
    line = line.strip()
    if is_contain_kor(line):
        srcs, tgts = get_data(line)
        for i in range(len(srcs)):
            output.write(srcs[i] + '\t' + tgts[i] + '\n')
    else:
        continue

output.close()