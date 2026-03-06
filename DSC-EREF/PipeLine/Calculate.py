from sentence_transformers import util
from sentence_transformers import SentenceTransformer as SBert
import json
import copy
import csv
# 加载预训练中文语义模型
model = SBert('../paraphrase-multilingual-MiniLM-L12-v2')
#计算相似性
def is_similar(text1, text2, threshold=0.8):
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)
    similarity = util.cos_sim(embedding1, embedding2).item()
    return similarity >= threshold

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def compute_f1(p, r):
    if p + r == 0:
        return 0
    return (2 * p * r) / (p + r)

#按文件加载预测数据，并与标注结果进行相似度对比并统计命中数
def compute():
    sanza_data = []
    total_answer = 0
    stanza_hit = 0
    total_sanza = 0
    for i in range(500):#
        current_hit_sanza = 0
        with open(f"../Result/dcom_answer_optimization/results_{i}.json", 'r', encoding="utf-8") as f:
            pre_sanza = json.load(f)
            total_sanza += len(pre_sanza)
            current_sanza = len(pre_sanza)
        with open(f"../data/conf/ori.json", 'r', encoding="utf-8") as f:
            text = json.load(f)
            answer = text[i]['answer']
            total_answer += len(answer)
            current_answer = len(answer)
            answer_sanza = copy.deepcopy(answer)

        for element in pre_sanza:
            if element in answer_sanza:
                print(element)
                stanza_hit += 1
                current_hit_sanza += 1
                answer_sanza.remove(element)
            else:
                for answer in answer_sanza:
                    if element[0] == answer[0]:
                        if is_similar(element[1][0], answer[1][0]) and is_similar(element[1][1], answer[1][1]):
                            print(answer)
                            stanza_hit += 1
                            current_hit_sanza += 1
                            answer_sanza.remove(answer)

        print(f"idx={i}")
        print(current_hit_sanza,current_answer)
        print(current_sanza)
        if current_hit_sanza != 0:
            print(f"P={current_hit_sanza / current_sanza},R = {current_hit_sanza / current_answer},F1 = {compute_f1(current_hit_sanza / current_sanza,current_hit_sanza / current_answer)}")
        else:
            print('0,0,0')
        sanza_data.append([current_hit_sanza, current_sanza, current_answer])

        def write_to_csv(filename, data):
            with open(filename, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(["命中数", "总数", "实际答案数"])  # 写入表头
                writer.writerows(data)  # 写入所有数据行

        #write_to_csv("./results/last_句法依存文本详细结果.csv", sanza_data)

        #print("数据已成功写入CSV文件。")

        print('\n')
        print(stanza_hit, total_answer)
        print(total_sanza)
        print(f"{i}th P={stanza_hit / total_sanza},R = {stanza_hit / total_answer},F1 = {compute_f1(stanza_hit / total_sanza,stanza_hit / total_answer)}")
        print('\n')
    return stanza_hit,total_answer,total_sanza

stanza_hit,total_answer,total_sanza = compute()

print(f"stanza_hit={stanza_hit}")
print(f"total_sanza={total_sanza},total_answer={total_answer}")
#准确率计算 hit/total
stanza_p = stanza_hit/total_sanza
print(f"Sanza_P={100 * stanza_p:.2f}%")
print('\n')
#召回率计算 hit/answer
sanza_r = stanza_hit/total_answer
print(f"Sanza_R={100 * sanza_r:.2f}%")
print('\n')
#F1计算
sanza_f1 = compute_f1(stanza_p, sanza_r)
print(f"sanza_F1={sanza_f1:.4f}")


