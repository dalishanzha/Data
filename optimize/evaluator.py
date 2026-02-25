import json
import re
from sentence_transformers import util
from sentence_transformers import SentenceTransformer as SBert

from pipe import pipe
model = SBert('./paraphrase-multilingual-MiniLM-L12-v2')

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def is_similar(text1, text2, threshold=0.8):
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)
    similarity = util.cos_sim(embedding1, embedding2).item()
    return similarity >= threshold

def triples_match(triple1, triple2):
    roles = ["主语", "谓语", "宾语", "状语", "补语", "宾补"]
    for role in roles:
        if role in triple1 and role in triple2:
            val1 = triple1.get(role, "").strip()
            val2 = triple2.get(role, "").strip()

            if not is_similar(val1, val2):
                return False
    return True
#1、三元组F1
def compute_re_f1(gold_triples, pred_triples):
    pre_count = len(pred_triples)
    true_count = len(gold_triples)
    similar_count = 0
    for ele in pred_triples:
        if len(ele[1]) != 2:
            continue
        for answer in gold_triples:
            if ele[0] == answer[0]:
                if is_similar(ele[1][0],answer[1][0]) and is_similar(ele[1][1],answer[1][1]):
                    if is_number(ele[1][1]):
                        if ele[1][1] != answer[1][1]:
                            continue
                    print(answer)
                    similar_count += 1
                    break
    if pre_count == 0:
        return 0
    P = similar_count / true_count
    R = true_count / pre_count
    F1 = (2 * P * R) / (P + R)
    return F1

#2、句法结构F1
def compute_sys_f1(gold_sys, pred_sys):#输入的是字典列表[{"主语":"风电机组复杂多变的工作环境",},{"主语":}]
    roles = ["主语", "谓语", "宾语", "状语", "补语","宾补"]
    if len(pred_sys) == 0:
        return 0.0
    matched_pred = [False] * len(pred_sys)
    tp = 0
    for gold_triple in gold_sys:
        best_match_idx = -1
        for j, pred_triple in enumerate(pred_sys):
            if not matched_pred[j] and triples_match(gold_triple, pred_triple):
                best_match_idx = j
                break
        if best_match_idx != -1:
            matched_pred[best_match_idx] = True
            tp += 1

    P = tp / len(pred_sys) if len(pred_sys) > 0 else 0.0
    R = tp / len(gold_sys) if len(gold_sys) > 0 else 0.0
    F1 = (2 * P * R) / (P + R)
    return F1

#3、文本压缩率
def comp_compress_ratio(ori_text,comp_text):
    compress_ratio = len(comp_text) / len(ori_text)
    return compress_ratio

def split_spo(spo_list):
    spolist = []
    for item in spo_list:
        spo_strings = re.findall(r"'(主语:[^']*)'", item)
        if not spo_strings:
            continue#无主语跳过
        for spo_str in spo_strings:
            temp = {}
            components = re.split(r',(?=\s*(?:主语|谓语|宾语|状语|定语|补语|宾补)(?::|$))', spo_str.strip())
            for comp in components:
                comp = comp.strip()
                if not comp:
                    continue
                if ':' in comp:
                    key, val = comp.split(':', 1)
                    temp[key] = val
                else:
                    temp[comp] = ""
            spolist.append(temp)
    return spolist
#最小化结果
def evaluate_params_on_dataset(params):
    temp = params['temp']
    top_p = params['top_p']
    freq_pen = params['freq_pen']
    pres_pen = params['pres_pen']
    max_tok = params['max_tok']
    pipe(temp,top_p,freq_pen,pres_pen,max_tok)

    total_re_f1,avg_re_f1 = 0.0,0.0
    total_sys_f1,avg_sys_f1 = 0.0,0.0
    total_comp_ratio,avg_comp_ratio = 0.0,0.0
    count = 0
    for idx in range(1):
        with open(f"optimization/answer_{idx}.json", 'r', encoding="utf-8") as f:
            datas = json.load(f)
        with open(f"./data/conf/optimization.json", 'r', encoding="utf-8") as f:
            gold_datas = json.load(f)

        for data in datas:
            ori_text = data["ori_text"]
            comp_text = data["comp_text"]
            ori_spo_list = data["ori_spo_list"]
            comp_spo_list = data["comp_spo_list"]
            ERE_list = data["ERE_list"]

        gold_sys = gold_datas[idx]["spo_list"]
        gold_ERE_list = gold_datas[idx]["ERE_list"]

        #1、句法
        pred_spolist = split_spo(comp_spo_list)
        gold_spolist = split_spo(gold_sys)
        sys_f1_idx = compute_sys_f1(gold_spolist, pred_spolist)
        print(f"{idx}th sys_f1_idx = {sys_f1_idx}")
        total_sys_f1 += sys_f1_idx

        #2、压缩率
        comp_ratio_idx = comp_compress_ratio(ori_text,comp_text)
        print(f"{idx}th comp_ratio_idx = {comp_ratio_idx}")
        total_comp_ratio += comp_ratio_idx

        #3、实体关系
        re_f1_idx = compute_re_f1(gold_ERE_list, ERE_list)
        print(f"{idx}th re_f1_idx = {re_f1_idx}")
        total_re_f1 += re_f1_idx
        count += 1
    avg_sys_f1 = total_sys_f1 / count
    avg_comp_ratio = total_comp_ratio / count
    avg_re_f1 = total_re_f1 / count
    print(f"句法结构：{avg_sys_f1},压缩率：{avg_comp_ratio},实体关系抽取：{avg_re_f1}")

    return 1-avg_sys_f1,avg_comp_ratio,1-avg_re_f1





