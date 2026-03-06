from getapi import get_chat_res, build_message
import json
import ast
import re
from ltp import StnSplit
from concurrent.futures import ThreadPoolExecutor

def is_valid_triple(relation, head, tail):
    INVALID_TAIL_FOR_CONTAINS = {"输入","框架","数据"}
    INVALID_TAIL_FOR_APPLIED = {"SCADA数据"}
    INVALID_TAIL_FOR_PROPOSE = {"用","提出","采用"}
    if relation == "包含" or relation == "监测" or relation == "采集":
        tail_str = tail.lower()
        for word in INVALID_TAIL_FOR_CONTAINS:
            if word == tail_str:
                return False
    if relation == "应用于":
        tail_str = tail.lower()
        for word in INVALID_TAIL_FOR_APPLIED:
            if word == tail_str:
                return False

    if relation == "提出/尝试":
        tail_str = tail.lower()
        for word in INVALID_TAIL_FOR_PROPOSE:
            if word == tail_str:
                return False
    return True
def filter_elements(data,text):
    result = []
    for current in data:
        if len(current[1]) != 2:
            continue
        current_key = current[0]
        current_a, current_b = current[1][0], current[1][1]
        is_covered = False
        to_remove = []
        if not is_valid_triple(current_key, current_a, current_b):
            continue
        for idx, existing in enumerate(result):
            if existing[0] != current_key:
                continue
            exist_a, exist_b = existing[1][0], existing[1][1]
            if (current_a in exist_a) and (current_b in exist_b):
                is_covered = True
                break
            if (exist_a in current_a) and (exist_b in current_b):
                to_remove.append(idx)
        if not is_covered:
            for idx in reversed(to_remove):
                del result[idx]
            result.append(current)
    return result

def two_stage_relation_extraction(text,re_list):
    rel_list = []
    for key, value in re_list.items():
        rel_list.append(key)
    prompts = json.load(open("../Data/ere_order.json", 'r', encoding="utf-8"))
    stage1_prompt_list = prompts["stage1_prompt"]
    stage1_prompt = f'''给定文本为{text}，给定本体关系列表：{rel_list}。\n'''
    for ele in stage1_prompt_list:
        stage1_prompt += ele
    results = []
    mess = build_message('您是一个关系抽取专家，负责从给定的句子中识别存在的关系。。', stage1_prompt)
    res = get_chat_res('gpt', mess)
    if isinstance(res, list):
        res = ''.join(res)
    if '"' not in res:
        return results
    matches = re.findall(r'"(.*?)"', res)
    stage1_response = [match for match in matches if match in rel_list]
    print(stage1_response)

    if not stage1_response:
        return results
    def extract_entities(relation):
        for value in re_list[relation]:
            entity_1, entity_2 = value
            stage2_prompt_list = prompts["stage2_prompt"]
            stage2_prompt = f'''给定head实体类型'{entity_1}'、tail实体类型'{entity_2}'且之间的关系为'{relation}',文本内容：{text}\n'''
            for ele in stage2_prompt_list:
                stage2_prompt += ele
            mess = build_message('您是一个实体抽取专家，负责根据关系从给定的句子中识别对应的实体对。', stage2_prompt)
            res = get_chat_res('gpt', mess)
            matches = re.findall(r"'(.*?)'", res)
            stage2_response = [matches[i:i + 2] for i in range(0, len(matches), 2)]
            print(relation)
            for ele in stage2_response:
                print(f"ele={ele}")
                temp_result = [relation, ele]
                results.append(temp_result)

    with ThreadPoolExecutor(max_workers=3) as executor:
        executor.map(extract_entities, stage1_response)

    unique_results = [list(item) for item in set(
        tuple(tuple(pair) if isinstance(pair, list) else pair for pair in result) for result in results)]
    return unique_results
re_list = {
        #数据流
        "输入": [["数据/特征", "模型/算法/方法/组件"]],"输出": [["模型/算法/方法", "数据/结果/诊断结论"]],"监测":[["系统","物理量"]],
        #方法与目标
        "解决": [["方法", "问题"]],"应用于": [["模型/信号/方法/组件","数据/设备/场景/领域"]],
        #因果与关联
        "导致": [["方法/操作（原因）", "现象/故障/后果（结果）"],["故障", "现象/故障/后果（结果）"],["现象（原因）", "现象/故障/后果（结果）"]],
        #组成结构
        "包含":[["诊断方法","子诊断方法"],["信号","子信号"],["故障类型","子类"],["系统","子系统"],["工作流程","阶段"],["部件/组件/网络结构","子部件/子组件"],
                ["方法(技术)/分析方法","子方法/代表方法"],["数据/总物理量","子物理量"],["运行状态","子状态"],["载荷","力矩"]],
        #属性与描述
        "数量":[["参数","数值/范围"],["数据/物理量","数值/范围"],["计量对象","数值/范围"]],
        "型号": [["部件", "型号"]],
        "失效形式": [["部件", "失效模式"]],
        "定义":[["符号/变量/物理量","代表量/概念/描述"]],
        "特点":[["信号/系统/数据/能源/组件","属性/特征"]],
        #研究行为
        "提出/尝试": [["研究人/文献/章节/本文", "操作"],["研究人/文献/章节/本文", "方法/模型/网络"]],"介绍": [["研究者/文献", "原理/综述"]],
        "研究目标/方向":[["研究人/文献/章节/本文","目标"]],
        "建立/构建":[["研究人/文献/章节/本文","模型/组件/机制"],["研究人/文献/章节/本文","函数/数据集"]],
        #时间地点
        "采集时间": [["数据", "时间"]],"维修时间": [["部件", "时间"]],"采集地点": [["数据", "地点"]],
        #评价
        "效果": [["方法/技术", "功能/优势/达成效果"],["组件/模型", "功能/优势/达成效果"],["技术操作", "达成效果"]],
        "缺点/不足": [["方法/模型/技术", "问题/局限/缺陷"]]
}
data = json.load(open("../Result/compress_text.json", 'r', encoding="utf-8"))  # 字典列表
for i, elem in enumerate(data):  # elem是字典
    ori_text = elem['ori_text']
    spo_list = elem['spo_list_gpt']
    com_text = elem['com_text_nospo']
    spo_text = elem['com_text_spo_gpt']
    idx = elem['idx']
    spo_text_santa = elem['com_text_spo_santa']

    # 句法依存压缩文本抽取结果(优化)
    results_spocom = two_stage_relation_extraction(spo_text, re_list)
    results_spocom = filter_elements(results_spocom, spo_text)
    print(results_spocom)
    with open(f"../Result/dcom_answer_optimization/results_{idx}.json", "w", encoding='utf-8') as f:
        json.dump(results_spocom, f, indent=2, ensure_ascii=False)

'''
    # 句法依存压缩文本抽取结果
    results_spocom = two_stage_relation_extraction(spo_text, re_list)
    results_spocom = filter_elements(results_spocom,spo_text)
    print(results_spocom)
    with open(f"../Result/dcom_answer/results_{idx}.json", "w", encoding='utf-8') as f:
        json.dump(results_spocom, f, indent=2, ensure_ascii=False)
    
    # 原文本抽取结果
    results_ori = two_stage_relation_extraction(ori_text, re_list)
    results_ori = filter_elements(results_ori,ori_text)
    print(results_ori)
    with open(f"../Result/ori_answer/results_{idx}.json", "w", encoding='utf-8') as f:
        json.dump(results_ori, f, indent=2, ensure_ascii=False)
    
    # 压缩文本抽取结果
    results_com = two_stage_relation_extraction(com_text, re_list)
    results_com = filter_elements(results_com,com_text)
    print(results_com)
    with open(f"../Result/com_answer/results_{idx}.json", "w", encoding='utf-8') as f:
        json.dump(results_com, f, indent=2, ensure_ascii=False)
        
    # 句法依存压缩文本抽取结果(优化)
    results_spocom = two_stage_relation_extraction(spo_text, re_list)
    results_spocom = filter_elements(results_spocom, spo_text)
    print(results_spocom)
    with open(f"../Result/dcom_answer_optimization/results_{idx}.json", "w", encoding='utf-8') as f:
        json.dump(results_spocom, f, indent=2, ensure_ascii=False)
'''




