import json
import os
import sys
from openai import OpenAI
from ltp import StnSplit
import re
from concurrent.futures import ThreadPoolExecutor

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
target_project_dir = os.path.join(parent_dir, 'DSC-EREF')
if target_project_dir not in sys.path:
    sys.path.insert(0, target_project_dir)
from PipeLine.LLMcompress import is_valid_triple,filter_elements

re_list = {
        #数据流
        "输入": [["数据/特征", "模型/算法/方法"]],"输出": [["模型/算法/方法", "数据/结果/诊断结论"]],"监测":[["系统","物理量"]],
        #方法与目标
        "解决": [["方法", "问题"]],"应用于": [["模型/信号/方法","数据/设备/场景/领域"]],
        #因果与关联
        "导致": [["方法/操作（原因）", "现象/故障（结果）"],["故障", "现象/故障（结果）"],["现象（原因）", "现象/故障（结果）"]],
        "伴随": [["故障", "特征信号/现象"]],
        #组成结构
        "包含":[["信号","子信号"],["故障类型","子类"],["系统","子系统"],["工作流程","阶段"],["部件/组件","子部件/子组件"],["方法","子方法"],["数据","物理量"],
                ["网络结构","组件"],["总物理量","子物理量"]],
        #属性与描述
        "数量":[["参数","数值/范围"],["数据","数值/范围"]],
        "型号": [["部件", "型号"]],
        "失效形式": [["部件", "失效模式"]],
        "定义":[["符号/变量/物理量/算法","语义解释"],["评估指标","计算方式或物理含义"]],
        "特点":[["信号/系统/数据/能源","属性/特征"]],
        #研究行为
        "提出/尝试": [["研究人/文献/章节/本文", "操作"],["研究人/文献/章节/本文", "方法/模型/网络"]],"介绍": [["研究者/文献", "原理/综述"]],
        "研究目标/方向":[["研究人/文献/章节/本文","目标"],["技术方法","工程目标"]],
        "建立/构建":[["研究人/文献/章节/本文","模型/组件"]],
        #时间地点
        "采集时间": [["数据", "时间"]],"维修时间": [["部件", "时间"]],"采集地点": [["数据", "地点"]],
        #评价
        "效果": [["方法/技术", "功能/优势/达成效果"],["组件/模型", "功能/优势/达成效果"]],"缺点/不足": [["方法/模型/技术", "问题/局限/缺陷"]]
}
#1、调用gpt
def get_chat_res(mess,temp,top_p,freq_pen,pres_pen,max_tok):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("未找到")
    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model='gpt-4o',
            messages=mess,
            temperature=temp,
            max_tokens=max_tok,
            top_p=top_p,
            frequency_penalty=freq_pen,
            presence_penalty=pres_pen
        )
        # 打印响应中的第一个结果
        res = response.choices[0].message.content
        return res
        # print(res)
    except Exception as e:
        print(f"发生错误: {e}")

def build_message(sys_prompt,user_prompt,assistant=''):
    mess = [{"role": "system", "content": sys_prompt}, ]
    if assistant:
        mess.append({"role": "assistant", "content": assistant})
    mess.append({"role": "user", "content": user_prompt})
    return mess
#2、句法分析
def extract_main_verb_obj(text,temp,top_p,freq_pen,pres_pen,max_tok):
    #1、prompt
    spo_prompt = json.load(open('data/conf/Syntactic.json', 'r', encoding="utf-8"))#字符串列表
    spo_sys,spo_user = "",""
    for ele in spo_prompt['system_prompt']:
        spo_sys += ele
    for ele in spo_prompt['user_prompt']:
        spo_user += ele
    #2、分句抽取
    sents = StnSplit().split(text)
    temp_sent = ""
    spo_list=[]
    for i, sent in enumerate(sents):
        if temp_sent:
            sent = temp_sent + sent
            temp_sent = ""
        if len(sent) < 15:
            temp_sent = sent
            continue
        prompt = spo_user.format(input_text=sent)
        mess = build_message(spo_sys, prompt)
        res_spo = get_chat_res(mess,temp,top_p,freq_pen,pres_pen,max_tok)
        spo_list.append(res_spo)
    print(spo_list)
    return spo_list
#3、文本压缩
def compress_with_syntax(text, syntax,temp,top_p,freq_pen,pres_pen,max_tok):
    #1、prompt
    compress_prompt = json.load(open('data/conf/compress_order.json', 'r', encoding="utf-8"))  # 字典 userprompt_1,userprompt_2和sysprompt
    comp_sys,comp_user = "",""
    for ele in compress_prompt['system_prompt_spo']:
        comp_sys += ele
    for ele in compress_prompt['user_prompt']:
        comp_user += ele
    #2、
    sents = StnSplit().split(text)
    temp_sent = ""
    res = ""
    for i, sent in enumerate(sents):
        if temp_sent:
            sent = temp_sent + sent
            temp_sent = ""
        if len(sent) < 15:
            temp_sent = sent
            continue
        com_prompt = comp_user.format(text_to_compress=sent)
        spo_sys_prompt = comp_sys.format(svo_relationship=syntax)
        mess = build_message(spo_sys_prompt, com_prompt)
        temp_res = get_chat_res(mess,temp,top_p,freq_pen,pres_pen,max_tok)
        res += temp_res
    print(res)
    return res
#4、实体关系抽取
def two_stage_relation_extraction(text, re_list,temp,top_p,freq_pen,pres_pen,max_tok):
    rel_list = []
    for key, value in re_list.items():
        rel_list.append(key)

    prompts = json.load(open("../DSC-EREF/Data/ere_order.json.json", 'r', encoding="utf-8"))
    stage1_prompt_list = prompts["stage1_prompt"]
    stage1_prompt = f'''给定文本为{text}，给定本体关系列表：{rel_list}。\n'''
    for ele in stage1_prompt_list:
        stage1_prompt += ele
    results = []
    mess = build_message('您是一个关系抽取专家，负责从给定的句子中识别存在的关系。。', stage1_prompt)
    res = get_chat_res(mess,temp,top_p,freq_pen,pres_pen,max_tok)
    if isinstance(res, list):
        res = ''.join(res)
    if '"' not in res:
        return results
    matches = re.findall(r'"(.*?)"', res)
    stage1_response = [match for match in matches if match in rel_list]
    stage1_response = list(set(stage1_response))
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
            res = get_chat_res(mess,temp,top_p,freq_pen,pres_pen,max_tok)
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

def extract_ere(idx,text,temp,top_p,freq_pen,pres_pen,max_tok):
    results_spocom = two_stage_relation_extraction(text,re_list,temp,top_p,freq_pen,pres_pen,max_tok)
    results_spocom = filter_elements(results_spocom)
    return results_spocom

def pipe(temp,top_p,freq_pen,pres_pen,max_tok):
    data = json.load(open("data/conf/optimization.json", 'r', encoding="utf-8"))
    for i, ele in enumerate(data):
        result = []
        temp_dic = {}
        idx = ele['idx']
        text = ele['ori_text']

        sys = extract_main_verb_obj(text,temp,top_p,freq_pen,pres_pen,max_tok)
        print(f"原始文本句法提取成功：{sys}")

        comp = compress_with_syntax(text,sys,temp,top_p,freq_pen,pres_pen,max_tok)
        print(f"压缩完成：{comp}")

        comp_sys = extract_main_verb_obj(comp,temp,top_p,freq_pen,pres_pen,max_tok)
        print(f"压缩文本句法提取成功：{comp_sys}")

        ere = extract_ere(idx,comp,temp,top_p,freq_pen,pres_pen,max_tok)
        print(f"三元组抽取完成：{ere}")

        temp_dic['idx'] = idx
        temp_dic['ori_text'] = text
        temp_dic['comp_text'] = comp
        temp_dic['ori_spo_list'] = sys
        temp_dic['comp_spo_list'] = comp_sys
        temp_dic['ERE_list'] = ere
        result.append(temp_dic)
        with open(f"optimization/answer_{idx}.json", "w", encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

#pipe(0.0,1.0,0.0,0.0,300)
