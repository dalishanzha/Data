import json
import os
from openai import OpenAI
from ltp import StnSplit
import re
from concurrent.futures import ThreadPoolExecutor

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
    proxy_url = 'http://127.0.0.1'
    proxy_port = '33210'

    os.environ['http_proxy'] = f'{proxy_url}:{proxy_port}'
    os.environ['https_proxy'] = f'{proxy_url}:{proxy_port}'

    client = OpenAI(api_key="sk-", base_url="")
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
def is_valid_triple(relation, head, tail):
    INVALID_TAIL_FOR_CONTAINS = {
        "输入","框架","数据"
    }

    INVALID_TAIL_FOR_APPLIED = {
        "SCADA数据"
    }
    if relation == "包含" or relation == "监测" or relation == "采集":
        # 检查 tail 是否包含任何非法关键词
        tail_str = tail.lower()
        for word in INVALID_TAIL_FOR_CONTAINS:
            if word == tail_str:
                return False
    if relation == "应用于":
        # 检查 tail 是否包含任何非法关键词
        tail_str = tail.lower()
        for word in INVALID_TAIL_FOR_APPLIED:
            if word == tail_str:
                return False
    return True

def filter_elements(data):
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
        # 遍历现有结果中的每个元素
        for idx, existing in enumerate(result):
            if existing[0] != current_key:
                continue
            exist_a, exist_b = existing[1][0], existing[1][1]

            # 情况1: 当前元素被已有元素覆盖
            if (current_a in exist_a) and (current_b in exist_b):
                is_covered = True
                break

            # 情况2: 当前元素覆盖已有元素
            if (exist_a in current_a) and (exist_b in current_b):
                to_remove.append(idx)

        # 若当前元素未被覆盖，则删除被它覆盖的旧元素并加入结果
        if not is_covered:
            for idx in reversed(to_remove):
                del result[idx]
            result.append(current)
    return result
def two_stage_relation_extraction(text, re_list,temp,top_p,freq_pen,pres_pen,max_tok):
    rel_list = []
    for key, value in re_list.items():
        rel_list.append(key)
    # ========== 第一阶段：关系识别 ==========
    #
    stage1_prompt = f"""
    给定文本为{text}，给定本体关系列表：{rel_list}。
    部分关系定义如下：
    -“包含”关系：表示一个整体实体（如系统、模型、信号、故障类型）在其结构或组成中持有或由若干子部分构成，不适用于描述非组成性元素。
    -“效果”关系用于刻画一种方法、策略、技术或组件在应用中所实现的功能、达成的具体效果、带来的性能优势，或为后续处理所创造的有利条件；包括其设计目的、结构性改变及因果性结果，只要该内容直接反映该方法的核心作用，即视为有效。
    -“定义”关系：表示一个术语、符号、缩写、变量或可视化表达（主体）与其确切含义、概念解释、数学/物理内涵。
    -“数量”关系：表示某一属性、参数、数据、评价指标、数据维度或现象（主体）被设置的具体数值、测量结果或取值范围（客体）。即使数值出现在括号中，或主体带有长定语修饰，只要语义上符合，即应抽取。
    -“导致”关系：表示一个原因性事件、操作、故障或现象（主体）直接引发或造成另一个结果性现象、问题或状态变化（客体），强调单向因果链。
    -“提出/尝试”关系：表示研究者、文献、本文或章节（主体）设计、构建或正式引入某种方法、模型、框架或操作流程（客体）。
    -“研究目标/方向”关系：表示研究者、文献或项目（主体）旨在解决的科学问题、工程挑战或希望达成的具体研究目的（客体）。
    -“介绍”关系：表示研究者、文献或综述性内容（主体）对某一原理、背景知识、已有技术综述或统计结果（客体）进行说明或概述，不涉及新方法提出。
    发挥你的语义理解能力，识别并返回存在的关系名称列表，注意只要语义上符合，即应抽取，不要求关系名称必须出现在文本中。最后，不要包含解释内容。
    示例响应格式：
    ["创始人", "创立时间", "总部地点"]
    """
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

    # ========== 第二阶段：实体对抽取 ==========
    if not stage1_response:
        return results
    def extract_entities(relation):
        temp_result = [relation]
        for value in re_list[relation]:
            entity_1, entity_2 = value
            #print(f"e1{entity_1},e2{entity_2}")
            stage2_prompt = f"""
            给定head实体类型'{entity_1}'、tail实体类型'{entity_2}'且之间的关系为'{relation}'，请从文本中提取符合条件的实体对。
            要求：
            务必清晰提取实体的起止位置，确保实体类型与要求一致，同时完整保留实体的原始表述,不得修改或简化实体的原始表达
            如果没有找到符合条件的实体对，请返回空列表 []，不要包含其他解释内容。
            注意：
            如果存在复合信息，将其分解为单个实体及其对应的属性，确保每个实体的属性独立列出
            如果关系为因果类型，并按照因果链逐步推理，确保不遗漏任何中间环节。
            如果存在嵌套包含的情况，递进抽取对应实体，不要将子实体与更高层级的概念（如整体系统、方法、数据或其他模块）直接关联。
            若参数或数据实体存在从属信息，保留该实体的从属表达。
            "缺点/不足"关系的客体必须描述方法的负面表现，不能是外部条件。
            “提出/尝试”关系的客体必须属于研究者、文献、本文或章节类型。
            按照示例格式返回结果，确保结构清晰、信息完整。
            文本内容：
            {text}
            示例响应格式：
            ['马云','阿里巴巴'],['马化腾','腾讯']
            """
            #务必清晰定义起始原因和最终结果
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
