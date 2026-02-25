from getapi import get_chat_res, build_message
import json
import ast
import re
from ltp import StnSplit
from concurrent.futures import ThreadPoolExecutor

def getexample(entity_pairs):
    example = ""
    for key, value in entity_pairs.items():
        stage1_prompt = f'''生成1对实体类型属于{value}实体对的实体对，严格按照['entity1','entity2']格式返回。'''
        stage1_mess = build_message('你是一个可靠的助手', stage1_prompt)
        stage1_res = get_chat_res('gpt', stage1_mess)
        print(stage1_res, type(stage1_res))
        matches = re.findall(r"'(.*?)'", stage1_res)
        results = [matches[i:i + 2] for i in range(0, len(matches), 2)]
        for i in results:
            stage2_prompt = f"生成一句关系为{key}实体对为{i}的句子。"
            stage2_mess = build_message('你是一个可靠的助手', stage2_prompt)
            stage2_res = get_chat_res('gpt', stage2_mess)
            results = f"关系为{key}，实体对为{i}的句子示例：" + stage2_res
            example += results
    return example

def is_valid_triple(relation, head, tail):
    """
    判断三元组 (relation, head, tail) 是否符合语义规则
    """
    # 定义非法 tail 词（针对"包含"关系）
    '''
     "输出", "数据", "特征", "信号", "参数", "结果", 
        "值", "信息", "样本", "向量", "矩阵", "时间序列", "测量"
    '''
    INVALID_TAIL_FOR_CONTAINS = {
        "输入","框架","数据"
    }

    INVALID_TAIL_FOR_APPLIED = {
        "SCADA数据"
    }

    INVALID_TAIL_FOR_PROPOSE = {
        "用","提出","采用"
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

    if relation == "提出/尝试":
        # 检查 tail 是否包含任何非法关键词
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

            # 情况3：当前元素不属于原文的一部分
            #if exist_a not in text:
            #    to_remove.append(idx)

        # 若当前元素未被覆盖，则删除被它覆盖的旧元素并加入结果
        if not is_covered:
            for idx in reversed(to_remove):
                del result[idx]
            result.append(current)
    return result

def two_stage_relation_extraction(text, spo_list, re_list):
    #example = getexample(re_list)
    #print(example)
    #学习示例{example}，
    # re_list ={'A':['B','C']}
    rel_list = []
    for key, value in re_list.items():
        rel_list.append(key)
    # ========== 第一阶段：关系识别 ==========
    #
    stage1_prompt = f"""
    给定文本为{text}，给定本体关系列表：{rel_list}。
    部分关系定义如下：
    -“包含”关系：表示一个整体实体（如系统、模型、信号、故障类型、状态、方法）在其结构或组成中持有或由若干子部分构成，不适用于描述非组成性元素以及“X是Y”或类似结构的定义句。
    -“效果”关系用于刻画一种方法、策略、函数、技术或组件在应用中所实现的优秀功能、达成的具体成效、带来的性能优势，或为后续处理所创造的有利条件；包括其设计目的、结构性改变及因果性结果，只要该内容直接反映该方法的核心作用，即视为有效。
    -“定义”关系：表示一个术语、符号、缩写、变量或可视化表达（主体）与其确切含义、概念解释、数学/物理内涵。如“设X为Y”、“X表示Y”、“X是Y”、“X(Y)”或类似结构的定义句。
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
    res = get_chat_res('gpt', mess)
    if isinstance(res, list):
        res = ''.join(res)
    if '"' not in res:
        return results
    matches = re.findall(r'"(.*?)"', res)
    stage1_response = [match for match in matches if match in rel_list]
    print(stage1_response)

    # ========== 第二阶段：实体对抽取 ==========
    if not stage1_response:
        return results
    def extract_entities(relation):
        temp_result = [relation]
        for value in re_list[relation]:
            entity_1, entity_2 = value
            '''实体顺序和类型必须与指定的{entity_1}和 {entity_2}完全匹配。
            提取的实体对之间的关系必须符合 {relation} 的定义。'''
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
            "特点"关系应在有明确指示，如“特性”、“特点”时才进行抽取。
            "研究目标/方向"关系应在有明确指示，如“目标”或类似含义时才进行抽取。
            "缺点/不足"关系的客体必须描述方法的负面表现，不能是外部条件。
            “提出/尝试”关系的主体必须属于研究者、文献、本文或章节类型。
            “包含”关系的客体不能是主体的非组成性元素。
            "数量"关系的客体必须存在具体数值。
            "定义"关系如“设符号X为Y”、“设Y为符号X”、“X表示Y”、“X是Y”、“X(Y)”或类似结构的定义句，应抽取['X','Y']
            “”
            按照示例格式返回结果，确保结构清晰、信息完整。
            文本内容：
            {text}
            示例响应格式：
            ['马云','阿里巴巴'],['马化腾','腾讯']
            """
            #务必清晰定义起始原因和最终结果
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
#"分析": [["方法", "故障/状态/性能指标"]] "分析": [["方法/模型", "物理量"]]
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
data = json.load(open("../data/temp.json", 'r', encoding="utf-8"))  # 字典列表
for i, elem in enumerate(data):  # elem是字典
    ori_text = elem['ori_text']
    spo_list = elem['spo_list_gpt']
    com_text = elem['com_text_nospo']
    spo_text = elem['com_text_spo_gpt']
    idx = elem['idx']
    spo_text_santa = elem['com_text_spo_santa']

    # 句法依存压缩文本抽取结果(优化)
    results_spocom = two_stage_relation_extraction(spo_text, spo_list, re_list)
    results_spocom = filter_elements(results_spocom, spo_text)
    print(results_spocom)
    with open(f"../dcom_answer_optimization/results_{idx}.json", "w", encoding='utf-8') as f:
        json.dump(results_spocom, f, indent=2, ensure_ascii=False)

'''
    # 句法依存压缩文本抽取结果
    results_spocom = two_stage_relation_extraction(spo_text, spo_list, re_list)
    results_spocom = filter_elements(results_spocom,spo_text)
    print(results_spocom)
    with open(f"../dcom_answer/results_{idx}.json", "w", encoding='utf-8') as f:
        json.dump(results_spocom, f, indent=2, ensure_ascii=False)
    
    # 原文本抽取结果
    results_ori = two_stage_relation_extraction(ori_text, spo_list, re_list)
    results_ori = filter_elements(results_ori,ori_text)
    print(results_ori)
    with open(f"../ori_answer/results_{idx}.json", "w", encoding='utf-8') as f:
        json.dump(results_ori, f, indent=2, ensure_ascii=False)
    
    # 压缩文本抽取结果
    results_com = two_stage_relation_extraction(com_text, spo_list, re_list)
    results_com = filter_elements(results_com,com_text)
    print(results_com)
    with open(f"../com_answer/results_{idx}.json", "w", encoding='utf-8') as f:
        json.dump(results_com, f, indent=2, ensure_ascii=False)
    
    # santa句法压缩文本抽取结果
    results_santa = two_stage_relation_extraction(spo_text_santa, spo_list, re_list)
    results_santa = filter_elements(results_santa,spo_text_santa)
    print(results_santa)
    with open(f"../santa_answer/results_{idx}.json", "w", encoding='utf-8') as f:
        json.dump(results_santa, f, indent=2, ensure_ascii=False)
        
    # 句法依存压缩文本抽取结果(优化)
    results_spocom = two_stage_relation_extraction(spo_text, spo_list, re_list)
    results_spocom = filter_elements(results_spocom, spo_text)
    print(results_spocom)
    with open(f"../dcom_answer_optimization/results_{idx}.json", "w", encoding='utf-8') as f:
        json.dump(results_spocom, f, indent=2, ensure_ascii=False)
'''




