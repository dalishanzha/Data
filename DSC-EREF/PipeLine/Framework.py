import os
import sys
import json
import ast
import re
import time
from getapi import build_message,get_chat_res
from concurrent.futures import ThreadPoolExecutor
from com_uncom_re import is_valid_triple,filter_elements,two_stage_relation_extraction
from LLMcompress import get_order
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

def main():
        print("文本压缩与句法分析...")
        data = json.load(open("../Data/ori.json", 'r', encoding="utf-8"))  #字典列表
        spo_prompt = json.load(open('../Data/Syntactic.json', 'r', encoding="utf-8"))  #字符串列表
        compress_prompt = json.load(
                open('../Data/compress_order.json', 'r', encoding="utf-8"))  #字典userprompt_1,userprompt_2和sysprompt
        sys_prompt_spo, sys_prompt_nospo, user_prompt_com = get_order(compress_prompt)
        spo_sys, spo_user = "", ""
        for ele in spo_prompt['system_prompt']:
                spo_sys += ele
        for ele in spo_prompt['user_prompt']:
                spo_user += ele

        results = []
        for i, dic in enumerate(data):
                ori_text = dic['ori_text']
                new_dic = {}
                new_dic['idx'] = dic['idx']
                new_dic['ori_text'] = ori_text
                # 1、分句
                sents = StnSplit().split(ori_text)
                spo_list = []
                com_text_spo = ""
                com_text_nospo = ""
                temp_sent = ""
                for i, sent in enumerate(sents):
                        if temp_sent:
                                sent = temp_sent + sent
                                temp_sent = ""
                        if len(sent) < 15:
                                temp_sent = sent
                                continue
                        if "{input_text}" in spo_user and "spo_list" not in dic:
                                prompt = spo_user.format(input_text=sent)
                                print(prompt)
                                mess = build_message(spo_sys, prompt)
                                res_spo = get_chat_res('gpt', mess)  ##
                                spo_list.append(res_spo)
                        if "spo_list" in dic:
                                res_spo = dic["spo_list"]
                        com_prompt = user_prompt_com.format(text_to_compress=sent)
                        print(com_prompt, spo_list)
                        # 2、压缩文本
                        if "{svo_relationship}" not in sys_prompt_nospo:
                                mess = build_message(sys_prompt_nospo, com_prompt)
                                res = get_chat_res('gpt', mess)
                                com_text_nospo += res  # 不带spo
                                print(com_text_nospo)

                        if "{svo_relationship}" in sys_prompt_spo:
                                spo_sys_prompt = sys_prompt_spo.format(svo_relationship=res_spo)
                                mess = build_message(spo_sys_prompt, com_prompt)
                                res = get_chat_res('gpt', mess)
                                com_text_spo += res
                                print(com_text_spo)  # 带spo
                        new_dic['com_text_nospo'] = com_text_nospo
                        new_dic['com_text_spo'] = com_text_spo
                if "spo_list" not in dic:
                        new_dic['spo_list'] = spo_list
                else:
                        new_dic['spo_list'] = dic["spo_list"]
                results.append(new_dic)

                with open(f"../Result/compress_text.json", "w", encoding='utf-8') as f:
                        f.write(json.dumps(results, indent=2, ensure_ascii=False))
                f.close()

                time.sleep(2)
        print("实体关系提取...")
        data_extract = json.load(open("../Result/compress_text.json", 'r', encoding="utf-8"))
        for i, elem in enumerate(data_extract):
                ori_text = elem['ori_text']
                com_text = elem['com_text_nospo']
                spo_text = elem['com_text_spo_gpt']
                idx = elem['idx']

                # 压缩文本抽取结果
                results_com = two_stage_relation_extraction(com_text, re_list)
                results_com = filter_elements(results_com, com_text)
                print(results_com)
                with open(f"../Result/com_answer/results_{idx}.json", "w", encoding='utf-8') as f:
                        json.dump(results_com, f, indent=2, ensure_ascii=False)
                # 原文本抽取结果
                results_ori = two_stage_relation_extraction(ori_text, re_list)
                results_ori = filter_elements(results_ori, ori_text)
                print(results_ori)
                with open(f"../Result/ori_answer/results_{idx}.json", "w", encoding='utf-8') as f:
                        json.dump(results_ori, f, indent=2, ensure_ascii=False)
                # 原文本抽取结果
                results_ori_op = two_stage_relation_extraction(ori_text, re_list)
                results_ori_op = filter_elements(results_ori_op, ori_text)
                print(results_ori_op)
                with open(f"../Result/ori_answer_optimization/results_{idx}.json", "w", encoding='utf-8') as f:
                        json.dump(results_ori_op, f, indent=2, ensure_ascii=False)
                # 句法依存压缩文本抽取结果
                results_spocom = two_stage_relation_extraction(spo_text, re_list)
                results_spocom = filter_elements(results_spocom, spo_text)
                print(results_spocom)
                with open(f"../Result/dcom_answer/results_{idx}.json", "w", encoding='utf-8') as f:
                        json.dump(results_spocom, f, indent=2, ensure_ascii=False)
                # 句法依存压缩文本抽取结果(优化)
                results_spocom_op = two_stage_relation_extraction(spo_text, re_list)
                results_spocom_op = filter_elements(results_spocom_op, spo_text)
                print(results_spocom_op)
                with open(f"../Result/dcom_answer_optimization/results_{idx}.json", "w", encoding='utf-8') as f:
                        json.dump(results_spocom_op, f, indent=2, ensure_ascii=False)
