import json
from getapi import get_chat_res,build_message
from ltp import StnSplit

def get_order(order):
    sys_prompt_spo,sys_prompt_nospo, user_prompt= '','',''
    for i in order['system_prompt_nospo']:
        sys_prompt_nospo += ''.join(i)
    for i in order['system_prompt_spo']:
        sys_prompt_spo += ''.join(i)
    for i in order['user_prompt']:
        user_prompt += ''.join(i)

    return sys_prompt_spo,sys_prompt_nospo, user_prompt

#data = json.load(open("data/ori_text.json", 'r', encoding="utf-8"))#字典列表
#data = json.load(open("data/windpower/ori.json", 'r', encoding="utf-8"))#字典列表
#data = json.load(open("data/compress.json", 'r', encoding="utf-8"))#字典列表
data = json.load(open("santaori.json", 'r', encoding="utf-8"))#字典列表
spo_prompt = json.load(open('data/conf/Syntactic.json', 'r', encoding="utf-8"))#字符串列表
compress_prompt = json.load(open('data/conf/compress_order.json','r',encoding="utf-8"))#字典 userprompt_1,userprompt_2和sysprompt

sys_prompt_spo,sys_prompt_nospo, user_prompt_com = get_order(compress_prompt)

spo_sys,spo_user = "",""
for ele in spo_prompt['system_prompt']:
    spo_sys += ele
for ele in spo_prompt['user_prompt']:
    spo_user += ele

results = []
for i, dic in enumerate(data):
    ori_text = dic['ori_text']  # 原文本
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
        '''
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
        '''
        if "{svo_relationship}" in sys_prompt_spo:
            spo_sys_prompt = sys_prompt_spo.format(svo_relationship=res_spo)
            mess = build_message(spo_sys_prompt, com_prompt)
            res = get_chat_res('gpt', mess)
            com_text_spo += res
            print(com_text_spo)  # 带spo
        #new_dic['com_text_spo'] = com_text_spo.replace(' ', '')
        new_dic['com_text_spo'] = com_text_spo
    if "spo_list" not in dic:
        new_dic['spo_list'] = spo_list
    else:
        new_dic['spo_list'] = dic["spo_list"]
    results.append(new_dic)

    with open(f"change.json", "w", encoding='utf-8') as f:
        f.write(json.dumps(results, indent=2, ensure_ascii=False))
    f.close()


