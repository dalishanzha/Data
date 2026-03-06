import json
import os
from openai import OpenAI
#import ollama
import requests
from zhipuai import ZhipuAI

#from transformers import AutoTokenizer
#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

def get_chat_res(my_model,mess):
    if my_model == 'gpt':
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("未找到")
        client = OpenAI(api_key=api_key)

        try:
            response = client.chat.completions.create(
                model='gpt-4o',
                messages=mess,
                #temperature=0.0
                temperature=0.8,
                top_p=0.2,
                frequency_penalty=-0.32,
                presence_penalty=1.50,
                max_tokens=202
            )
            res = response.choices[0].message.content
            return res
        except Exception as e:
            print(f"发生错误: {e}")
    elif my_model == 'lla':
        try:
            url = 'http://localhost:11434/api/chat'
            data = {
                "model" : "deepseek-r1:1.5b",
                "messages":mess,
                "stream": False
            }
            response = requests.post(url, data=json.dumps(data), headers={'Content-Type': 'application/json'})
            result = response.json()
            res = result.get("message", {}).get("content", "")
            return res
        except Exception as e:
            print(f"发生错误: {e}")
    elif my_model == 'zhipuai':
        client = ZhipuAI(api_key="8436efbfa15748dcb0c34e485616ddc1.R4YeOBeeAAyhecYl")
        try:
            response = client.chat.completions.create(
                model="glm-4-plus",  # 填写需要调用的模型编码
                messages=mess,
                temperature=0.0,
                seed=0
            )
            res = response.choices[0].message.content
            return res
        except Exception as e:
            print(f"发生错误: {e}")

def build_message(sys_prompt,user_prompt,assistant=''):
    mess = [{"role": "system", "content": sys_prompt}, ]
    if assistant:
        mess.append({"role": "assistant", "content": assistant})
    mess.append({"role": "user", "content": user_prompt})
    return mess



