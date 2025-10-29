# -*- coding: utf-8 -*-
# !/usr/bin/env python3

"""
@日期: 2025/10/29 10:49
@作者: HKW
@说明: 
"""

from dotenv import load_dotenv
from my_llm import MyLLM

# 加载环境变量
load_dotenv()


if __name__ == "__main__":
    # 实例化我们重写的客户端，并指定provider
    llm = MyLLM(model="Qwen/Qwen3-VL-8B-Instruct",provider="modelscope")

    # 准备消息
    messages = [{"role": "user", "content": "你好，请介绍一下你自己。比如你的参数量"}]

    # 发起调用，think等方法都已从父类继承，无需重写
    response_stream = llm.think(messages)

    # 打印响应
    print("ModelScope Response:")
    for chunk in response_stream:
        # print(chunk, end="", flush=True)
        pass

