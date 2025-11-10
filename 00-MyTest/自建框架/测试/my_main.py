# -*- coding: utf-8 -*-
# !/usr/bin/env python3

"""
@日期: 2025/10/29 10:49
@作者: HKW
@说明: 
"""

from dotenv import load_dotenv
from my_hello_agents.agents.simple_agent import SimpleAgent
from my_hello_agents.core.llm import HelloAgentsLLM

# 加载环境变量
load_dotenv()


if __name__ == "__main__":
    llm = HelloAgentsLLM()
    agent = SimpleAgent("测试对话智能体", llm)
    question = "请介绍一下你自己"
    for chunk in agent.stream_run(question):
        print(chunk)

