# -*- coding: utf-8 -*-
# !/usr/bin/env python3

"""
@日期: 2025/10/29 09:58
@作者: HKW
@说明: 
"""

from hello_agents import SimpleAgent, HelloAgentsLLM
from hello_agents.tools import CalculatorTool
from dotenv import load_dotenv


# 加载环境变量
load_dotenv()


if __name__ == "__main__":
    # 创建LLM实例 - 框架自动检测provider
    llm = HelloAgentsLLM()

    # 创建SimpleAgent
    agent = SimpleAgent(
        name="AI助手",
        llm=llm,
        system_prompt="你是一个幽默的AI助手"
    )

    # 基础对话
    response = agent.run("你好！请介绍一下自己")
    print(response)

    # 添加工具
    calculator = CalculatorTool()
    agent.add_tool(calculator)

    # 现在可以使用工具了
    response = agent.run("请帮我计算 2 + 3 * 4")
    print(response)

    # 查看对话历史
    print(f"历史消息数: {len(agent.get_history())}")
