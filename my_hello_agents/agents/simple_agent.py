# -*- coding: utf-8 -*-
# !/usr/bin/env python3

"""
@日期: 2025/10/30 16:05
@作者: HKW
@说明: 简单Agent实现 - 基于OpenAI原生API
"""

from typing import Optional, Iterator
from my_hello_agents.core.agnet import Agent
from my_hello_agents.core.llm import HelloAgentsLLM
from my_hello_agents.core.config import Config
from my_hello_agents.core.message import Message


class SimpleAgent(Agent):
    """简单的对话Agent"""
    
    def __init__(
            self, 
            name: str,
            llm: HelloAgentsLLM,
            system_prompt: Optional[str] = None,
            config: Optional[Config] = None
    ):
        super().__init__(name, llm, system_prompt, config)

    def run(self, input_text: str, **kwargs) -> str:
        """
        运行简单Agent
        Args:
            input_text: 用户输入
            **kwargs: 其他参数

        Returns: Agent的响应
        """
        # 构建消息列表
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # 添加历史消息
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        # 添加用户当前消息
        messages.append({"role": "user", "content": input_text})

        # 调用LLM
        response = self.llm.invoke(messages, **kwargs)

        # 保存到历史记录
        self.add_message(Message( "user", input_text,))
        self.add_message(Message("assistant", response))

        return response

    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        """
        流式运行Agent
        Args:
            input_text: 用户输入
            **kwargs: 其他参数

        Yields: Agent响应片段
        """
        # 构建消息列表
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # 添加历史消息
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        # 添加用户当前消息
        messages.append({"role": "user", "content": input_text})

        # 流式调用LLM
        full_response = ""
        for chunk in self.llm.stream_invoke(messages, **kwargs):
            full_response += chunk
            yield chunk

        # 保存完整对话到历史记录
        self.add_message(Message("user", input_text))
        self.add_message(Message("assistant", full_response))


if __name__ == "__main__":
    llm = HelloAgentsLLM()
    agent = SimpleAgent("对话智能体",llm)
    for chunk in agent.stream_run("介绍一下你自己"):
        print(chunk, end="", flush=True)
