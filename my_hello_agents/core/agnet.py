# -*- coding: utf-8 -*-
# !/usr/bin/env python3

"""
@日期: 2025/10/30 15:33
@作者: HKW
@说明: Agent基类
"""

from abc import ABC, abstractmethod
from typing import Optional
from my_hello_agents.core.message import Message
from my_hello_agents.core.llm import HelloAgentsLLM
from my_hello_agents.core.config import Config


class Agent(ABC):
    """Agent基类"""

    def __init__(
            self,
            name: str,
            llm: HelloAgentsLLM,
            system_prompt: Optional[str] = None,
            config: Optional[Config] = None
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()
        self._history:list[Message] = []

    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """运行Agent"""
        pass

    def add_message(self, message: Message):
        """添加消息到历史记录"""
        self._history.append(message)

    def get_history(self) -> list[Message]:
        """获取历史记录"""
        return self._history.copy()

    def __str__(self) -> str:
        return f"Agent(name={self.name}, provider={self.llm.provider})"

    def __repr__(self) -> str:
        return self.__str__()


if __name__ == "__main__":
    pass
