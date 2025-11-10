# -*- coding: utf-8 -*-
# !/usr/bin/env python3

"""
@日期: 2025/10/30 15:56
@作者: HKW
@说明: 异常体系
"""


class HelloAgentsException(Exception):
    """HelloAgents基础异常类"""
    pass

class LLMException(HelloAgentsException):
    """LLM相关异常"""
    pass

class AgentException(HelloAgentsException):
    """Agent相关异常"""
    pass

class ConfigException(HelloAgentsException):
    """配置相关异常"""
    pass

class ToolException(HelloAgentsException):
    """工具相关异常"""
    pass


if __name__ == "__main__":
    pass
