# -*- coding: utf-8 -*-
# !/usr/bin/env python3

"""
@日期: 2025/10/29 16:16
@作者: HKW
@说明: 配置管理
"""

import os
from typing import Optional, Dict, Any
from pydantic import BaseModel


class Config(BaseModel):
    """HelloAgents配置类"""

    # LLM配置
    default_model: str = "qwen2.5:7b"
    default_provider: str = "ollama"
    temperature: float = 0.7
    max_token: Optional[int] = None

    # 系统配置
    debug: bool = False
    log_level: str = "INFO"

    # 其他配置
    max_history_length: int = 100

    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量创建配置"""
        return cls(
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_token=int(os.getenv("MAX_TOKENS")) if os.getenv("MAX_TOKENS") else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.model_dump()


if __name__ == "__main__":
    pass
