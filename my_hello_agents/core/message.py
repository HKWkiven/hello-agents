# -*- coding: utf-8 -*-
# !/usr/bin/env python3

"""
@日期: 2025/10/29 15:51
@作者: HKW
@说明: 系统消息
"""

from typing import Optional, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel


MessageRole = Literal["user", "assistant", "system", "tool"]


class Message(BaseModel):
    """消息类"""

    content: str
    role: MessageRole
    timestamp: datetime = None
    metadata: Optional[Dict[str, Any]] = None

    def __init__(self, role: MessageRole, content: str, **kwargs):
        super().__init__(
            role=role,
            content=content,
            timestamp=kwargs.get("timestamp", datetime.now()),
            metadata=kwargs.get("metadata", {})
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（OpenAI API格式）"""
        return {
            "role": self.role,
            "content": self.content
        }

    def __str__(self) -> str:
        return f"[{self.role}] {self.content}"


if __name__ == "__main__":
    msg = Message(role="user", content="测试")
    print(msg)
    print(msg.to_dict())
    pass
