# -*- coding: utf-8 -*-
# !/usr/bin/env python3

"""
@日期: 2025/10/27 15:46
@作者: HKW
@说明: 
"""
import re

from llm_client import HelloAgentsLLM
from tools import ToolExecutor, search

class ReActAgent:
    def __init__(self, llm_client: HelloAgentsLLM, tool_executor: ToolExecutor, max_steps: int = 5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []

    def _parse_output(self, text: str):
        """解析LLM的输出，提取Thought和Action。"""
        thought_match = re.search(r"Thought: (.*)", text)
        action_match = re.search(r"Action: (.*)", text)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):
        """解析Action字符串，提取工具名称和输入。"""
        match = re.match(r"(\w+)\[(.*)]", action_text)
        if match:
            return match.group(1), match.group(2)
        return None, None

    def run(self, question: str):
        """
        运行ReAct智能体来回答一个问题。
        """
        self.history = []  # 每次运行时重置历史记录
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"--- 第 {current_step} 步 ---")

            # 1. 格式化提示词
            tools_desc = self.tool_executor.getAvailableTools()
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc,
                question=question,
                history=history_str
            )

            # 2. 调用LLM进行思考
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages)

            if not response_text:
                print("错误：LLM未能返回有效回应。")
                break

            # 3. 解析LLM的输出
            thought, action = self._parse_output(response_text)

            if thought:
                print(f"🤔 思考：{thought}")

            if not action:
                print("警告：未能解析出有效的Action，流程终止。")
                break

            # 4. 执行Action
            if action.startswith("Finish"):
                # 如果是Finish指令，提取最终答案并结束
                final_answer = re.match(r"Finish\[(.*)]", action).group(1)
                print(f"🎉 最终答案：{final_answer}")
                return final_answer

            tool_name, tool_input = self._parse_action(action)
            if not tool_name or not tool_input:
                # ...处理无效Action格式...
                continue

            print(f"🎬 行动：{tool_name}[{tool_input}]")

            tool_function = self.tool_executor.getTool(tool_name)
            if not tool_function:
                observation = f"错误：未找到名为 {tool_name} 的工具"
            else:
                observation = tool_function(tool_input)  # 调用真实工具

            # 5. 将本轮的Action和Observation添加到历史记录中
            print(f"👀 观察: {observation}")
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")

        # 循环结束
        print("已达到最大步数，流程终止。")
        return None


if __name__ == "__main__":
    # 提示词模板
    REACT_PROMPT_TEMPLATE = """
    请注意，你是一个有能力调用外部工具的智能助手。

    可用工具如下：
    {tools}

    请严格按照以下格式进行回应：

    Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
    Action: 你决定采取的行动，必须是以下格式之一：
    - `tool_name[tool_input]`：调用一个可用工具。
    - `Finish[最终答案]`：当你认为已经获得最终答案时。

    现在，请开始解决以下问题：
    Question: {question}
    History: {history}
    """

    llm_client = HelloAgentsLLM()
    tool_executor = ToolExecutor()
    search_description = ("一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，"
                          "应使用此工具。入参是一个需要浏览器检索的问题。")
    tool_executor.registerTool("Search", search_description, search)

    myAgent = ReActAgent(llm_client, tool_executor, 3)
    myAgent.run("告诉我华为最新的手机型号和主要卖点")