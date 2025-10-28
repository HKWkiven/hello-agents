# -*- coding: utf-8 -*-
# !/usr/bin/env python3

"""
@日期: 2025/10/27 15:46
@作者: HKW
@说明: 
"""
import ast
import re
from llm_client import HelloAgentsLLM
from tools import ToolExecutor, search
from typing import List, Dict, Any, Optional


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


class Planner:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def plan(self, question: str) -> list[str]:
        """
        根据用户问题生成一个行动计划。
        """
        PLANNER_PROMPT_TEMPLATE = """
        你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
        请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
        你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

        问题：{question}

        请严格按照以下格式输出你的计划，```python与```作为前后缀是必要的：
        ```python
        ["步骤1", "步骤2", "步骤3", ...]
        ```
        """
        prompt = PLANNER_PROMPT_TEMPLATE.format(question=question)

        # 为了生成计划，我们构建一个简单的消息列表
        messages = [{"role": "user", "content": prompt}]

        print("--- 正在生成计划 ---")
        response_text = ""
        # 使用流式输出来获取完整的计划
        for chunk in self.llm_client.think(messages=messages):
            response_text += chunk

        print(f"✅ 计划已生成：\n{response_text}")

        # 解析LLM输出的列表字符串
        try:
            # 找到```python和```之间的内容
            plan_str = response_text.split("```python")[1].split("```")[0].strip()
            # 使用ast.literal_eval来安全地执行字符串，将其转换为python列表
            plan = ast.literal_eval(plan_str)
            return plan if isinstance(plan, list) else []
        except (ValueError, SyntaxError, IndexError) as e:
            print(f"❌ 解析计划时出错：{e}")
            print(f"原始响应：{response_text}")
            return []
        except Exception as e:
            print(f"❌ 解析计划时发生未知错误：{e}")
            return []


class Executor:
    def __init__(self, llm_client: HelloAgentsLLM):
        self.llm_client = llm_client

    def execute(self, question: str, plan: list[str]) -> str:
        """
        根据计划，逐步执行并解决问题。
        """
        EXECUTOR_PROMPT_TEMPLATE = """
        你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
        你将收到原始问题、完整的计划、到目前为止已经完成的步骤和结果。
        请你专注于解决“当前步骤”，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

        # 原始问题：
        {question}

        # 完整计划：
        {plan}

        # 历史步骤与结果：
        {history}

        # 当前步骤：
        {current_step}

        请仅输出针对“当前步骤”的回答：
        """
        history = ""  # 用于存储历史步骤和结果的字符串

        print("\n--- 正在执行计划 ---")

        for i, step in enumerate(plan):
            print(f"\n-> 正在执行步骤 {i+1}/{len(plan)}: {step}")

            prompt = EXECUTOR_PROMPT_TEMPLATE.format(
                question=question,
                plan=plan,
                history=history if history else "无",  # 如果是第一步，则历史为空
                current_step=step
            )

            messages = [{"role": "user", "content": prompt}]
            response_text = ""
            for chunk in self.llm_client.think(messages=messages):
                response_text += chunk

            # 更新历史记录，为下一步做准备
            history += f"步骤 {i+1}: {step}\n结果: {response_text}\n\n"

            print(f"✅ 步骤 {i+1} 已完成，结果: {response_text}")

        # 循环结束后，最后一步的响应就是最终答案
        final_answer = response_text
        return final_answer


class PlanAndSolveAgent:
    def __init__(self, llm_client: HelloAgentsLLM):
        """
        初始化智能体，同时创建规划器和执行器实例
        """
        self.llm_client = llm_client
        self.planner = Planner(self.llm_client)
        self.executor = Executor(self.llm_client)

    def run(self, question: str):
        """
        运行智能体的完整流程：先规划，后执行。
        """
        print(f"\n--- 开始处理问题 ---\n问题：{question}")

        # 1. 调用规划器生成计划
        plan = self.planner.plan(question)

        # 检查计划是否成功生成
        if not plan:
            print("\n--- 任务终止 ---\n无法生成有效的行动计划。")
            return

        # 2. 调用执行器执行计划
        final_answer = self.executor.execute(question, plan)

        print(f"\n--- 任务完成 ---\n最终答案：{final_answer}")


class Memory:
    """
    一个简单的短期记忆模块，用于存储智能体的行动与反思轨迹
    """

    def __init__(self):
        """
        初始化一个空列表来存储所有记录。
        """
        self.records: List[Dict[str, Any]] = []

    def add_record(self, record_type: str, content: str):
        """
        向记忆中添加一条新记录
        Args:
            record_type: str, 记录的类型（'execution' 或 'reflection'）
            content: str, 记录的具体内容（例如：生成的代码或反思的反馈）
        """
        record = {"type": record_type, "content": content}
        self.records.append(record)
        print(f"📝 记忆已更新，新增一条 '{record_type}' 记录。")

    def get_trajectory(self) -> str:
        """
        将所有记忆记录格式化为一个连贯的字符串文本，用于构建提示词。
        """
        trajectory_parts = []
        for record in self.records:
            if record.get('type') == 'execution':
                trajectory_parts.append(f"--- 上一轮尝试（代码） ---\n{record.get('content')}")
            elif record.get('type') == 'reflection':
                trajectory_parts.append(f"--- 评审员反馈 ---\n{record.get('content')}")
        return "\n\n".join(trajectory_parts)

    def get_last_execution(self) -> Optional[str]:
        """
        获取最近一次的执行结果（例如：最新生成的代码）。
        如果不存在，则返回 None.
        """
        for record in reversed(self.records):
            if record.get("type") == "execution":
                return record.get("content")
        return None


class ReflectionAgent:
    def __init__(self, llm_client: HelloAgentsLLM, max_iterations=3):
        self.llm_client = llm_client
        self.memory = Memory()
        self.max_iterations = max_iterations

    def _get_llm_response(self, prompt: str) -> str:
        """
        一个辅助方法，用于调用LLM并获取反正的流式响应。
        """
        messages = [{"role": "user", "content": prompt}]
        response_text = ""
        for chunk in self.llm_client.think(messages=messages):
            response_text += chunk
        return response_text

    def run(self, task: str):
        INITIAL_PROMPT_TEMPLATE = """
        你是一位资深的Python程序员。请根据以下要求，编写一个Python函数。
        你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。

        要求：{task}

        直接输出代码，不要包含任何额外的解释。
        """

        REFLECT_PROMPT_TEMPLATE = """
        你是一位极其严格的代码评审专家和资深算法工程师，对代码性能有极致的要求。
        你的任务是审查一下Python代码，并专注于找出其在<strong>算法效率</strong>上的主要瓶颈。

        # 原始任务：
        {task} 

        # 待审查的代码：
        ```python
        {code}
        ```

        请分析该代码的时间复杂度，并思考是否存在一种<strong>算法上更优</strong>的解决方案来显著提升性能。
        如果存在，请清晰地指出当前算法的不足，并提出具体的、可行的改进算法建议（例如：使用筛法替代试除法）。
        如果代码在算法层面已经达到最优，才能回答“无需改进”。

        请直接输出你的反馈，不要包含任何额外的解释以及代码示例。
        """

        REFINE_PROMPT_TEMPLATE = """
        你是一位资深的Python程序员。你正在根据一位代码评审专家的反馈来优化你的代码。

        # 原始任务：
        {task}

        # 你上一轮尝试的代码：
        ```python
        {last_code_attempt}
        ```
        评审员的反馈：
        {feedback}

        请根据评审员的反馈，生成一个优化后的新版本代码。
        你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。
        请直接输出优化后的代码，不要包含任何额外的解释。
        """
        print(f"\n--- 开始处理任务 ---\n任务：{task}")

        # --- 1. 初始执行 ---
        print(f"\n--- 正在进行初始尝试 ---")
        initial_prompt = INITIAL_PROMPT_TEMPLATE.format(task=task)
        initial_code = self._get_llm_response(initial_prompt)
        self.memory.add_record("execution", initial_code)

        # --- 2. 迭代循环：反思与优化 ---
        for i in range(self.max_iterations):
            print(f"\n--- 第 {i+1}/{self.max_iterations} 轮迭代 ---")

            # a. 反思
            print(f"\n-> 正在进行反思...")
            last_code = self.memory.get_last_execution()
            reflect_prompt = REFLECT_PROMPT_TEMPLATE.format(task=task, code=last_code)
            feedback = self._get_llm_response(reflect_prompt)
            self.memory.add_record("reflection", feedback)

            # b. 检查是否需要停止
            if "无需改进" in feedback:
                print("\n ✅ 反思认为的代码已无需改进，任务完成。")
                break

            # c. 优化
            print(f"\n-> 正在进行优化...")
            refine_prompt = REFINE_PROMPT_TEMPLATE.format(
                task=task,
                last_code_attempt=last_code,
                feedback=feedback
            )
            refine_code = self._get_llm_response(refine_prompt)
            self.memory.add_record("execution", refine_code)

        final_code = self.memory.get_last_execution()
        print(f"\n--- 任务完成 ---\n最终生成的代码：\n```python\n{final_code}\n```")
        return final_code

if __name__ == "__main__":
    llm_client = HelloAgentsLLM()
    tool_executor = ToolExecutor()

    # --- ReAct Agent 测试 ---
    # search_description = ("一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，"
    #                       "应使用此工具。入参是一个需要浏览器检索的问题。")
    # tool_executor.registerTool("Search", search_description, search)
    # myAgent = ReActAgent(llm_client, tool_executor, 3)
    # myAgent.run("告诉我华为最新的手机型号和主要卖点")
    # ------

    # --- Planner Agent 测试 ---
    # question = "一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？"
    # myAgent = PlanAndSolveAgent(llm_client)
    # myAgent.run(question)
    # ------

    # --- Reflection Agent 测试 ---
    question = "编写一个Python函数，找出1到n之间所有的素数 (prime numbers)。"
    myAgent = ReflectionAgent(llm_client)
    myAgent.run(question)
    # ------