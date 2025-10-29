# -*- coding: utf-8 -*-
# !/usr/bin/env python3

"""
@日期: 2025/10/27 08:39
@作者: HKW
@说明: 
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    cache_dir= "/00-MyTest/transformers_cache"

    # 指定模型ID
    model_id = "Qwen/Qwen1.5-0.5B-Chat"

    # 设置设备，优先使用GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    print("分词器加载完成！")

    # 加载模型，并将其移动到指定设备
    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir).to(device)

    print("模型加载完成！")

    # 准备对话输入
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你好，请介绍你自己。"}
    ]

    # 使用分词器的模板格式化输入
    text = tokenizer.apply_chat_template(
        messages,
        tokenizer=False,
        add_generation_prompt=True
    )

    # 编码输入文本
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    print(f"编码后输入文本：{model_inputs}")

    # 使用模型生成回答
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )

    # 将生成的 Token ID 截取掉输入部分
    # 这样我们只解码模型新生成的部分
    generated_ids = [
        output_ids[len(input_ids)] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # 解码生成的 Token ID
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print("\n模型的回答:")
    print(response)




















