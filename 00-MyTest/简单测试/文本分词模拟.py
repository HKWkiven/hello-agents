# -*- coding: utf-8 -*-
# !/usr/bin/env python3

"""
@日期: 2025/10/24 08:45
@作者: HKW
@说明: 
"""


import re, collections


def get_stats(vocab):
    """统计词元对频率"""
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs


def merge_vocab(pair, v_in):
    """合并次元的对"""
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out


if __name__ == "__main__":
    # 准备语料库，每个词末尾加上</w>表示结束，并切分好字符
    vocab = {'h u g </w>': 1, 'p u g </w>': 1, 'p u n </w>': 1, 'b u n </w>': 1}
    num_merges = 10  # 设置合并次数

    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        print(f"第{i + 1}次合并: {best} -> {''.join(best)}")
        print(f"新词表（部分）: {list(vocab.keys())}")
        print("-" * 20)
