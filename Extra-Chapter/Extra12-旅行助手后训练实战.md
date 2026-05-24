# 旅行助手后训练实战：从产品协议到 SFT、DPO 与 Rerank 收尾

项目来自我维护的 `helloagents-trip-planner`。它不是论文项目，也不是为了刷一个标准榜单。更像是一次完整的工程实验：把一个看起来能聊天的旅行助手，慢慢改成一个能被前后端接住、能被规则评测解释、也能继续迭代的 Planner。

旅行规划这个场景很容易让人误判。第一版 Demo 通常很好看：用户说“我想去杭州玩 4 天，预算 3500”，模型很快就能写出景点、酒店、餐厅和注意事项。但接到真实前后端以后，问题会变得很具体：预算到底是整趟还是人均，酒店应该按几晚算，景点门票要不要乘同行人数，餐厅是不是工具候选里真的有，最后一天还要不要安排晚餐。

这篇 Extra-Chapter 写的是这条后训练线的精简复盘。完整教程里有更多命令、配置和归档路径，这里重点讲主线和取舍。

项目与配套材料：

- 旅行助手项目仓库：[helloagents-trip-planner](https://github.com/nameless0120/helloagents-trip-planner)
- 完整后训练教程：[旅行助手后训练实战教程](https://github.com/nameless0120/helloagents-trip-planner/blob/main/training/docs/%E6%95%99%E7%A8%8B/%E6%97%85%E8%A1%8C%E5%8A%A9%E6%89%8B%E5%90%8E%E8%AE%AD%E7%BB%83%E5%AE%9E%E6%88%98%E6%95%99%E7%A8%8B.md)
- 配套数据：`helloagents-后训练数据`，网盘链接：<https://pan.baidu.com/s/5oNsK7pwQnqzQEUg5ykb09Q>

一句话概括这条路线：**Prompt 固定协议，SFT 学会结构，DPO 学偏好，Rerank 在候选里选更稳的答案。**

![后训练主线总览图](./images/Extra12-figures/01-后训练主线总览图.png)

---

## 目录

- [第一章：先看一条前后对比](#第一章先看一条前后对比)
- [第二章：为什么不能一上来就训练](#第二章为什么不能一上来就训练)
- [第三章：先改产品协议](#第三章先改产品协议)
- [第四章：冻结评测集](#第四章冻结评测集)
- [第五章：Prompt 调试是在找边界](#第五章prompt-调试是在找边界)
- [第六章：SFT 数据生成和审计](#第六章sft-数据生成和审计)
- [第七章：LoRA SFT 多阶段训练](#第七章lora-sft-多阶段训练)
- [第八章：Best-of-N Replay 和 SFT Rerank](#第八章best-of-n-replay-和-sft-rerank)
- [第九章：DPO 学偏好，核心指标换成 PlannerSoft](#第九章dpo-学偏好核心指标换成-plannersoft)
- [第十章：最终多候选 Rerank](#第十章最终多候选-rerank)
- [第十一章：和 MiMo 外部参考怎么比](#第十一章和-mimo-外部参考怎么比)
- [第十二章：这次实验留下来的经验](#第十二章这次实验留下来的经验)
- [复现资源](#复现资源)

---

## 第一章：先看一条前后对比

先不讲 LoRA，不讲 DPO，也不讲 rerank。看一个普通请求：

> 一个人去杭州玩 4 天，打车，住经济型酒店，喜欢美食和城市地标，总预算 3500 元左右，而且不能超。

基础模型不是完全不会写。它能写出西湖、灵隐寺、城市阳台，也会给酒店和餐厅。但细看会发现不踏实：行程偏空，酒店天数不稳，餐厅重复，预算也离用户目标太远。它看起来像旅行计划，真的拿给用户就有点悬。

后训练后的版本也不是满分，比如餐饮预算还有一个 60 元的小账误差。但它至少开始像一个认真排过的行程：酒店晚数稳定，景点密度正常，餐厅不再一路重复，预算也更接近用户给的硬约束。

| 观察点 | 基础模型 | 后训练后 |
| --- | --- | --- |
| 行程密度 | 4 天基本每天 1 个景点，偏空 | 每天 2 到 3 个景点，覆盖西湖、断桥、雷峰塔、灵隐寺、西溪、清河坊等点位 |
| 酒店 | 前 3 天有酒店，第 4 天写成“无住宿”，预算里又按 4 晚算 | 前 3 晚稳定使用同一家经济型酒店，预算按 3 晚算 |
| 餐饮 | 肯德基重复较多，午晚餐轮换差 | 餐厅有轮换，包含杭帮菜、海鲜、烤肉、面馆和少量快餐 |
| 预算 | 报 1840 元，离 3500 元 hard budget 太远 | 报 2500 元，落在可接受区间下沿 |
| 规则评测 | 抓出 9 类错误 | 当前主要剩餐饮小账误差 |

![开篇 Case 前后对比图](./images/Extra12-figures/02-开篇Case前后对比图.png)

这个例子想说明一件事：后训练不是为了让模型把文案写得更漂亮，而是让它更接近一个能被产品接住的 Planner。住宿晚数、餐饮 grounding、预算关系、日期天气、输出 JSON，这些东西看起来琐碎，但真实产品里就是这些琐碎问题最容易把体验打穿。

---

## 第二章：为什么不能一上来就训练

我一开始也很想直接训练。跑 LoRA 最有进度感：数据一准备，脚本一启动，loss 开始往下掉，看起来项目就在向前走。

后来发现这个顺序是反的。

如果业务事实没有固定，训练只会把混乱学得更稳定。用户说“预算 3000”，模型要知道这是整趟预算还是人均预算；酒店价格是单间每晚，不是全程总价；景点门票要乘同行人数；餐厅不能凭空编，最好来自工具候选。只靠 prompt 反复提醒，能救一部分，但救不了整条链路。

所以这条后训练主线不是：

```text
写 prompt -> 造数据 -> 训练 -> 看指标
```

而是：

```text
前后端协议改造
  -> 冻结 standard / hard 评测集
  -> prompt 调试和失败画像
  -> 强模型生成 SFT 数据
  -> 数据审计与 LLaMA-Factory 导出
  -> LoRA SFT 多阶段训练
  -> Best-of-N Replay
  -> DPO 偏好训练
  -> 规则评测、切片对比和 checkpoint 选择
  -> 多候选 Rerank 收尾
```

这条路更慢，但每一步都能回答两个问题：为什么变好，为什么变坏。

---

## 第三章：先改产品协议

刚开始做旅行助手时，很容易把问题都丢给模型：让模型从自然语言里猜人数，猜预算口径，猜住宿晚数，再猜景点门票和餐厅价格。第一版能跑，但后训练会很痛苦，因为训练数据里的“事实”本身就是飘的。

后来我做的第一个决定很朴素：**不要让模型猜业务事实。**

前端不再只提交一段自由文本，而是显式提交：

- `party`：成人、儿童、老人、总人数、出行类型；
- `budget_constraint`：金额、币种、预算范围、预算档位、约束强度；
- `travel_days`、交通方式、住宿偏好、兴趣偏好等结构化字段。

后端也不再把工具结果一股脑塞进 prompt，而是先编译成 `PlannerContext`。这个上下文会明确告诉模型：这次是几个人，预算是整趟还是人均，酒店按几晚计算，每个景点、餐厅、酒店候选来自哪里，价格 hint 和预算策略是什么，最后输出必须满足什么 JSON shape。

![产品协议改造图](./images/Extra12-figures/03-产品协议改造图.png)

读代码可以从这些文件看起：

| 位置 | 看什么 |
| --- | --- |
| `frontend/src/types/index.ts` | 前端 `TripFormData`、`PartyInfo`、`BudgetConstraint` 类型 |
| `frontend/src/views/Home.vue` | 同行人数、预算档位、总预算、自由文本怎么收集 |
| `backend/app/models/schemas.py` | 后端 `TripRequest`、`PartyInfo`、`BudgetConstraint`、`TripPlan` schema |
| `backend/app/planner/policy.py` | 把请求编译成预算、住宿、人数和价格策略 |
| `backend/app/planner/context.py` | 并行收集景点、天气、酒店等工具快照 |
| `backend/app/planner/compact.py` | 把完整上下文裁剪成模型真正看到的输入 |
| `backend/app/planner/output.py` | 提取顶层 `TripPlan JSON`，并做 shape validation |

有了这层协议，后训练的任务才变窄：模型不再凭感觉写旅行计划，而是在结构化候选里做选择，并输出合法 JSON。

---

## 第四章：冻结评测集

很多训练失败不是模型没变好，而是每次评测的题目都变了。

旅行助手尤其容易这样：今天地图候选变了，明天天气变了，后天预算生成逻辑又变了。最后你分不清是模型变强，还是考卷变简单。

所以第二步不是训练，而是固定评测集。

![评测集冻结图](./images/Extra12-figures/04-评测集冻结图.png)

我把评测拆成两类：

| 评测集 | 作用 |
| --- | --- |
| `standard eval` | 看普通请求下的稳定性，比如常规城市、常规预算、常规偏好 |
| `hard eval` | 主动放大难点，比如多人、老人儿童、严格预算、负向偏好、特殊饮食 |

这里还有一个细节：后面检索策略和上下文修复变了，确实需要重建评测上下文，但不应该重新采样用户请求。我的做法是保持 request signature 不变，只重建工具候选和上下文。这样能保留可比性，又能修掉旧上下文里的脏数据。

这套 frozen eval 后来被反复用于模型选择。严格论文口径下，它更像 validation set，不是 blind test。所以我在文中把它说成“固定评测集上的阶段评估”，不把它包装成独立盲测。

最后做 DPO 收尾数据时，我专门检查过签名重叠：`selected_eval_signature_overlap = 0`。也就是说，评测 prompt 没有进入训练数据。

---

## 第五章：Prompt 调试是在找边界

前后端协议和评测集稳定后，才进入 prompt 调试。

这里的目标不是写一条“神 prompt”。我更关心的是：哪些问题 prompt 能解决，哪些问题必须交给数据、规则和工程。

前几轮 prompt 大概解决了三类问题：

| 轮次 | 主要目标 | 结论 |
| --- | --- | --- |
| 输出协议 | 日期不能乱、餐次不能缺、酒店字段不能飘、JSON 不能半截 | prompt 能明显提升 shape 稳定性，但还要配合 parser 和 validation |
| 餐饮 grounding | 餐厅必须来自候选，不写“附近小吃”“当地特色餐厅” | prompt 里写还不够，评测也必须能抓没 grounded 的输出 |
| 伪精确路线 | 不写工具没给过的“步行 10 分钟”“打车 15 分钟” | 这类 hallucination 更适合在输出规则里拦掉 |

这一步最有价值的不是 prompt 本身，而是失败画像。看完 bad case 以后，问题自然会分层：schema、日期、餐次缺失适合 shape validation；餐厅和景点不 grounded 适合 prompt 加规则评测一起压；预算关系复杂，最好拆成工程重算和模型选择两部分；偏好满足度不够，可能需要数据补齐。

Prompt 调试的终点不是“再写长一点”，而是知道什么时候该停。

---

## 第六章：SFT 数据生成和审计

SFT 数据生成是最容易让人放松警惕的一步。

强模型确实能生成很像样的旅行计划，但“像样”不等于“能训练”。如果 teacher 输出里预算口径错、餐厅不 grounded、酒店每天乱换，学生模型会学得更稳定，也会更稳定地错。

所以我把数据生成拆成几步：

1. 先 dry-run 请求分布，看城市、天数、预算、同行类型是否合理。
2. 再 dry-run `PlannerContext`，确认工具候选、价格 hint、天气和预算策略都能编译出来。
3. 小批量生成，先跑 20 到 100 条，不要一上来就生成上千条。
4. 每次强模型调用都记录 usage、manifest 和 run 配置。
5. 生成后先审计，再进入训练集。

![SFT 数据生成与审计漏斗图](./images/Extra12-figures/05-SFT数据生成与审计漏斗图.png)

审计里最关键的是硬过滤：

| 过滤项 | 为什么重要 |
| --- | --- |
| JSON / schema 合法 | 后端必须能解析 |
| 日期和天数一致 | 旅行计划不能少天、多天、错日期 |
| 酒店和餐厅 grounded | 不能凭空编候选 |
| 餐饮不重复 | 不能连续几顿同一家快餐 |
| 预算 hard constraint | 硬预算不能超 |
| 预算关系合理 | 酒店晚数、门票人数、餐饮尺度要对 |

还有一个容易忽略的点：旧数据不要舍不得。项目早期有一批旧 SFT 数据，局部看挺干净，但来自旧预算口径。后来我选择全量归档，不再修修补补继续用。这个决定当时有点痛，但回头看是对的。后训练最怕“新协议 + 旧口径数据”混在一起，模型表面学到了更多样本，实际学到的是互相冲突的规则。

---

## 第七章：LoRA SFT 多阶段训练

有了数据之后，才真正进入 LoRA SFT。

这条线使用 Qwen2.5-7B-Instruct 做 LoRA。训练不是一轮完成，而是多阶段推进。我的原则是：**尽量少同时改变量。**

很多参数一直没动：

| 参数 | 主线设置 | 为什么这样设 |
| --- | --- | --- |
| LoRA rank | `r=32` | 长 JSON 协议、候选复制、预算口径都要学，容量不能太小 |
| `lora_alpha` | `64` | 和 r32 搭配，后面不频繁改 |
| `lora_dropout` | `0.05` | 防止小数据阶段过拟合 |
| `target_modules` | `all` | Planner 任务不只是语言风格，还涉及结构化选择 |
| `cutoff_len` | `24576` | `PlannerContext` 很长，降到 16k 会截掉上下文信号 |
| batch | `micro_batch_size=1`，`global_batch_size=32` | 单卡放不下大 batch，就用梯度累积 |
| 精度与显存 | bf16 + activation checkpointing | 长上下文训练的基本生存配置 |

真正反复调的是三类东西：数据、学习率、训练轮数。

| 阶段 | 起点 | 数据 | 主要参数 | 想解决什么 |
| --- | --- | --- | --- | --- |
| main clean lr sweep | base Qwen2.5-7B | `main_clean` | `lr=8e-5 / 6e-5`，`epoch=4` | 先学稳 TripPlan 协议 |
| usage700 mixed | 从 `lr6e-5` adapter 接着训 | main clean + realbudget usage700 | `lr=2e-5`，`epoch=1` | 补预算使用和真实预算口径 |
| patch700 only | 从 `lr6e-5` adapter 接着训 | budget utilization patch 700 | `lr=1e-5`，`epoch=2` | 诊断预算利用型补数上限 |
| Best-of-N 600 replay | 从 usage700 adapter 接着训 | old replay + Best-of-N winner | `lr=1e-5`，半轮保存 | 注入规则筛出来的更好候选 |
| Best-of-N 1200 retry | 从 Best-of-N 600 final 接着训 | old replay + 更多 Best-of-N winner | `lr=1e-5`，半轮保存 | 增加 winner 占比，看是否继续提升 |

![LoRA 多阶段训练时间线图](./images/Extra12-figures/06-LoRA多阶段训练时间线图.png)

这里有个细节很容易混：`adapter_name_or_path` 不是 `resume_from_checkpoint`。它只是拿上一轮导出的 LoRA adapter 做 warm-start，优化器状态不会接着上一轮走。也就是说，每一阶段都会重新使用当前配置里的学习率和调度器。

这反而适合阶段实验。上一轮学到的能力留在 adapter 里，下一轮用更小的学习率继续修局部问题。

学习率一路往下降，也是这个原因：

```text
main clean:       6e-5 / 8e-5
usage700 mixed:   2e-5
patch700 only:    1e-5
Best-of-N replay: 1e-5
DPO closing:      1e-6 到 1.5e-6 级别
```

越往后，数据越像在修局部问题。学习率太高，预算指标可能上去了，餐饮 grounding、住宿连续性或者日期天气又掉下来。

---

## 第八章：Best-of-N Replay 和 SFT Rerank

SFT 学稳协议后，我先做 Best-of-N replay，再做最终展示阶段的 rerank。名字有点像，但它们不是一件事。

![Best-of-N Replay 与 Rerank 图](./images/Extra12-figures/07-Best-of-N-Replay与Rerank图.png)

Best-of-N Replay 是训练数据构造流程：同一个 `PlannerContext`，让当前模型采样多个答案，用规则评估器挑一个更好的，再把 winner 导出成下一轮 SFT 数据。

```text
PlannerContext
  -> t=0.2 / 0.5 / 0.8 多温度采样
  -> 每个候选跑 rule metrics
  -> 优先选 hardpass 候选
  -> 再看预算、餐饮尺度、多样性等软奖励
  -> winner 进入下一轮 SFT
```

最终 Rerank 是推理时流程：同一个 prompt 生成多个候选，不再把 winner 写回训练集，而是在线上从候选池里选一个更稳的答案返回给用户。

这两个流程都要回到 frozen eval 看全局指标。原因也简单：规则挑 winner 是有偏的。如果 reward 过度偏向某个指标，模型可能会变保守，也可能牺牲体验。只看单个 winner 的分数，很容易误判。

SFT 阶段接入多温度候选 + 规则 rerank 后，几个版本整体上了一个台阶：

| 版本 | hardpass | softpass | 重算预算 softpass | 预算算术 | 预算偏好 | 预算关系 | 餐饮尺度 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ckpt104 + rerank | 98.0 | 65.6 | 54.6 | 81.2 | 77.0 | 86.4 | 88.8 |
| final1200 + rerank | 98.2 | 66.8 | 54.6 | 78.0 | 78.4 | 85.0 | 88.0 |
| old600final + rerank | 98.2 | 66.2 | 59.2 | 78.4 | 75.4 | 87.0 | 89.4 |

![SFT Rerank 对比图](./images/Extra12-figures/09-sft-rerank-comparison.png)

到这里，SFT 阶段可以收束。继续追加 SFT 的收益已经变钝，后面的主要增益应该来自偏好数据和候选选择。

---

## 第九章：DPO 学偏好，核心指标换成 PlannerSoft

SFT 已经能把 TripPlan 的壳子写稳，但合法答案之间也有好坏。两个计划都能过 schema，都能找到酒店和餐厅，一个可能很省但不像用户想要的旅行，另一个预算更贴合、餐饮更少重复、景点也更顺。

SFT 很难从单条 teacher 里稳定学到这种取舍，DPO 更适合做这件事。

我这里没有把 DPO 当成万能增强。它只做一件事：**在 hardpass 已经过关的候选里，学习哪个更像一个好行程。**

### DPO pair 先过硬门槛

DPO pair 的 chosen / rejected 不能乱来。坏 JSON 对好 JSON，这种 pair 对模型当然有信号，但它学到的是格式，不是偏好。这个项目里更有用的是下面这种 pair：

```text
同一个 PlannerContext
  -> chosen: schema 过、hardpass 过、planner soft 过
  -> rejected: schema 过、hardpass 过，但预算/重复/偏好没过
```

这样训出来的模型才是在合法计划之间学选择，而不是重新学怎么写 JSON。

还有一条底线：不能从冻结评测集里挖训练 pair。预算收尾数据里专门做了签名过滤：

```text
frozen eval signature count = 497
selected eval signature overlap = 0
```

这件事很烦，但必须做。不然分数看起来好，实际上是在背题。

![DPO 样本筛选与防泄漏图](./images/Extra12-figures/12-DPO样本筛选与防泄漏图.png)

### 主指标换成 PlannerSoft

后来我越来越觉得，普通 softpass 还不够贴近真实体验。旅行助手的输出不是一道选择题，它是一个可以被用户拿去执行的计划。所以核心指标逐步转成 `planner soft`：预算贴合、餐饮重复、景点重复、预算关系这些都要看。

![PlannerSoft 指标分解图](./images/Extra12-figures/16-PlannerSoft指标分解图.png)

几轮 DPO 的路线大概是：

| 阶段 | 目的 | 结论 |
| --- | --- | --- |
| 高置信偏好 DPO 试跑 | 先验证长上下文 DPO 能跑通 | 流程跑通，后面开始换指标 |
| PlannerSoft 规则 DPO | 把优化目标从 hardpass 转向 planner soft | checkpoint-25 成为下一轮起点 |
| PlannerSoft 扩数据 + Direct 锚定 | 扩大 planner soft 数据，同时保留 direct preference | 形成后续 ckpt126 起点 |
| PlannerSoft Clean 单生成提升 | 用更大规模 clean 数据继续训 | `checkpoint-138` 成为单生成最佳点 |
| 预算收尾 DPO | 针对预算偏保守、超支、重复构造 clean pair | 单生成没继续涨，但改变了候选分布 |

DPO loss 也不能跨批次硬比。前几轮 pair 很容易分，loss 低、accuracy 高；预算收尾 pair 更接近，chosen 和 rejected 都是 hardpass 计划，只是在预算使用、重复和偏好上有差别，loss 自然会更高。

![DPO loss 跨批次不可比图](./images/Extra12-figures/15-DPO-loss跨批次不可比图.png)

我后来更看重两个信号：reward accuracy 有没有稳定上来，frozen eval 上 planner soft 和预算相关指标有没有真的动。训练日志让你知道这炉有没有坏，评测才告诉你这炉有没有用。

---

## 第十章：最终多候选 Rerank

DPO 后半段最容易误读。单看单生成，`260519 checkpoint-138` 更稳；继续做预算收尾训练后，`ckpt66` 和 `ckpt64` 没有把单生成分数继续推高。

但最终展示版本不是单生成。它是多候选 rerank。

![单生成与多候选 Rerank 对比图](./images/Extra12-figures/17-单生成与多候选Rerank对比图.png)

最终几组指标大概是：

| 版本 | hardpass | planner soft | 重算预算 soft |
| --- | ---: | ---: | ---: |
| ckpt126 baseline | 98.4% | 66.9% | 48.5% |
| 260519 ckpt138 single | 98.4% | 71.5% | 50.9% |
| 260520 ckpt66 single | 99.0% | 70.1% | 48.3% |
| 260521 ckpt64 single | 98.2% | 69.7% | 47.6% |
| 260521 ckpt64 rerank n4 | **99.4%** | **80.6%** | **68.2%** |

![DPO 收尾 Rerank 效果图](./images/Extra12-figures/10-dpo-rerank-closing.png)

这里的结论不是“最后一炉单生成最好”。更准确的说法是：

- 单生成最佳：`260519 ps2400clean_plus_direct402 checkpoint-138`。
- 多生成 rerank 最佳：`260521 closing checkpoint-64 rerank n4`。
- 展示主推：`ckpt64_rerank_n4`，500 条 planner soft `80.6%`，hard split planner soft `77.0%`。

单生成看的是一次采样的平均质量；rerank 看的是候选池里有没有更好的答案，以及规则能不能把它选出来。两者可以不是同一个 checkpoint。

---

## 第十一章：和 MiMo 外部参考怎么比

最后可以加一个外部强模型参照，但这块一定要写清楚口径。MiMo 不是我们这条 LoRA 训练线里的 checkpoint，也不是严格同一套脚本、同一版规则下的 leaderboard。更合适的用法是：看它告诉我们强模型大概会在哪里强，哪里和本地规则不完全合拍。

历史上跑过 `mimo_v2_5_pro_external_mt1p5`，它是 MiMo v2.5 Pro 外部 API，w50，max token 按 1.5x 放大。和最终 `ckpt64_rerank_n4` 放在一起看，大概是这样：

| 模型 | hardpass | planner soft | 重算预算 soft | 预算偏好 | 重算预算贴合 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MiMo v2.5 Pro mt1p5 | 98.8% | 78.7% | **76.6%** | 85.5% | **82.4%** |
| ckpt64_rerank_n4 | **99.4%** | **80.6%** | 68.2% | **86.0%** | 73.4% |

![MiMo 外部参考对比图](./images/Extra12-figures/11-mimo-reference-comparison.png)

这张表可以这么读：

- 本地最终版在本项目规则口径下，`hardpass` 和 `planner soft` 已经追上并略高于 MiMo 参考。
- MiMo 的重算预算 soft 和重算预算贴合仍然更强，说明预算总额控制这件事它做得更稳。
- MiMo 的预算关系、餐饮尺度在早期报告里不算高，主要是它会给出更真实的人均餐费，但这些餐费有时低于我们当前规则档位的下限。

所以这里不把结论写成“全面超过 MiMo”。更准确的说法是：在本项目冻结评测和规则口径下，最终本地模型的 planner soft 已经追平强模型参考；预算贴合仍有差距，后续如果继续做，应该补预算总额控制和预算档位之间的协调。

---

## 第十二章：这次实验留下来的经验

这次旅行助手后训练给我的最大教训是：Agent 后训练不是单纯“多造点数据再训一下”。它更像把产品协议、数据、训练、评测和推理策略一层层对齐。

最后留下来的经验大概是这些：

1. **先改产品协议。** 能结构化提交的字段，不要让模型猜。
2. **把上下文编译好。** 模型应该基于候选做选择，而不是凭空写事实。
3. **评测集先冻结。** 训练可以换数据，评测不要每轮换题。
4. **Prompt 调试用来找边界。** prompt 能解决的写进 prompt，解决不了的交给数据、规则或工程。
5. **强模型生成数据，但 teacher 必须被审计。** 像样不等于可训练。
6. **LoRA SFT 分阶段推进。** 每轮尽量只解决一类主要问题。
7. **Best-of-N Replay 和最终 Rerank 要分清。** 前者造训练数据，后者推理时选答案。
8. **DPO 只在合法候选之间学偏好。** 不要把格式修复 pair 当成偏好 pair。
9. **训练数据要避开冻结评测集。** 实验项目也没必要背题，重叠检查很便宜。
10. **指标拆开看。** hardpass、planner soft、预算关系、重算预算不要揉成一个总分。

如果只用一句话概括，就是：

> 能结构化的交给工程，能规则化的做成评测，必须由模型学习的再进入 SFT 或偏好训练。

这样做出来的模型不会只是“更会说”，而是更接近一个能被产品接住、能被指标解释、也能继续迭代的 Planner。

## 复现资源

如果你想完整复现，可以从这些材料开始：

- 旅行助手项目仓库：[https://github.com/nameless0120/helloagents-trip-planner](https://github.com/nameless0120/helloagents-trip-planner)
- 完整教程：[旅行助手后训练实战教程](https://github.com/nameless0120/helloagents-trip-planner/blob/main/training/docs/%E6%95%99%E7%A8%8B/%E6%97%85%E8%A1%8C%E5%8A%A9%E6%89%8B%E5%90%8E%E8%AE%AD%E7%BB%83%E5%AE%9E%E6%88%98%E6%95%99%E7%A8%8B.md)
- 配套数据：`helloagents-后训练数据`，<https://pan.baidu.com/s/5oNsK7pwQnqzQEUg5ykb09Q>

主线 LoRA 复现建议配置是 2 张 40GB 级别 GPU。4 张 40GB 会更舒服，尤其是多轮训练和并行评测时。2 张 24GB 可以做短上下文或 QLoRA 实验，但不建议拿来复现这条长上下文主线，因为 `cutoff_len=24576`、LoRA r32、bf16、activation checkpointing、FSDP2 + CP=2 是这条实验线的一部分。把这些砍掉，也能跑，但就不是同一个实验了。
