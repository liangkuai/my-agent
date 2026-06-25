# Learn Claude Code

> 参考：[learn-claude-code](https://github.com/shareAI-lab/learn-claude-code)

## 一句话主旨

**Agent = 模型(Model)+ 框架(Harness)。** 智能(agency)来自模型训练,不来自外部代码编排。本仓库教你构建"框架"——让一个已经聪明的模型,能在特定领域(写代码)中真正运作起来的运行环境。模型是司机,Harness 是车,这个仓库教你造车。

## 四个核心论点

1. **智能是训练出来的,不是编码出来的**
   以一连串里程碑为证,它们架构完全一致("训练好的模型 + 操作环境"):
   - 2013 DeepMind DQN(仅凭像素学会 7 款 Atari)→ 2015 扩展到 49 款达职业水平
   - 2019 OpenAI Five(自对弈 4.5 万年,2-0 击败 Dota2 世界冠军 OG)
   - 2019 AlphaStar(星际2 欧服宗师,前 0.15%)
   - 2019 腾讯绝悟(王者荣耀 KPL 职业选手 1v1 15 局仅胜 1 局)
   - 2024-2025 LLM 编码 agent(Claude/GPT/Gemini 读代码、写实现、调 bug、组队协作)

2. **什么不是 Agent**
   拖拽式工作流、no-code 平台、prompt 链编排,共享同一个幻觉:把 LLM API 调用用 if-else、节点图、硬编码路由串起来就算"造 agent"。原文直言这只是"一个披着浮夸外衣的 shell 脚本"——你无法靠堆叠流程逻辑把智能"凑"出来。

3. **思维转变:从"造 Agent"到"造 Harness"**
   说"我在造 agent"只有两种真实含义:
   - **训练模型** —— 调权重(RL/微调/RLHF),Anthropic、DeepMind、OpenAI 等做的事
   - **构建 Harness** —— 写代码给模型一个操作环境,大多数人做的事,也是本仓库主题

   ```
   Harness = 工具 + 知识 + 观察 + 行动接口 + 权限
       工具:   文件 I/O、shell、网络、数据库、浏览器
       知识:   产品文档、领域参考、API 规范、风格指南
       观察:   git diff、错误日志、浏览器状态、传感器数据
       行动:   CLI 命令、API 调用、UI 交互
       权限:   沙箱隔离、审批流、信任边界
   ```

4. **为什么选 Claude Code**
   因为它是最优雅完整的 Harness 实现——关键在于它"不做什么":不试图取代模型判断、不强加僵化工作流、不用手写决策树替代模型自己的判断,只给模型工具/知识/上下文管理/权限边界,然后让开。剥到本质,Claude Code = 一个 agent loop + 工具 + 按需技能加载 + 上下文压缩 + 子 agent + 任务系统 + 邮箱协作 + worktree 隔离 + 权限治理 + hooks + 记忆 + MCP。

## Harness 工程师实际做什么

如果你在读这个仓库,你大概率是一名 harness 工程师,工作内容是 5 件事:

- **实现工具** —— 给 agent 一双手(读写文件、执行 shell、调 API、控浏览器、查数据库),设计成原子、可组合、描述清晰
- **策划知识** —— 给 agent 领域专长,按需加载而非一次性塞入
- **管理上下文** —— 给 agent 干净的记忆(子 agent 隔离防噪声泄漏,上下文压缩防历史淹没当下,任务系统让目标跨会话留存)
- **控制权限** —— 给 agent 边界(沙箱、破坏性操作需审批、信任边界)
- **收集轨迹数据** —— agent 在你 harness 中的每段行动序列都是训练信号,是微调下一代模型的原料

> 你不是在编写智能,而是在构建智能所栖居的世界;这个世界的质量,直接决定智能能多有效地表达自己。

## 核心模式:唯一不变的 agent loop

```python
def agent_loop(messages):
    while True:
        response = client.messages.create(...)
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return
        results = [执行每个 tool_use,收集 tool_result]
        messages.append({"role": "user", "content": results})
```

调用 LLM → 若 `stop_reason == "tool_use"` 则执行工具、把结果追加回 `messages` → 循环。**每节课只在这个循环外叠加一个 harness 机制,循环本身永不改变。** 循环属于 agent,机制属于 harness。

## 课程结构(两条线)

- **新版主线:根目录 `s01`–`s20`**,20 节课,每节配完整 README、多语言翻译(en/ja)、可运行 `code.py`、必要时附 SVG 图
- **旧版过渡:`docs/`、`agents/`、`web/`**,12 节课老版本,为旧链接和 web 平台暂时保留;README 含新旧章节映射表(注意两套编号不一一对应,勿混用)

## 学习路径:两大 Phase × 每 Phase 三 Stage

**🌱 Phase 1(Stage 1-3):核心能力,由简到繁**

| Stage | 章节 | 主题 |
|---|---|---|
| 1. 让 Agent 行动 | s01-s04 | Agent Loop、Tool Use、Permission、Hooks |
| 2. 处理复杂工作 | s05、s06、s08 | TodoWrite、Subagent、Context Compact |
| 3. 记忆与恢复 | s09-s11 | Memory、System Prompt、Error Recovery |

**🚀 Phase 2(Stage 4-6):进阶能力,长任务/协作/集成**

| Stage | 章节 | 主题 |
|---|---|---|
| 4. 运行长任务 | s12-s14 | Task System、Background Tasks、Cron Scheduler |
| 5. 多 Agent 协作 | s15-s18 | Agent Teams、Team Protocols、Autonomous Agents、Worktree Isolation |
| 6. 扩展与组装 | s07、s19、s20 | Skill Loading、MCP Plugin、Comprehensive Agent |

> 注意:s07 Skill Loading 按主题归入第 6 阶段,不按数字顺序排列。

## 每章 motto 速览

| 章 | 口号(意译) |
|---|---|
| s01 | 一个循环 + Bash 足矣 |
| s02 | 加工具只是加一个 handler,循环不动 |
| s03 | 先立边界,再给自由 |
| s04 | 在循环周围挂钩子,绝不改写循环 |
| s05 | 没计划的 agent 会漂移,先列步骤完成率翻倍 |
| s06 | 大任务拆小,每个子任务一份干净上下文 |
| s07 | 知识按需加载,而非一次性塞满 |
| s08 | 上下文总会填满,得有腾地方的办法 |
| s09 | 记住重要的,忘掉无关的 |
| s10 | Prompt 运行时拼装,而非硬编码 |
| s11 | 出错不是终点,是重试的起点 |
| s12 | 大目标拆成有序小任务,落盘持久化 |
| s13 | 慢操作转后台,agent 继续思考 |
| s14 | 按时触发,无需人工启动 |
| s15 | 一个 agent 装不下就委派给队友 |
| s16 | 队友需要共享的沟通规则 |
| s17 | 队友自己看板认领工作,无需逐个分配 |
| s18 | 各在各的目录工作,互不干扰 |
| s19 | 能力不够?用 MCP 接入更多工具 |
| s20 | 众多机制,归于一个循环 |

## 其他要点

- **范围说明**:0-to-1 教学项目,刻意简化或省略部分生产机制(完整 hook 事件总线、规则化权限治理、session resume/fork、MCP 运行时传输/OAuth/订阅等);JSONL 邮箱协议为教学实现,非生产内部实现声明
- **快速开始**:`git clone` → `pip install -r requirements.txt` → 配置 `.env` 中 `ANTHROPIC_API_KEY` → 从 `python s01_agent_loop/code.py` 开始
- **产品化路径**:Kode Agent CLI(`npm i -g @shareai-lab/kode`)、Kode Agent SDK
- **姊妹仓库**:[claw0](https://github.com/shareAI-lab/claw0) 讲"常驻型 agent",在同一 agent 核心上加 `heartbeat + cron + IM 多渠道 + memory + soul`,把"戳一下动一下"的工具变成全天候个人助手

## 收尾金句(原文,分三段)

> Agency comes from the model. The harness gives agency a place to land. Build the harness well, and the model will do the rest.
>
> Bash is all you need. Real agents are all the universe needs.
>
> This is not "copy the source code." This is "grasp the key designs and build it yourself."

—— *智能来自模型,Harness 给智能一个落脚点;把 Harness 造好,模型会完成其余的一切。这不是"抄源码",而是"抓住关键设计,自己把它造出来"。*
