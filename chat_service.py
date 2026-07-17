"""
需求：SmartVoyage核心对话服务，供CLI和API共同调用

架构说明：
    本模块实现了智能旅游助手的核心对话流程，采用优化的 Planning + ReAct 架构：
    1. 意图识别（intent_agent）：分析用户输入，识别出用户想做什么（查天气/查机票/查景点等）
    2. 启发式路由（_should_skip_planning）：快速判断任务复杂度
       - 单意图或独立多意图：跳过规划，直接执行（省去 1 次 LLM 调用）
       - 含依赖关系的复杂任务：调用 planning_agent 生成执行计划
    3. ReAct循环（react_loop）：针对复杂任务，逐步执行每个步骤
       - Action（行动）：直接调用对应的 agent 执行查询（跳过 Thought 推理，省去每步 1 次 LLM 调用）
       - Observation（观察）：记录执行结果，供后续步骤参考
    4. 结果汇总：所有步骤完成后，生成一条连贯的最终回复

整体流程：
    用户输入 → 意图识别 → [跳过规划 / 调用planning_agent] → [直接执行 / ReAct循环] → 返回回复

A2A架构说明：
    Agent2Agent (A2A) 是一种代理间通信协议。本系统通过 AgentNetwork 管理多个子代理：
    - WeatherAssistant（天气查询代理）：运行在 127.0.0.1:5005
    - TicketAssistant（票务代理）：运行在 127.0.0.1:5006
    - TripAssistant（行程管家代理）：运行在 127.0.0.1:5007
    主助手通过 A2A 协议向子代理发送任务请求，获取查询结果后整合回复用户。

Mermaid 流程图：

```mermaid
flowchart TD
    Start([用户输入]) --> SaveUser["写入记忆 memory"]
    SaveUser --> Intent["① 意图识别 intent_agent()\nLLM → intents + queries"]

    Intent --> Check{意图有效?}
    Check -- "out_of_scope" --> End1([返回系统提示])
    Check -- "需追问" --> End2([返回追问消息])
    Check -- 有效 --> Route{"② 启发式路由\n单意图 / 独立多意图?"}

    Route -- 简单 --> Parallel["④-A 并行执行\nasyncio.gather()\n各意图调 _call_agent_intent()"]
    Route -- 复杂 --> Plan["③ 任务规划 planning_agent()\nLLM → 执行计划 steps"]

    Plan --> NeedPlan{need_plan?}
    NeedPlan -- 否 --> Parallel
    NeedPlan -- 是 --> React["④-B ReAct 循环\n按依赖分组 → 并行执行 → 汇总"]

    Parallel --> SaveResp["写入记忆 → 返回回复"]
    React --> SaveResp

    style Intent fill:#e8f4fd,stroke:#2196F3
    style Route fill:#fff3e0,stroke:#FF9800
    style Plan fill:#fce4ec,stroke:#E91E63
    style Parallel fill:#e8f5e9,stroke:#4CAF50
    style React fill:#f3e5f5,stroke:#9C27B0
```

_call_agent_intent() 内部路由：

```mermaid
flowchart TD
    Entry(["_call_agent_intent(intent, query_str)"]) --> CheckIntent{intent 类型?}

    CheckIntent -- attraction --> AttrBlock
    subgraph AttrBlock ["景点推荐（直连 LLM）"]
        direction TB
        A1["_fetch_weather_for_attraction()\n提取城市 → 调天气代理获取天气"]
        A2["attraction_prompt() + LLM\n生成景点推荐"]
        A1 --> A2
    end
    AttrBlock --> RetAttr([返回推荐结果])

    CheckIntent -- flight/train/concert/\ncar_rental/tour_group/insurance --> AgentBlock
    subgraph AgentBlock ["票务/行程（A2A 子代理）"]
        direction TB
        B1["memory.extract_entities()\n提取城市、日期等实体 → 持久化到 DB"]
        B2["memory.update_task_context()\n更新当前任务上下文"]
        B3["构建 agent_input:\n最近2轮对话 + 当前查询"]
        B4["A2A 协议发送:\nagent.send_task_async(task)"]
        B5{"子代理响应状态?"}
        B6["artifacts 有结果\n→ 直接返回"]
        B7["status.message 有内容\n→ 返回追问/错误信息"]
        B8["未知状态\n→ '查询失败'"]
        B1 --> B2 --> B3 --> B4 --> B5
        B5 -- completed --> B6
        B5 -- 有 message --> B7
        B5 -- 其他 --> B8
    end
    AgentBlock --> RetAgent([返回代理结果])

    CheckIntent -- 未识别 --> RetUnknown(["'暂不支持此意图'"])

    style AttrBlock fill:#e8f5e9,stroke:#4CAF50
    style AgentBlock fill:#e8f4fd,stroke:#2196F3
```

子代理调用关系：

```mermaid
flowchart LR
    CS["ChatService\n主助手"] -->|A2A :5005| WA["WeatherAssistant\n天气查询\nMCP→和风天气API"]
    CS -->|A2A :5006| TA["TicketAssistant\n票务查询/预定\nMCP→MySQL"]
    CS -->|A2A :5007| TR["TripAssistant\n行程管家\nMCP→MySQL+向量库"]
    CS -->|直连 LLM| AT["景点推荐\nattraction_prompt()"]
```

优化策略总结:

    1. 启发式路由   → 简单任务跳过 planning_agent，省 1 次 LLM 调用
    2. 省略 Thought → ReAct 循环中跳过 Thought 推理，每步省 1 次 LLM 调用
    3. 并行执行     → 无依赖步骤通过 asyncio.gather() 并发执行
    4. 工具缓存     → _available_tools_text 缓存，避免重复查询 agent card
    5. 跳过总结     → 子 Agent 已格式化输出，主助手直接返回，省 1 次 LLM 调用
"""

# ==================== 导入依赖 ====================
import asyncio  # 异步IO库，用于并发调用多个 agent（异步网络请求）
import json  # JSON处理，用于解析大模型的JSON输出和序列化数据
import uuid  # 唯一标识符生成，用于给每个任务分配唯一ID
from datetime import datetime  # 时间处理，用于获取当前日期和时间戳
import pytz  # 时区库，用于将时间转换到 Asia/Shanghai 时区
import re  # 正则表达式，用于清理大模型返回的JSON（去掉代码块标记）
import mysql.connector  # MySQL 数据库驱动，用于记忆持久化

# python_a2a 是 A2A 协议的 Python 客户端库
from python_a2a import AgentNetwork, TextContent, Message, MessageRole, Task
from python_a2a.client import A2AClient
# AgentNetwork: 代理网络管理器，负责注册和获取子代理
# A2AClient: A2A 客户端，支持自定义超时
# TextContent: 消息文本内容包装
# Message: A2A 消息对象，包含角色和内容
# MessageRole: 消息角色枚举（USER/ASSISTANT）
# Task: A2A 任务对象，用于向子代理发送请求

from langchain_openai import ChatOpenAI  # LangChain 的大模型接口，兼容 OpenAI API 格式
from langchain_core.prompts import ChatPromptTemplate  # LangChain 的提示模板类

# 导入项目内部模块
from config import Config  # 配置模块（模型地址、API Key、意图映射等）
from create_logger import logger  # 日志模块，用于记录运行日志
from main_prompts import SmartVoyagePrompts  # Prompt 模板管理
from memory import ConversationMemory  # 记忆管理（短期记忆、用户偏好、任务上下文）

conf = Config()  # 全局配置实例，包含模型参数、意图映射等


class ChatService:
    """
    SmartVoyage 对话服务类

    这是整个系统的核心类，负责处理用户的所有输入并生成回复。
    内部流程为：意图识别 → 任务规划 → 执行（直接/ReAct循环）→ 返回回复

    使用示例：
        service = ChatService()          # 创建服务实例
        response = await service.chat("北京明天天气怎么样？")  # 发送用户输入
        print(response)                  # 打印助手回复
    """

    def __init__(self):
        """
        初始化 ChatService 实例

        完成以下初始化工作：
        1. 配置子代理网络（注册天气代理和票务代理）
        2. 初始化大模型连接
        3. 初始化记忆管理器
        4. 初始化对话历史和工具列表缓存
        """
        # ========== 1. 配置 A2A 代理网络 ==========
        # 记录每个子代理的访问地址
        self.agent_urls = {
            "WeatherAssistant": "http://127.0.0.1:5005",  # 天气查询代理的地址
            "TicketAssistant": "http://127.0.0.1:5006",        # 票务代理的地址
            "TripAssistant": "http://127.0.0.1:5007"           # 行程管家代理的地址
        }
        # 创建代理网络实例，给它起个名字
        self.agent_network = AgentNetwork(name="旅行助手网络")
        # 向网络中注册子代理，使用自定义超时（120秒），避免子代理 LLM+MCP 调用链超时
        # 注意：调用链 = LLM选工具(5-15s) + MCP/MySQL(1-3s) + LLM格式化(5-15s)，总计可达 33s+
        self.agent_network.add("WeatherAssistant", A2AClient("http://127.0.0.1:5005", timeout=300))
        self.agent_network.add("TicketAssistant", A2AClient("http://127.0.0.1:5006", timeout=300))
        self.agent_network.add("TripAssistant", A2AClient("http://127.0.0.1:5007", timeout=300))

        # ========== 2. 初始化大模型连接 ==========
        # 使用 LangChain 的 ChatOpenAI 接口连接大模型
        self.llm = ChatOpenAI(
            model=conf.model_name,     # 模型名称
            api_key=conf.api_key,      # API 密钥
            base_url=conf.base_url,    # API 基础地址
            temperature=0.1            # 温度参数：越小输出越确定、越稳定（0.1表示比较严谨）
        )

        # ========== 3. 初始化记忆和对话状态 ==========
        self.memory = ConversationMemory(short_term_limit=10)  # 短期记忆最多保留10条对话
        self.messages = []           # 对话消息列表，格式：[{"role": "user/assistant", "content": "..."}]
        self.conversation_history = ""  # 完整对话历史字符串，用于调试和展示
        self._available_tools_text = "" # 缓存的可用工具列表文本，避免重复查询
        self._max_messages = 20         # 消息列表上限（10轮对话），防止长会话内存泄漏

        # 城市提取 prompt，用于从景点推荐查询中提取目的地城市
        self._city_extract_prompt = ChatPromptTemplate.from_template(
"""从以下用户查询和对话历史中提取目的地城市（用于天气查询）。规则：
- 只输出城市名称，不要输出其他内容。
- 如果查询中没有明确城市，尝试从对话历史中推断。
- 如果无法确定城市，输出"未知"。

对话历史：{conversation_history}
用户查询：{query}
""")

        # ========== 4. 初始化数据库连接，加载持久化记忆 ==========
        self._init_db_and_load_memory()

    def _add_message(self, role: str, content: str):
        """
        添加消息到 self.messages 和 self.conversation_history，并自动裁剪超出上限的旧消息。
        替代直接调用 self.messages.append() 和 self.conversation_history += ...
        """
        self.messages.append({"role": role, "content": content})
        prefix = "User" if role == "user" else "Assistant"
        self.conversation_history += f"\n{prefix}: {content}"

        # 裁剪：保留最近 _max_messages 条消息，防止长会话内存泄漏
        if len(self.messages) > self._max_messages:
            self.messages = self.messages[-self._max_messages:]
            # 从裁剪后的列表重建 conversation_history（避免字符串无限增长）
            self.conversation_history = "\n".join([
                f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                for m in self.messages
            ])

    def get_available_tools(self) -> str:
        """
        从 A2A 代理网络中动态获取可用工具列表，格式化为文本

        这是 ReAct 架构的关键：告诉大模型当前有哪些工具/agent 可以调用，
        让大模型在推理时从这些工具中选择合适的。

        优势：当新增或删除 agent 时，无需修改 prompt 模板，系统自动感知变化。

        返回值：
            str: 格式化的工具列表文本，例如：
                - Agent: WeatherAssistant, 查询天气信息
                  Skill: 天气查询: 根据城市查询天气
                - Agent: TicketQueryAssistant, 查询票务信息
                  Skill: 票务查询: 查询机票、火车票等
        """
        try:
            # 调用 get_agent_cards() 获取所有在线 agent 的卡片信息
            cards = self.get_agent_cards()
            lines = []
            for card in cards:
                # 每个 agent 一行，包含名称和描述
                lines.append(f"- Agent: {card['name']}, {card['description']}")
                # 每个 agent 的技能作为子项列出
                for skill in card['skills']:
                    lines.append(f"  Skill: {skill}")
            self._available_tools_text = "\n".join(lines)
        except Exception as e:
            # 如果 agent server 没有启动或网络不通，使用默认的工具列表作为降级方案
            logger.warning(f"获取可用工具列表失败: {e}")
            self._available_tools_text = (
                "- WeatherAssistant: 查询天气信息\n"
                "- TicketQueryAssistant: 查询票务信息（机票/高铁票/演唱会票）\n"
                "- TicketAssistant: 预定票务\n"
                "- TripAssistant: 行程管家（租车/旅游团/保险的查询与预订）"
            )
        return self._available_tools_text

    def _init_db_and_load_memory(self):
        """
        初始化数据库连接，并从数据库加载持久化记忆

        启动流程：
        1. 创建 MySQL 连接，注入到 memory 实例
        2. 从数据库加载用户偏好、查询历史、短期对话
        """
        try:
            db_conn = mysql.connector.connect(
                host=conf.host,
                port=conf.port,
                user=conf.user,
                password=conf.password,
                database=conf.database
            )
            self.memory.set_db_connection(db_conn)
            logger.info("数据库连接初始化成功")

            # 从数据库恢复记忆数据
            self.memory.load_profile_from_db()
            self.memory.load_entities_from_db()
            self.memory.load_messages_from_db()

            # 将加载的短期消息同步到 messages 列表
            for msg in self.memory.short_term_messages:
                self.messages.append({"role": msg["role"], "content": msg["content"]})

            logger.info(f"记忆加载完成: 偏好={len(self.memory.user_profile)}项, "
                        f"历史={len(self.memory.entity_history)}条, 对话={len(self.memory.short_term_messages)}条")
        except Exception as e:
            logger.warning(f"数据库连接初始化失败: {e}")
            self.memory.set_db_connection(None)

    async def _fetch_weather_for_attraction(self, query_str: str) -> str:
        """
        为景点推荐查询目的地天气，返回简要天气信息。
        如果无法确定目的地，返回空字符串。

        参数：
            query_str (str): 用户的景点查询内容

        返回值：
            str: 天气信息摘要，或空字符串
        """
        try:
            # 从查询中提取目的地城市
            chain = self._city_extract_prompt | self.llm
            city = chain.invoke({
                "conversation_history": self.memory.get_short_term_text(),
                "query": query_str
            }).content.strip()

            if city == "未知" or not city:
                logger.info(f"无法从查询中提取目的地城市: {query_str}")
                return ""

            logger.info(f"从景点查询中提取到城市: {city}")

            # 通过天气代理查询该城市天气
            agent = self.agent_network.get_agent("WeatherAssistant")
            if agent is None:
                return ""

            weather_query = f"{city}明天天气"
            msg = Message(content=TextContent(text=weather_query), role=MessageRole.USER)
            task = Task(id="task-" + str(uuid.uuid4()), message=msg.to_dict())

            # raw_response = await agent.send_task_async(task)
            raw_response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: asyncio.run(agent.send_task_async(task))
            )

            if raw_response.status.state == 'completed' and raw_response.artifacts:
                return raw_response.artifacts[0]['parts'][0]['text']
            return ""

        except Exception as e:
            logger.warning(f"查询景点天气失败: {e}")
            return ""

    def update_user_profile(self, profile: dict):
        """
        更新用户偏好设置，由前端 API 直接调用

        参数：
            profile (dict): 用户偏好键值对，如 {"seat_type": "二等座", "cabin_type": "经济舱"}
                           会合并到现有偏好中（新增/覆盖），不会删除已有项
        """
        self.memory.update_profile(profile)
        logger.info(f"更新用户偏好: {profile}")

    def intent_agent(self, user_input: str):
        """
        意图识别：分析用户输入，判断用户想做什么

        工作流程：
        1. 将用户输入 + 对话历史 + 装用户偏好等上下文组后发送给大模型
        2. 大模型返回 JSON，包含识别到的意图列表、改写后的查询、追问消息

        支持的意图类型：
            - weather: 天气查询
            - flight: 机票查询
            - train: 高铁/火车票查询
            - concert: 演唱会票查询
            - order: 票务预定
            - attraction: 景点推荐
            - out_of_scope: 超出系统能力范围

        参数：
            user_input (str): 用户本次的输入内容

        返回值：
            tuple: (intents, user_queries, follow_up_message)
                - intents (list): 识别到的意图列表，如 ["weather", "flight"]
                - user_queries (dict): 改写后的查询，如 {"weather": "北京明天天气", "flight": "北京到上海机票"}
                - follow_up_message (str): 追问消息，当意图不明确时用于追问用户
        """
        # 组装 Prompt 模板 + 大模型，形成完整的处理链
        chain = SmartVoyagePrompts.intent_prompt() | self.llm

        # 获取当前日期，注入到 prompt 中（用户可能说"明天"、"后天"等相对时间）
        current_date = datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d')

        # 调用大模型，传入所有上下文信息
        intent_response = chain.invoke({
            "conversation_history": self.memory.get_short_term_text(),  # 短期对话历史
            "query": user_input,          # 用户本次输入
            "current_date": current_date, # 当前日期
            "user_profile": self.memory.get_profile_text(),  # 用户偏好（如"二等座"、"经济舱"）
            "task_context": json.dumps(self.memory.current_task, ensure_ascii=False)  # 当前任务上下文
        }).content.strip()  # .content 获取文本内容，.strip() 去掉首尾空白
        logger.info(f"意图识别原始响应: {intent_response}")

        # 大模型有时会返回 ```json ... ``` 格式的代码块，需要用正则去掉
        intent_response = re.sub(r'^```json\s*|\s*```$', '', intent_response).strip()
        logger.info(f"清理后响应: {intent_response}")

        # 将 JSON 字符串解析为 Python 字典
        intent_output = json.loads(intent_response)
        intents = intent_output.get("intents", [])  # 意图列表，如 ["weather", "flight"]
        user_queries = intent_output.get("user_queries", {})  # 改写后的查询字典
        follow_up_message = intent_output.get("follow_up_message", "")  # 追问消息
        logger.info(f"intents: {intents}||user_queries: {user_queries}||follow_up_message: {follow_up_message} ")

        return intents, user_queries, follow_up_message

    def _should_skip_planning(self, intents: list) -> bool:
        """
        启发式判断：是否可以跳过 planning_agent，直接执行

        判断规则（满足任一即跳过）：
        1. 单意图：只有一个意图，无需拆解
        2. 多意图但全部为独立查询类意图（weather/flight/train/concert/attraction），
           彼此之间无依赖，可以直接并行或串行执行

        需要进入 planning 的情况：
        - 包含 order（预订）等需要分步决策的意图
        - 意图之间存在明显的先后依赖（如先查天气再决定是否租车）

        优化收益：省去 1 次 LLM 调用（planning_agent），大幅降低简单请求的响应时间

        参数：
            intents (list): 识别到的意图列表

        返回值：
            bool: True 表示跳过 planning_agent，False 表示需要调用 planning_agent
        """
        # 规则1：单意图直接跳过
        if len(intents) <= 1:
            return True

        # 规则2：多意图但全部为独立查询类，无需分步规划
        # 执行顺序：1，1，1，1，1
        independent_intents = {"weather", "flight", "train", "concert", "attraction",
                               "car_rental", "tour_group", "insurance", "trip_order"}
        for intent in intents:
            if intent not in independent_intents:
                # 存在需要分步的意图（如 order），不跳过
                # TODO 如果大模型认为意图非常复杂，使用complex。这里命中False
                return False
        return True

    def planning_agent(self, intents: list, user_queries: dict) -> dict:
        """
        任务规划：直接进行任务的拆解

        注意：在调用此方法前会先通过 _should_skip_planning 做启发式判断，
        只有复杂任务才会进入此方法，省去不必要的 LLM 调用。

        参数：
            intents (list): 识别到的意图列表，如 ["weather", "flight", "attraction"]
            user_queries (dict): 改写后的查询字典，如 {"weather": "...", "flight": "..."}

        返回值：
            dict: 规划结果，格式为：
                - 简单任务：{"need_plan": false, "reason": "单意图，直接查询即可", "steps": []}
                - 复杂任务：{"need_plan": true, "reason": "多意图需要分步执行",
                            "steps": [{"step": 1, "action": "...", "intent": "weather", "depends_on": 0}, ...]}
        """
        chain = SmartVoyagePrompts.planning_prompt() | self.llm

        planning_response = chain.invoke({
            "conversation_history": self.memory.get_short_term_text(),
            "query": self.messages[-1]["content"] if self.messages else "",
            "intents": json.dumps(intents, ensure_ascii=False),
            "user_queries": json.dumps(user_queries, ensure_ascii=False)
        }).content.strip()
        logger.info(f"规划响应: {planning_response}")

        planning_response = re.sub(r'^```json\s*|\s*```$', '', planning_response).strip()
        plan = json.loads(planning_response)
        return plan

    async def _call_agent_intent(self, intent: str, query_str: str) -> str:
        """
        统一调用 agent 的底层逻辑

        这是一个私有方法（以 _ 开头），不直接对外暴露，供以下两个场景复用：
        1. 简单任务直接执行时（chat 方法的 else 分支）
        2. ReAct 循环中的步骤执行时（execute_step 方法）

        这样做的好处：避免了代码重复，修改 agent 调用逻辑时只需改一处。

        处理逻辑说明：
        - 如果是景点推荐（attraction），直接让大模型生成，不需要调子代理
        - 如果是其他意图，通过 A2A 协议调用对应的子代理
        - 收到结果后，根据代理类型做适当的总结（天气/票务需要总结，其他直接返回）

        参数：
            intent (str): 意图类型，如 "weather"、"flight"、"attraction"
            query_str (str): 查询内容，如 "北京明天天气"

        返回值：
            str: agent 返回的结果文本
        """
        # 根据意图类型，从配置中查找对应的 agent 名称
        agent_name = conf.intent.get(intent)

        # ========== 情况1：景点推荐 ===========
        # 景点推荐不需要调用子代理，直接让大模型生成推荐内容
        # 但会先查询目的地天气，确保推荐基于真实天气而非虚构
        if intent == "attraction":
            chain = SmartVoyagePrompts.attraction_prompt() | self.llm
            weather_info = await self._fetch_weather_for_attraction(query_str)
            return chain.invoke({"query": query_str, "weather_info": weather_info}).content.strip()

        # ========== 情况2：需要调用子代理 ===========
        elif agent_name:
            # 对于票务查询类意图，提取关键实体（城市、日期等）保存到记忆中
            # 这样后续对话可以引用这些实体（多轮对话场景）
            # extract_entities 内部已自动持久化到数据库
            if intent in ["flight", "train", "concert", "car_rental", "tour_group", "insurance"]:
                self.memory.extract_entities(intent, query_str)
                self.memory.update_task_context({"type": intent, "query": query_str})

            # 从代理网络中获取对应的 agent 实例
            agent = self.agent_network.get_agent(agent_name)
            if agent is None:
                # agent 未注册或不可用时，返回友好提示
                logger.warning(f"未找到代理：{agent_name}")
                return f"抱歉，{agent_name} 暂时不可用，请稍后重试。"

            # 构建发送给 agent 的消息：只传最近2轮成功对话 + 当前查询（过滤错误消息）
            recent_msgs = self.messages[-5:-1] if len(self.messages) > 1 else []
            clean_msgs = [
                m for m in recent_msgs
                if not (isinstance(m.get('content', ''), str) and
                        ('error' in m['content'].lower()[:50] or
                         'HTTPConnectionPool' in m['content'] or
                         '暂时不可用' in m['content'] or
                         '查询失败' in m['content']))
            ]
            recent_history = "\n".join([
                f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:200]}"
                for m in clean_msgs[-4:]
            ]) if clean_msgs else ""
            agent_input = json.dumps({
                "history": recent_history,
                "query": query_str
            }, ensure_ascii=False)
            msg = Message(content=TextContent(text=agent_input), role=MessageRole.USER)
            task = Task(id="task-" + str(uuid.uuid4()), message=msg.to_dict())

            # 通过 A2A 协议异步发送任务给子代理
            # send_task_async 内部已使用 run_in_executor 处理阻塞调用，直接 await 即可
            try:
                # 调用A2AServer执行任务
                raw_response = await agent.send_task_async(task)


                logger.info(f"{agent_name} 原始响应: {raw_response}")

                # 解析 agent 返回结果
                if raw_response.status.state == 'completed' and raw_response.artifacts:
                    agent_result = raw_response.artifacts[0]['parts'][0]['text']
                elif raw_response.status.message:
                    msg = raw_response.status.message
                    if isinstance(msg, dict):
                        agent_result = msg.get('content', {}).get('text', str(msg))
                    else:
                        agent_result = str(msg)
                else:
                    agent_result = f"查询失败：未知错误"
            except Exception as e:
                # agent 服务不可用（如网络不通、服务崩溃），返回友好错误提示
                logger.error(f"{agent_name} 调用异常: {str(e)}")
                return f"抱歉，{agent_name} 服务暂时不可用，请稍后重试。"

            # 子 Agent 已自行格式化输出，直接返回，省去一次 LLM 总结调用（节省 3~5 秒）
            return agent_result

        # ========== 情况3：未识别的意图 ===========
        else:
            return "暂不支持此意图。"

    async def execute_step(self, step: dict, user_queries: dict) -> str:
        """
        执行单个计划步骤（供 ReAct 循环调用）

        这个方法负责将规划中的单个步骤转化为实际的执行：
        1. 从步骤中提取意图和查询内容
        2. 调用 _call_agent_intent 执行实际的查询

        参数：
            step (dict): 计划中的单个步骤，格式如：
                {"step": 1, "action": "查询天气", "intent": "weather", "depends_on": 0}
            user_queries (dict): 改写后的查询字典

        返回值：
            str: 该步骤的执行结果
        """
        intent = step.get("intent", "")
        # 优先从 user_queries 中获取改写后的查询，如果没有则用用户原始输入
        query_str = user_queries.get(intent, self.messages[-1]["content"] if self.messages else "")
        logger.info(f"执行步骤：{step.get('action', '')}，意图：{intent}")

        # 委托给统一的 agent 调用方法
        return await self._call_agent_intent(intent, query_str)

    async def react_loop(self, steps: list, user_queries: dict) -> str:
        """
        ReAct 循环：按规划步骤逐步执行，直接行动（Action），跳过 Thought 推理

        执行流程：
        1. 按依赖关系分组步骤
        2. 无依赖的步骤并行执行
        3. 记录每步结果（Observation）
        4. 最终汇总生成连贯回复

        优化：省略 Thought LLM 调用（plan 已确定动作，Thought 无额外决策价值），
        省去每步 1 次 LLM 调用，多步骤场景下显著减少响应时间。

        参数：
            steps (list): 计划步骤列表，如：
                [{"step": 1, "action": "...", "intent": "flight", "depends_on": 0},
                 {"step": 2, "action": "...", "intent": "weather", "depends_on": 0}]
            user_queries (dict): 改写后的查询字典

        返回值：
            str: 综合所有步骤结果的最终回复
        """
        observations = []     # 存储每个步骤的执行结果（Observation）
        step_results = []     # 存储每个步骤的详细结果（步骤号、描述、结果）

        # 按依赖关系分组：depends_on 值相同的步骤可以并行
        from collections import OrderedDict
        dep_groups = OrderedDict()
        for step in steps:
            dep = step.get("depends_on", 0)
            dep_groups.setdefault(dep, []).append(step)

        # 逐组执行，组内并行
        for dep_key, group_steps in dep_groups.items():
            # 组内步骤并行执行 Action
            if len(group_steps) > 1:
                logger.info(f"并行执行 {len(group_steps)} 个无依赖步骤")
                tasks = [self.execute_step(s, user_queries) for s in group_steps]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for step, result in zip(group_steps, results):
                    step_num = step.get("step", 0)
                    step_desc = step.get("description", step.get("action", ""))
                    if isinstance(result, Exception):
                        result = f"执行失败：{result}"
                    observations.append(result)
                    step_results.append({"step": step_num, "description": step_desc, "result": result})
                    logger.info(f"ReAct 步骤 {step_num} 结果: {result[:100]}...")
            else:
                # 单步骤，直接执行
                step = group_steps[0]
                step_num = step.get("step", 0)
                step_desc = step.get("description", step.get("action", ""))
                result = await self.execute_step(step, user_queries)
                observations.append(result)
                step_results.append({"step": step_num, "description": step_desc, "result": result})
                logger.info(f"ReAct 步骤 {step_num} 结果: {result[:100]}...")

        # ========== 所有步骤完成后的最终汇总 ==========
        if len(step_results) > 1:
            # 多个步骤：使用专门的汇总 prompt，将各步骤结果整合成一条连贯回复
            summary_chain = SmartVoyagePrompts.react_summary_prompt() | self.llm
            all_obs = "\n".join([f"步骤{s['step']} ({s['description']}): {s['result']}" for s in step_results])
            final_response = summary_chain.invoke({
                "query": self.messages[-1]["content"] if self.messages else "",
                "all_observations": all_obs
            }).content.strip()
        else:
            # 只有一个步骤：直接使用该步骤的结果
            final_response = observations[0] if observations else "暂无结果"

        return final_response

    async def chat(self, user_input: str) -> str:
        """
        处理用户输入的主方法 —— 整个对话流程的入口

        完整流程：
        1. 记录用户消息到记忆中
        2. 意图识别：判断用户想做什么
        3. 处理特殊情况（超出范围 / 需要追问）
        4. 任务规划：判断是简单任务还是复杂任务
        5. 执行：简单任务直接执行，复杂任务进入 ReAct 循环
        6. 记录回复到记忆中
        7. 返回最终回复

        参数：
            user_input (str): 用户的输入内容

        返回值：
            str: 助手的回复内容

        使用示例：
            service = ChatService()
            response = await service.chat("北京明天天气怎么样？")
        """
        try:
            # 将用户消息保存到记忆中（短期记忆、对话历史等）
            # user_input = "广州明天天气怎么样？如果不下雨，帮我预定明天到深圳的高铁"
            self.memory.add_message("user", user_input)
            self._add_message("user", user_input)

            # ========== 步骤1：意图识别 ==========
            intents, user_queries, follow_up_message = self.intent_agent(user_input)
            # intents: [weather, train]
            # user_queries: [查询广州明天天气, 查询明天广州到深圳的高铁]
            # follow_up_message：""，不需要追问

            # ========== 步骤2：处理特殊情况 ==========
            if "out_of_scope" in intents:
                # 用户输入超出系统能力范围，返回追问消息
                response = follow_up_message if follow_up_message else \
                    "我是SmartVoyage旅行助手，可以帮您查询天气、机票、火车票、演唱会门票、景点推荐、租车、旅游团和保险等。请问有什么可以帮您的？"
            elif follow_up_message != "":
                # 意图不明确，需要追问用户
                response = follow_up_message
            else:
                # ========== 步骤3：任务规划 ==========
                # 启发式判断：单意图或独立多意图跳过 planning_agent，省去1次LLM调用
                if self._should_skip_planning(intents):
                    plan = {"need_plan": False, "reason": "启发式判断：任务简单，可直接执行", "steps": []}
                    logger.info(f"跳过规划: {plan['reason']}")
                else:
                    plan = self.planning_agent(intents, user_queries)
                    # [1.查询广州明天天气；2.如果不下雨，查询明天广州到深圳的高铁]
                need_plan = plan.get("need_plan", False)
                logger.info(f"规划结果: need_plan={need_plan}, reason={plan.get('reason', '')}")

                if need_plan:
                    # ========== 复杂任务：进入 ReAct 循环 ==========
                    # 获取计划步骤，逐步执行
                    steps = plan.get("steps", [])
                    response = await self.react_loop(steps, user_queries)
                else:
                    # ========== 简单任务：多意图并行执行（asyncio.gather）==========
                    intent_queries = [
                        (intent, user_queries.get(intent, ""))
                        for intent in intents
                    ]
                    responses = await asyncio.gather(*[
                        self._call_agent_intent(intent, query_str)
                        for intent, query_str in intent_queries
                    ])

                    routed_agents = [
                        conf.intent[intent] for intent in intents
                        if intent != "attraction" and conf.intent.get(intent)
                    ]

                    # 将多个意图的回复用空行拼接成最终回复
                    response = "\n\n".join(responses)
                    if routed_agents:
                        logger.info(f"路由到代理：{routed_agents}")

            # 将助手回复也保存到记忆中
            self.memory.add_message("assistant", response)
            self._add_message("assistant", response)
            return response

        except json.JSONDecodeError as e:
            # 大模型返回的不是有效 JSON（虽然概率低，但偶尔会发生）
            logger.error(f"意图识别JSON解析失败")
            error_message = f"意图识别JSON解析失败：{str(e)}。请重试。"
            self.memory.add_message("assistant", error_message)
            self._add_message("assistant", error_message)
            return error_message
        except Exception as e:
            # 捕获其他所有异常（网络错误、服务不可用等）
            logger.error(f"处理异常: {str(e)}")
            error_message = f"处理失败：{str(e)}。请重试。"
            self.memory.add_message("assistant", error_message)
            self._add_message("assistant", error_message)
            return error_message

    def get_agent_cards(self) -> list:
        """
        获取所有代理的卡片信息

        Agent Card 是 A2A 协议中描述一个 agent 能力的方式，
        包含：名称、技能列表、描述、地址、状态等。

        返回值：
            list: 代理卡片列表，每个卡片格式为：
                {
                    "name": "WeatherAssistant",
                    "skills": ["天气查询: 根据城市查询天气"],
                    "description": "天气查询助手",
                    "url": "http://127.0.0.1:5005",
                    "status": "在线"
                }
        """
        cards = []
        for agent_name in self.agent_network.agents.keys():
            agent_card = self.agent_network.get_agent_card(agent_name)
            agent_url = self.agent_urls.get(agent_name, "未知地址")
            cards.append({
                "name": agent_name,
                # 将每个技能对象格式化为 "技能名: 技能描述" 的字符串
                "skills": [s.name + ": " + s.description for s in agent_card.skills],
                "description": agent_card.description,
                "url": agent_url,
                "status": "在线"
            })
        return cards

    def get_memory_state(self) -> dict:
        """
        获取当前记忆状态的摘要信息（用于调试和展示）

        返回值：
            dict: 包含以下信息：
                - short_term_messages: 最近5条对话消息
                - user_profile: 用户偏好
                - current_task: 当前任务上下文
                - entity_history: 最近5条实体记录
        """
        return {
            "short_term_messages": [
                {"role": "用户" if m["role"] == "user" else "助手", "content": m["content"], "timestamp": m["timestamp"]}
                for m in self.memory.short_term_messages[-5:]
            ],
            "user_profile": self.memory.user_profile,
            "current_task": self.memory.current_task,
            "entity_history": self.memory.entity_history[-5:]
        }

    def clear_memory(self):
        """
        清空所有记忆（短期记忆、用户偏好、任务上下文等）

        用户输入 'clear' 命令时调用，用于重置对话状态。
        memory.clear() 内部会自动清理数据库中的持久化数据。
        """
        self.memory.clear()
        self.messages.clear()
        self.conversation_history = ""

    async def chat_stream(self, user_input: str):
        """
        流式处理用户输入的主方法 —— 与普通 chat 方法流程相同，但最终回复通过 yield 逐字返回

        完整流程：
        1. 记录用户消息到记忆中
        2. 意图识别：判断用户想做什么
        3. 处理特殊情况（超出范围 / 需要追问）
        4. 任务规划：判断是简单任务还是复杂任务
        5. 执行：简单任务直接执行，复杂任务进入 ReAct 循环
        6. 最终回复通过 yield 逐字返回

        参数：
            user_input (str): 用户的输入内容

        Yields：
            str: 助手回复的文本片段
        """
        try:
            self.memory.add_message("user", user_input)
            self._add_message("user", user_input)

            intents, user_queries, follow_up_message = self.intent_agent(user_input)

            if "out_of_scope" in intents:
                response = follow_up_message if follow_up_message else \
                    "我是SmartVoyage旅行助手，可以帮您查询天气、机票、火车票、演唱会门票、景点推荐、租车、旅游团和保险等。请问有什么可以帮您的？"
                yield response
            elif follow_up_message != "":
                response = follow_up_message
                yield response
            else:
                if self._should_skip_planning(intents):
                    plan = {"need_plan": False, "reason": "启发式判断：任务简单，可直接执行", "steps": []}
                    logger.info(f"跳过规划(stream): {plan['reason']}")
                else:
                    plan = self.planning_agent(intents, user_queries)
                need_plan = plan.get("need_plan", False)
                logger.info(f"规划结果: need_plan={need_plan}, reason={plan.get('reason', '')}")

                if need_plan:
                    steps = plan.get("steps", [])
                    parts = []
                    async for chunk in self._react_loop_stream(steps, user_queries):
                        parts.append(chunk)
                        yield chunk
                    response = "".join(parts)
                else:
                    responses = []
                    for intent in intents:
                        logger.info(f"处理意图：{intent}")
                        query_str = user_queries.get(intent, "")
                        parts = []
                        async for chunk in self._call_agent_intent_stream(intent, query_str):
                            parts.append(chunk)
                            yield chunk
                        responses.append("".join(parts))

                    response = "\n\n".join(responses)

            self.memory.add_message("assistant", response)
            self._add_message("assistant", response)

        except json.JSONDecodeError as e:
            logger.error(f"意图识别JSON解析失败")
            error_message = f"意图识别JSON解析失败：{str(e)}。请重试。"
            self.memory.add_message("assistant", error_message)
            self._add_message("assistant", error_message)
            yield error_message
        except Exception as e:
            logger.error(f"处理异常: {str(e)}")
            error_message = f"处理失败：{str(e)}。请重试。"
            self.memory.add_message("assistant", error_message)
            self._add_message("assistant", error_message)
            yield error_message

    async def _call_agent_intent_stream(self, intent: str, query_str: str):
        """
        流式版本的 _call_agent_intent，仅在生成最终回复时使用流式输出
        调用方需要自行收集 yield 的片段拼接为完整回复
        """
        agent_name = conf.intent.get(intent)

        if intent == "attraction":
            chain = SmartVoyagePrompts.attraction_prompt() | self.llm
            async for chunk in chain.astream({"query": query_str}):
                yield chunk.content if hasattr(chunk, 'content') else str(chunk)

        elif agent_name:
            if intent in ["flight", "train", "concert", "car_rental", "tour_group", "insurance"]:
                self.memory.extract_entities(intent, query_str)
                self.memory.update_task_context({"type": intent, "query": query_str})

            agent = self.agent_network.get_agent(agent_name)
            if agent is None:
                logger.warning(f"未找到代理：{agent_name}")
                yield f"抱歉，{agent_name} 暂时不可用，请稍后重试。"
            else:
                # 只传最近2轮成功对话 + 当前查询（过滤掉错误消息，防止历史污染）
                recent_msgs = self.messages[-5:-1] if len(self.messages) > 1 else []
                clean_msgs = [
                    m for m in recent_msgs
                    if not (isinstance(m.get('content', ''), str) and
                            ('error' in m['content'].lower()[:50] or
                             'HTTPConnectionPool' in m['content'] or
                             '暂时不可用' in m['content'] or
                             '查询失败' in m['content']))
                ]
                chat_history = "\n".join([
                    f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:200]}"
                    for m in clean_msgs[-4:]
                ]) if clean_msgs else ""
                agent_input = chat_history + (f'\nUser: {query_str}' if chat_history else query_str)
                msg = Message(content=TextContent(text=agent_input), role=MessageRole.USER)
                task = Task(id="task-" + str(uuid.uuid4()), message=msg.to_dict())

                try:

                    # 使用A2AServer来执行任务
                    raw_response = await agent.send_task_async(task)

                    logger.info(f"{agent_name} 原始响应: {raw_response}")

                    if raw_response.status.state == 'completed' and raw_response.artifacts:
                        agent_result = raw_response.artifacts[0]['parts'][0]['text']
                    elif raw_response.status.message:
                        msg = raw_response.status.message
                        if isinstance(msg, dict):
                            agent_result = msg.get('content', {}).get('text', str(msg))
                        else:
                            agent_result = str(msg)
                    else:
                        agent_result = f"查询失败：未知错误"
                except Exception as e:
                    logger.error(f"{agent_name} 调用异常: {str(e)}")
                    agent_result = f"抱歉，{agent_name} 服务暂时不可用，请稍后重试。"

                # 子 Agent 已格式化输出，直接返回，省去 summarize LLM 调用
                yield agent_result

        else:
            yield "暂不支持此意图。"

    async def _react_loop_stream(self, steps: list, user_queries: dict):
        """
        流式版本的 ReAct 循环，直接行动（跳过 Thought 推理），最终汇总时流式输出

        优化：与 react_loop 一致，省略 Thought LLM 调用，减少每步 1 次 LLM 请求。
        调用方需要自行收集 yield 的片段拼接为完整回复。
        """
        observations = []
        step_results = []

        from collections import OrderedDict
        dep_groups = OrderedDict()
        for step in steps:
            dep = step.get("depends_on", 0)
            dep_groups.setdefault(dep, []).append(step)

        for dep_key, group_steps in dep_groups.items():
            if len(group_steps) > 1:
                logger.info(f"并行执行 {len(group_steps)} 个无依赖步骤")
                tasks = [self.execute_step(s, user_queries) for s in group_steps]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for step, result in zip(group_steps, results):
                    step_num = step.get("step", 0)
                    step_desc = step.get("description", step.get("action", ""))
                    if isinstance(result, Exception):
                        result = f"执行失败：{result}"
                    observations.append(result)
                    step_results.append({"step": step_num, "description": step_desc, "result": result})
                    logger.info(f"ReAct 步骤 {step_num} 结果: {result[:100]}...")
            else:
                step = group_steps[0]
                step_num = step.get("step", 0)
                step_desc = step.get("description", step.get("action", ""))
                result = await self.execute_step(step, user_queries)
                observations.append(result)
                step_results.append({"step": step_num, "description": step_desc, "result": result})
                logger.info(f"ReAct 步骤 {step_num} 结果: {result[:100]}...")

        if len(step_results) > 1:
            summary_chain = SmartVoyagePrompts.react_summary_prompt() | self.llm
            all_obs = "\n".join([f"步骤{s['step']} ({s['description']}): {s['result']}" for s in step_results])
            async for chunk in summary_chain.astream({
                "query": self.messages[-1]["content"] if self.messages else "",
                "all_observations": all_obs
            }):
                yield chunk.content if hasattr(chunk, 'content') else str(chunk)
        else:
            yield observations[0] if observations else "暂无结果"
