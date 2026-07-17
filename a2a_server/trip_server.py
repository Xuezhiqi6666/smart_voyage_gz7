"""
需求：实现基于A2A的行程管家服务器，处理用户的租车、旅游团、保险查询及预订请求

架构说明：
    本服务器是 SmartVoyage 系统中的又一个子代理（Sub-Agent），负责处理所有行程相关任务。
    它运行在独立的进程中（localhost:5007），通过 A2A（Agent2Agent）协议与主助手通信。

    与 ticket_server.py 和 weather_server.py 保持一致的架构模式：
    - 使用 LangChain Agent + MCP Tools 模式
    - LLM 从自然语言中提取参数（城市、日期、类型等）
    - 调用 MCP Server 的参数化工具（端口 8003）
    - MCP Server 返回结果后，LLM 将结果格式化为友好的中文回复

    工作流程：
    1. 主助手通过 A2A 协议向本服务器发送任务（Task）
    2. 本服务器收到任务后，提取用户的自然语言查询
    3. 使用 LangChain Agent 处理查询：
       a. LLM 分析用户输入，决定调用哪个 MCP 工具
       b. LangChain 自动调用 MCP Server 的工具（端口 8003）
       c. 工具返回结果后，LLM 将结果格式化为友好的中文回复
    4. 将结果返回给主助手

    MCP Server（端口 8003）提供的工具包括：
    - 租车查询工具、旅游团查询工具、保险查询工具
    - 租车预订工具、旅游团报名工具、保险购买工具

架构图：

+─────────────────────────────────────────────────────────────────────────────+
│                          SmartVoyage 主助手 (Main Agent)                     │
│                        通过 A2A 协议分派任务到子代理                           │
+──────────────────────────────────┬──────────────────────────────────────────+
                                   │ A2A 协议 (Task)
                                   ▼
+─────────────────────────────────────────────────────────────────────────────+
│                        TripQueryServer (A2AServer)  :5007                    │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  AgentCard: TripAssistant                                              │  │
│  │  Skills: query/order car rental, tour group, insurance                │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                     │                                       │
│  ┌──────────────────────────────────▼──────────────────────────────────┐   │
│  │  handle_task(task)                                                  │   │
│  │  1. 提取用户自然语言查询                                              │   │
│  │  2. asyncio.run(query_trip(conversation))                           │   │
│  │  3. 设置 TaskStatus: COMPLETED / INPUT_REQUIRED / FAILED             │   │
│  └──────────────────────────────────┬──────────────────────────────────┘   │
│                                     │                                       │
│  ┌──────────────────────────────────▼──────────────────────────────────┐   │
│  │  query_trip(conversation)                                            │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │  ChatOpenAI (LLM) + ChatPromptTemplate                          │ │   │
│  │  │  create_tool_calling_agent() → AgentExecutor                     │ │   │
│  │  │  ainvoke({"input": conversation, "current_date": ...})          │ │   │
│  │  └────────────────────────┬───────────────────────────────────────┘ │   │
│  │                           │ 选择工具 & 参数                          │   │
│  │  ┌────────────────────────▼───────────────────────────────────────┐ │   │
│  │  │  to_structured_langchain_tool(MCP_URL)                           │ │   │
│  │  │  MCP Server → LangChain Tools 桥接                              │ │   │
│  │  └───────────────────────────────────────────────────────────────┘ │   │
│  └───────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────┬────────────────────────────────────────┘
                                   │ MCP 协议
                                   ▼
+─────────────────────────────────────────────────────────────────────────────+
│                      MCP Server  :8003                                      │
│                                                                             │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                     │
│   │ 租车工具      │  │ 旅游团工具    │  │ 保险工具      │                     │
│   │ - query_car  │  │ - query_tour │  │ - query_ins  │                     │
│   │ - order_car  │  │ - order_tour │  │ - order_ins  │                     │
│   └──────────────┘  └──────────────┘  └──────────────┘                     │
│                                                                             │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │  数据层 (Database / 向量库)                                           │    │
│   │  租车数据 / 旅游团数据 / 保险数据                                    │    │
│   └───────────────────────────────────────────────────────────────────┘    │
+─────────────────────────────────────────────────────────────────────────────+

请求/响应流向：
    用户 → 主助手 → [A2A] → TripQueryServer.handle_task()
                                  ↓
                          query_trip() → LangChain Agent (LLM)
                                  ↓
                          MCP Tools → MCP Server :8003
                                  ↓
                          MCP Server 返回工具结果
                                  ↓
                          LLM 格式化为中文回复
                                  ↓
                          handle_task() 设置 TaskStatus
                                  ↓
                          主助手 → 用户
"""

# ==================== 导入依赖 ====================
import asyncio  # 异步 IO 库
import requests as http_requests  # HTTP 请求，用于调用 MCP 工具

from python_a2a import MCPClient
from python_a2a import A2AServer, run_server, AgentCard, AgentSkill, TaskStatus, TaskState
# A2AServer: A2A 服务器基类
# run_server: 启动服务器的函数
# AgentCard: 代理卡片
# AgentSkill: 代理技能
# TaskStatus: 任务状态对象
# TaskState: 任务状态枚举（COMPLETED / FAILED / INPUT_REQUIRED）

from langchain_openai import ChatOpenAI  # LangChain 的大模型接口
from langchain_core.prompts import ChatPromptTemplate  # 提示模板
from langchain_core.tools import StructuredTool  # 结构化 LangChain 工具
from langchain.agents import create_tool_calling_agent, AgentExecutor
# create_tool_calling_agent: 创建基于工具调用的 Agent
# AgentExecutor: Agent 执行器，负责运行 Agent 循环

from datetime import datetime  # 时间处理
import pytz  # 时区库
from typing import Optional  # 可选类型标注
from pydantic import create_model  # 动态创建 Pydantic 模型
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from utils.mcp_tools import to_structured_langchain_tool
from config import Config  # 项目配置
from create_logger import logger  # 日志模块

conf = Config()  # 全局配置实例

# ==================== 初始化大模型 ====================
# 创建一个 LLM 实例，供 Agent 使用
llm = ChatOpenAI(
    model=conf.model_name,
    base_url=conf.base_url,
    api_key=conf.api_key,
    temperature=conf.temperature
)

# ==================== MCP 客户端配置 ====================
# MCP Server 运行在 8003 端口，提供行程相关的工具（查询 + 预订）
MCP_URL = "http://127.0.0.1:8003"

# 缓存 MCP 工具，避免每次请求都发 HTTP GET 获取 Schema
_cached_tools = None

def get_mcp_tools():
    """获取并缓存 MCP 工具列表"""
    global _cached_tools
    if _cached_tools is None:
        _cached_tools = to_structured_langchain_tool(MCP_URL)
    return _cached_tools

# ==================== 行程查询函数 ====================
async def query_trip(conversation: str) -> dict:
    """
    通过 LangChain Agent + MCP Tools 执行行程查询或预订

    与 ticket_server.py 的 query_tickets 使用完全相同的架构模式：
    1. 连接 MCP Server 加载所有可用的工具
    2. 创建一个 LangChain Agent，让它自主选择调用哪个工具
    3. AgentExecutor 负责执行 Agent 循环

    参数：
        conversation (str): 用户的查询内容，例如：
            "我想租一辆车，北京，明天"
            "有没有去丽江的旅游团"
            "买一份旅行保险"
            "帮我报个北京三日游的团"

    返回值：
        dict: 查询结果，格式为：
            - {"status": "success", "message": "格式化后的查询结果"}
            - {"status": "error", "message": "错误信息"}
    """
    try:
        tools = get_mcp_tools()
        # 定义 Agent 的系统 Prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个行程管家助手，能够调用工具来完成租车查询/预定、旅游团查询/报名、保险查询/购买。
你需要仔细分析用户的问题，从问题中提取工具需要的参数，然后调用对应的工具。
如果用户提供的信息不足以提取到调用工具的所有必要参数，则向用户追问，以获取该信息。不能自己编撰参数。
注意：
- 用户可能使用相对时间，如"明天"、"后天"、"今天"、"下周六"等，请根据当前日期转换为具体日期（YYYY-MM-DD格式）。
- 查询租车时，必需参数：取车城市、还车城市、日期；如果缺少这三个必需参数，需要追问。车型为可选参数，用户未指定时直接查询全部车型，不要追问车型。
- 查询旅游团时，必需参数：查询描述（如"想看雪山的地方"、"有什么旅游团"）；城市为可选过滤参数，用户未指定城市时不传city参数，不要追问城市。
- 查询保险时，保险类型为可选参数（综合型/意外型/医疗型/境外型），用户未指定时直接查询全部类型，不要追问保险类型，不传insurance_type参数即可。
- 预定租车时，需要日期、车型和数量；如果缺少任一参数，需要追问。
- 报名旅游团时，需要出发日期、旅游团名称和人数；如果缺少任一参数，需要追问。
- 购买保险时，需要保险类型、生效日期和份数；如果缺少任一参数，需要追问。
- 【重要】当你需要向用户追问时，回复必须以"[追问]"开头，例如："[追问]请问您想在哪一天取车？"
查询到结果后，请用清晰的中文格式化输出，格式如下：
- 租车：取车城市 到 还车城市 日期: 车型XX，公司XX，价格XX元/天，余量XX
- 旅游团：城市 团名: 天数XX天，价格XX元，余位XX/总数XX，亮点XX，出发日期列表
- 保险：保险类型: 名称XX，价格XX元/份，保障范围XX，保险公司XX
预定成功后请直接告知用户预定结果。
如果未查到数据，请回复"未找到相关行程数据，请确认或修改查询条件。"
当前日期是{current_date}。"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        # 创建基于工具调用的 Agent
        agent = create_tool_calling_agent(llm, tools, prompt)

        # 创建 Agent 执行器
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        # 获取当前日期，注入到 Prompt 中
        current_date = datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d')

        # 执行 Agent
        response = await agent_executor.ainvoke({
            "input": conversation,
            "current_date": current_date
        })

        return {"status": "success", "message": response["output"]}

    except ExceptionGroup as eg:
        # MCP 工具内部使用 anyio.create_task_group()，失败时抛出 ExceptionGroup
        first_exc = eg.exceptions[0] if eg.exceptions else eg
        logger.error(f"行程 MCP 查询出错：{first_exc}")
        return {"status": "error", "message": f"行程 MCP 查询出错：{first_exc}"}
    except BaseException as e:
        # 兜底捕获（包括 ExceptionGroup 等 BaseException 子类）
        logger.error(f"行程 MCP 查询出错：{str(e)}")
        return {"status": "error", "message": f"行程 MCP 查询出错：{str(e)}"}


# ==================== Agent Card（代理卡片） ====================
# 描述行程管家代理的能力、技能等信息
agent_card = AgentCard(
    name="TripAssistant",  # 代理名称
    description="基于 LangChain 提供行程管家服务的统一助手，支持租车、旅游团、保险的查询与预订",  # 代理描述
    url="http://localhost:5007",  # 代理的访问地址
    version="2.0.0",  # 版本号
    capabilities={"streaming": True, "memory": True},  # 支持的能力
    skills=[
        # 技能1：租车查询
        AgentSkill(
            name="query car rental",
            description="查询租车信息，支持指定取车城市、还车城市、日期和车型",
            examples=["租车 北京 上海 2025-08-01", "北京租一辆SUV 明天"]
        ),
        # 技能2：旅游团查询（语义搜索）
        AgentSkill(
            name="query tour group",
            description="通过语义搜索查询旅游团信息，支持自然语言描述需求（如想看雪山的地方）和可选的城市过滤",
            examples=["想看雪山的地方", "适合亲子游的短途旅行", "美食之旅", "北京的旅游团"]
        ),
        # 技能3：保险查询
        AgentSkill(
            name="query insurance",
            description="查询旅行保险产品，支持指定保险类型和日期",
            examples=["旅行保险 2025-08-01", "买一份综合型旅行保险"]
        ),
        # 技能4：租车预订
        AgentSkill(
            name="order car rental",
            description="根据日期、车型和数量预订租车服务",
            examples=["预订租车 2025-08-01 SUV 1辆"]
        ),
        # 技能5：旅游团报名
        AgentSkill(
            name="order tour group",
            description="根据日期、团名和人数报名旅游团",
            examples=["报名丽江三日游 2025-08-01 2人"]
        ),
        # 技能6：保险购买
        AgentSkill(
            name="order insurance",
            description="根据保险类型、日期和份数购买旅行保险",
            examples=["购买综合型保险 2025-08-01 1份"]
        )
    ]
)


# ==================== 行程查询服务器类 ====================
class TripServer(A2AServer):
    """
    行程管家 A2A 服务器

    这个类继承自 A2AServer，负责处理所有行程相关的任务。
    工作流程与 TicketQueryServer 完全一致：
    1. 接收来自主助手的任务（Task）
    2. 从任务中提取用户的查询内容
    3. 调用 query_trip 函数执行查询
    4. 将查询结果设置到任务状态中，返回给主助手
    """

    def __init__(self):
        """
        初始化行程查询服务器
        """
        super().__init__(agent_card=agent_card)  # 调用父类初始化，注册 Agent Card

    def handle_task(self, task):
        """
        处理来自 A2A 客户端的任务 —— 本服务器的核心方法

        完整流程：
        1. 提取输入：从任务消息中获取用户的查询内容
        2. 调用 MCP 查询：通过 query_trip 函数执行查询
        3. 根据查询结果设置任务状态：
           - 成功：COMPLETED 状态，结果放在 task.artifacts
           - 需要输入：INPUT_REQUIRED 状态，追问信息放在 task.status.message
           - 失败：FAILED 状态，错误信息放在 task.status.message

        参数：
            task: A2A 任务对象

        返回值：
            task: 处理后的任务对象（已设置状态和结果）
        """
        # ========== 步骤1：提取输入 ==========
        content = (task.message or {}).get("content", {})
        conversation = content.get("text", "") if isinstance(content, dict) else ""
        logger.info(f"用户查询: {conversation}")

        try:
            # ========== 步骤2：调用 MCP 查询 ==========
            trip_result = asyncio.run(query_trip(conversation))
            logger.info(f"MCP 查询返回: {trip_result}")

            # ========== 步骤3：根据结果设置任务状态 ==========
            if trip_result.get("status") == "success":
                result_text = trip_result.get("message", "")

                # 检查是否是追问消息（LLM 发现信息不足时会追问）
                # 判断优先级：[追问] 前缀（Prompt 约定）> 关键词兜底
                # 注意：LLM 格式化输出中可能包含"请告诉我"等礼貌用语，不能仅凭关键词判断
                # 如果输出中包含实际数据（价格、车型、团名等），说明是查询结果而非追问
                has_data = (
                    "¥" in result_text or "￥" in result_text
                    or "元/天" in result_text
                    or "元/份" in result_text
                    or "余量" in result_text
                    or "余位" in result_text
                    or "天数" in result_text
                    or "保障" in result_text
                    or "车型" in result_text
                    or "旅行社" in result_text
                    or "评分" in result_text
                    or "亮点" in result_text
                    or "相似度" in result_text
                )
                is_followup = (
                    result_text.strip().startswith("[追问]")  # Prompt 约定的追问标记（最高优先级）
                    or (not has_data and (
                        "请提供" in result_text
                        or "请确认" in result_text
                        or "请告诉我" in result_text
                        or "请问您" in result_text
                        or "哪一天" in result_text
                        or "哪个城市" in result_text
                        or "哪种车型" in result_text
                        or "哪种保险" in result_text
                    ))
                )
                # ① 缺少参数，需要追问
                if is_followup:
                    clean_text = result_text.replace("[追问]", "").strip()
                    task.status = TaskStatus(
                        state=TaskState.INPUT_REQUIRED,
                        message={"role": "agent", "content": {"text": clean_text}}
                    )
                else:
                    # ② 查询成功
                    task.artifacts = [{"parts": [{"type": "text", "text": result_text}]}]
                    task.status = TaskStatus(state=TaskState.COMPLETED)

            elif trip_result.get("status") == "error":
                task.status = TaskStatus(
                    state=TaskState.FAILED,
                    message={"role": "agent", "content": {"text": trip_result.get("message", "查询失败，请重试。")}}
                )
            else:
                task.status = TaskStatus(
                    state=TaskState.FAILED,
                    message={"role": "agent", "content": {"text": "查询失败，请重试或提供更多细节。"}}
                )

            return task

        except Exception as e:
            logger.error(f"查询失败: {str(e)}")
            task.status = TaskStatus(
                state=TaskState.FAILED,
                message={"role": "agent",
                         "content": {"text": f"查询失败: {str(e)} 请重试或提供更多细节。"}}
            )
            return task


# ==================== 主函数：启动服务器 ====================
if __name__ == "__main__":
    # 创建行程查询服务器实例
    trip_server = TripServer()

    # 打印服务器信息
    print("\n=== 服务器信息 ===")
    print(f"名称: {trip_server.agent_card.name}")
    print(f"描述: {trip_server.agent_card.description}")
    print("\n技能:")
    for skill in trip_server.agent_card.skills:
        print(f"- {skill.name}: {skill.description}")

    # 启动 A2A 服务器，监听 5007 端口
    run_server(trip_server, host="127.0.0.1", port=5007)
