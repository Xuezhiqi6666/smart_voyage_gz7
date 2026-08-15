# SmartVoyage - 智能旅行助手

基于 **A2A（Agent-to-Agent）协议 + MCP（Model Context Protocol）工具服务** 的多代理旅行助手系统，支持天气查询、票务预订、行程管理、景点推荐等自然语言交互。

## 架构概览

```
用户 (浏览器/CLI)
    │
    ▼
┌─────────────────────────────────────────────┐
│  api_server.py (FastAPI :8080)              │
│  chat_service.py (意图识别 → 路由 → 执行)    │
└─────────────────┬───────────────────────────┘
                  │ A2A 协议
    ┌─────────────┼─────────────┐
    ▼             ▼             ▼
┌────────┐  ┌────────┐  ┌────────┐
│Weather │  │Ticket  │  │Trip    │   A2A 代理层
│:5005   │  │:5006   │  │:5007   │   (LangChain 1.x create_agent)
└───┬────┘  └───┬────┘  └───┬────┘
    │           │           │
    ▼           ▼           ▼
┌────────┐  ┌────────┐  ┌────────┐
│MCP     │  │MCP     │  │MCP     │   MCP 工具层
│:8002   │  │:8001   │  │:8003   │   (数据库/API查询)
└────────┘  └────────┘  └────────┘
    │           │           │
    ▼           ▼           ▼
和风天气API   MySQL       MySQL+Milvus
```

## 核心流程

```
用户输入 → 意图识别(LLM) → 启发式路由 → 并行执行/ReAct循环 → 返回回复
```

- **意图识别**：LLM 分析用户输入，识别意图（weather/flight/train/concert/car_rental/tour_group/insurance/attraction）
- **启发式路由**：单意图或独立多意图跳过规划，直接并行执行
- **ReAct 循环**：复杂任务按依赖关系分组，组内并行执行，最终 LLM 汇总

## 项目结构

```
smart_voyage_gz7/
├── api_server.py              # FastAPI 主服务，自动启动所有子服务
├── chat_service.py            # 核心对话服务（意图识别 + 路由 + 执行）
├── config.py                  # 全局配置（LLM/DB/意图映射）
├── main_prompts.py            # Prompt 模板管理
├── memory.py                  # 记忆管理（短期对话/用户偏好/实体历史）
├── create_logger.py           # 日志模块
│
├── a2a_server/                # A2A 代理层（LangChain Agent + MCP Tools）
│   ├── weather_server.py      # 天气代理 :5005
│   ├── ticket_server.py       # 票务代理 :5006
│   └── trip_server.py         # 行程代理 :5007
│
├── mcp_server/                # MCP 工具层（数据查询接口）
│   ├── mcp_weather_server.py  # 天气工具 :8002（和风天气API/MySQL）
│   ├── mcp_ticket_server.py   # 票务工具 :8001（火车票/机票/演唱会）
│   └── mcp_trip_server.py     # 行程工具 :8003（租车/旅游团/保险）
│
├── utils/
│   ├── mcp_tools.py           # MCP → LangChain StructuredTool 桥接
│   ├── format.py              # JSON 编码器（date/Decimal）
│   └── spider_weather.py      # 天气数据爬虫
│
├── sql/
│   ├── create_all_tables.sql  # 建表语句
│   ├── insert_data.sql        # 测试数据
│   └── execute_sql.py         # SQL 执行脚本
│
├── milvus_redis_mysql/
│   ├── docker-compose.yml     # MySQL + Milvus + MinIO 容器编排
│   └── volumes/               # 数据持久化目录
│
└── static/
    └── index.html             # 前端页面
```

## 环境依赖

- Python 3.11+
- Docker（MySQL + Milvus）
- 通义千问 API Key（`.env` 中配置 `DASHSCOPE_API_KEY`）

### 安装

```bash
# 1. 启动基础设施
cd milvus_redis_mysql
docker-compose up -d

# 2. 初始化数据库
python sql/execute_sql.py sql/create_all_tables.sql
python sql/execute_sql.py sql/insert_data.sql

# 3. 配置环境变量
# 在 .env 中设置 DASHSCOPE_API_KEY=你的API密钥
```

## 启动

```bash
# 一键启动（自动拉起 6 个子服务 + 主 API）
python api_server.py
```

访问 `http://127.0.0.1:8080` 即可使用。

### 手动启动各服务（调试用）

```bash
# 终端1-3：MCP 工具层
python mcp_server/mcp_ticket_server.py   # :8001
python mcp_server/mcp_weather_server.py  # :8002
python mcp_server/mcp_trip_server.py     # :8003

# 终端4-6：A2A 代理层
python a2a_server/weather_server.py      # :5005
python a2a_server/ticket_server.py       # :5006
python a2a_server/trip_server.py         # :5007

# 终端7：主服务
python api_server.py                     # :8080
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/chat` | 发送消息，获取回复 |
| POST | `/api/chat/stream` | SSE 流式回复 |
| GET | `/api/agents` | 获取代理卡片列表 |
| GET | `/api/memory` | 获取记忆状态 |
| POST | `/api/memory/clear` | 清空记忆 |
| POST | `/api/memory/profile` | 更新用户偏好 |

### 请求示例

```bash
# 对话
curl -X POST http://127.0.0.1:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "8月1日从北京到成都的火车票"}'

# 设置偏好
curl -X POST http://127.0.0.1:8080/api/memory/profile \
  -H "Content-Type: application/json" \
  -d '{"profile": {"seat_type": "二等座"}}'
```

## 技术栈

| 组件 | 技术 |
|------|------|
| LLM | 通义千问 qwen3.7-max（DashScope API） |
| 代理框架 | LangChain 1.x `create_agent`（底层 LangGraph） |
| 代理通信 | python_a2a（A2A 协议） |
| 工具服务 | FastMCP（MCP 协议） |
| 数据库 | MySQL 8.0（Docker，端口 3307） |
| 向量库 | Milvus（旅游团语义搜索） |
| 天气数据 | 和风天气 API |
| Web 框架 | FastAPI + Uvicorn |

## 常见面试题

### Q1：请介绍项目的整体架构

**A**：SmartVoyage 采用 **三层架构**：

```
用户 → API 层（FastAPI :8080）→ A2A 代理层（LLM + 工具调用）→ MCP 工具层（数据库/API 查询）
```

- **API 层**：`api_server.py` + `chat_service.py`，负责接收用户输入、意图识别、任务规划、路由分发
- **A2A 代理层**：3 个 Agent（Weather/Ticket/Trip），每个 Agent 内部用 **LangChain 1.x `create_agent`**（底层 LangGraph 状态图）自主决策调用哪些工具
- **MCP 工具层**：3 个 MCP Server，封装具体的数据库查询和 API 调用，通过 REST 接口暴露工具 Schema

### Q2：什么是 A2A 协议？和 MCP 有什么区别？

**A**：
- **A2A（Agent-to-Agent）**：代理间通信协议，Agent 之间通过 HTTP POST（`/tasks/send`）传递 Task 对象，包含 Message、Status、Artifacts 等。用于 **Agent 之间的任务分发和协作**
- **MCP（Model Context Protocol）**：工具服务协议，Server 通过 REST 接口（`/tools`、`/tools/{name}`）暴露工具 Schema 和执行入口。用于 **Agent 调用外部工具和数据源**

简单说：**A2A 管 Agent 间通信，MCP 管 Agent 与工具的交互**。

### Q3：为什么用 A2A 协议，而不用 LangGraph 编排？用 LangGraph 怎么实现？

**A**：先澄清：**两者不是二选一，项目实际都用了**。A2A 是代理间**通信协议**（传输层），LangGraph 是进程内**编排框架**（执行层）。每个 A2A 子代理内部的 ReAct 循环就是用 LangChain 1.x `create_agent`（底层即 LangGraph 状态图，`recursion_limit=15`）实现的。

**为什么 Agent 间通信用 A2A**：

| 原因 | 说明 |
|------|------|
| 进程隔离 | 3 个子代理独立进程（:5005/:5006/:5007），故障域小，一个挂了不影响其他，可单独重启/排查 |
| 协议标准 | AgentCard 自描述能力，子代理可用任意框架/语言实现，主助手零改动；LangGraph 节点共享 TypedDict State，跨框架复用难 |
| 独立扩容 | 热点代理（如票务）只需把 AgentNetwork 的 URL 指向负载均衡，单独扩容 |
| 代价可接受 | LLM 单次 ~14s，HTTP 开销可忽略；启发式路由后大多数请求只有 1 次 A2A 调用 |

代价：HTTP 序列化开销 + 6 个子服务的运维复杂度。若低延迟单进程场景，全收进一个 LangGraph 图更简单。

**用 LangGraph 的实现思路**（只换编排层，MCP 工具层和子代理的 `create_agent` 原样复用）：

```python
class VoyageState(TypedDict):
    intents: list
    user_queries: dict
    observations: Annotated[list, add]   # reducer：并行分支结果自动合并

# Send API 扇出并行分支，替代原方案的 asyncio.gather
def route(state):
    if should_skip_planning(state["intents"]):
        return [Send("execute", {"intent": i, "query": state["user_queries"][i]})
                for i in state["intents"]]
    return "planner"

graph.add_node("intent", intent_node)            # 意图识别节点
graph.add_conditional_edges("intent", route)     # 启发式路由 = 条件边
app = graph.compile(checkpointer=MemorySaver())  # thread_id 区分会话
```

| A2A 方案（本项目） | LangGraph 方案 |
|------|------|
| `asyncio.gather` 并行多意图 | `Send` API + reducer 合并 |
| 自定义 `[追问]` / `INPUT_REQUIRED` | `interrupt()` 原生中断恢复 |
| 自研 `memory.py` 短期记忆 | Checkpointer（`thread_id` 区分会话） |
| HTTP 调子代理 | 进程内节点调用，LangSmith 全程可观测 |

**一句话总结**：进程间协作用 A2A，进程内编排用 LangGraph；若是单进程场景，用 LangGraph Supervisor 图替换 A2A 通信层即可，MCP 工具层和子代理内部实现完全复用。

**追问：有没有做分布式部署？主流的 A2A 项目怎么部署？**

**A**：诚实说，目前是**单机多进程**部署——`api_server.py` 用 subprocess 拉起 6 个子服务，均绑定 127.0.0.1。但架构为分布式预留：Agent 间全部通过 HTTP + URL 寻址，业务代码无进程内共享状态。切分布式只需 3 步，业务代码零改动：

1. 监听地址改 `run_server(host="0.0.0.0")`（127.0.0.1 其他机器无法访问）
2. Agent URL 从硬编码外置到环境变量/配置中心
3. 每个 Agent 容器化部署到 K8s，主服务 URL 指向集群内 DNS/负载均衡地址

主流工程做法（A2A 为 Google 2025 年开源并捐赠 Linux 基金会的协议，设计理念是 **Agent-as-a-Service**）：

| 实践 | 说明 |
|------|------|
| K8s 微服务 | 每个 Agent = Deployment + Service，HPA 对**单个 Agent** 独立扩缩容（LangGraph Platform 是整个图当一个服务部署，粒度更粗——这就是"独立扩容"优势的落地方式） |
| 服务发现 | 官方协议通过 `/.well-known/agent-card.json` 暴露能力卡片；本项目用社区库 `python_a2a`，AgentCard 为手动注册 |
| 网关统一入口 | 网关负责认证（OAuth2/API Key）、限流、路由，Agent 服务不直接暴露公网 |
| 可观测性 | OpenTelemetry 串联 主Agent → 子Agent → MCP 调用链为分布式 trace |
| 跨组织联邦 | A2A 独有价值：不同公司的 Agent 交换 AgentCard 即可互通，框架私有编排（如 LangGraph）做不到 |

### Q4：意图识别是怎么做的？为什么不用传统的分类模型？

**A**：用 **LLM 做意图识别**，Prompt 要求 LLM 返回 JSON 格式的结构化输出：

```json
{"intents": ["train"], "user_queries": {"train": "8月1日北京到成都"}, "follow_up_message": ""}
```

优势：
1. **零样本泛化**：新增意图只需修改 Prompt，无需重新训练模型
2. **同时提取参数**：LLM 在识别意图的同时提取查询参数（城市、日期等）
3. **支持追问**：信息不足时 LLM 生成 `follow_up_message` 直接追问用户

### Q5：什么是启发式路由？为什么要跳过规划？

**A**：启发式路由是 `_should_skip_planning()` 方法，对**单意图或独立多意图**直接并行执行，跳过 Planning Agent 的 LLM 调用。

原因：任务规划需要额外一次 LLM 调用（~15 秒），但大多数查询（如"查天气"、"查火车票"）是简单独立的，无需规划。启发式路由**节省一次 LLM 调用**，将响应时间从 ~40 秒降到 ~20 秒。

只有**存在依赖关系的复杂任务**（如"查成都天气，再根据天气推荐景点"）才走 Planning Agent → ReAct 循环。

### Q6：ReAct 循环是怎么实现的？

**A**：ReAct = **Re**asoning + **Act**ing。本项目有两层 ReAct：

**子代理层（A2A Server 内部）**：使用 LangChain 1.x 的 `create_agent`（底层是 LangGraph 的 `CompiledStateGraph`），自动实现 tool calling 循环：
```
model 节点 → 判断是否调用工具 → tools 节点执行工具 → 回到 model → ...直到 LLM 不再调用工具
```
通过 `recursion_limit` 控制最大迭代轮数，防止死循环。

**主代理层（ChatService）**：
1. **Planning Agent** 将复杂任务拆分为带依赖关系的步骤（`steps`）
2. 按 `depends_on` 分组，**同组步骤并行执行**（`asyncio.gather`）
3. 每组执行完后收集 `observations`，作为下一组的上下文
4. 所有步骤完成后，LLM 汇总所有 observations 生成最终回复

### Q7：LangChain Agent 的 Tool Calling 是怎么工作的？

**A**：LangChain 1.x 使用 `create_agent`（底层是 LangGraph 状态图），工作流程：

1. LLM 接收用户输入（messages 格式）+ 工具列表（名称、描述、参数 Schema）
2. LLM 判断需要调用哪个工具，输出 **tool_call**（工具名 + 参数）
3. LangGraph 的 `tools` 节点自动执行工具调用，将结果作为 `ToolMessage` 返回
4. `model` 节点根据 tool result 生成最终回复（或继续调用其他工具）
5. 通过 `recursion_limit` 控制最大迭代轮数，防止死循环

```python
# LangChain 1.x 写法
from langchain.agents import create_agent
agent = create_agent(llm, tools, system_prompt="...")
result = agent.invoke({"messages": [("human", query)]}, config={"recursion_limit": 15})
output = result["messages"][-1].content
```

### Q8：MCP 工具是怎么集成到 LangChain Agent 的？

**A**：通过 `to_structured_langchain_tool()` 桥接函数：

1. HTTP GET `/tools` 获取 MCP Server 的工具列表和 JSON Schema
2. 将每个 MCP 工具转换为 LangChain 的 `StructuredTool`
3. 工具的 `_func` 封装为 HTTP POST 到 `/tools/{tool_name}` 执行查询
4. 转换后的工具传给 `create_agent` 使用

注意：带可选参数的 MCP 工具需要自定义转换（`to_structured_langchain_tool`），`python_a2a` 库自带的 `to_langchain_tool` 不支持可选参数。

### Q9：项目的记忆系统是怎么设计的？

**A**：三层记忆，MySQL 持久化：

| 记忆类型 | 存储 | 用途 |
|---------|------|------|
| 短期对话 | `short_term_messages` 表 | 多轮对话上下文传递（如"那边天气"→ 从上文推断城市） |
| 用户偏好 | `user_profiles` 表 | UPSERT 模式，如 `seat_type: 二等座` 自动应用到后续查询 |
| 实体历史 | 内存字典 | 提取查询中的城市、日期等实体，辅助意图理解 |

### Q10：SSE 流式输出是怎么实现的？遇到过什么问题？

**A**：前端通过 `POST /api/chat/stream` 发送请求，后端用 FastAPI 的 `StreamingResponse` + SSE 协议逐块返回。

**遇到的问题**：`chat_stream` 内部的意图识别和 A2A 调用是同步阻塞的（~20 秒），期间无数据发送，导致客户端 Read Timeout。

**解决方案**：Queue + 后台 Producer 模式：
- Producer 在后台 Task 中运行 `chat_stream`，将结果放入 `asyncio.Queue`
- 主循环从 Queue 取数据，取不到就每 10 秒发送 `: heartbeat` 心跳保活
- 客户端收到心跳（SSE 注释行，自动忽略）保持连接不断

### Q11：项目中遇到过哪些性能瓶颈？怎么优化的？

**A**：

| 瓶颈 | 原因 | 优化 |
|------|------|------|
| LLM 调用慢 | 单次 LLM 调用 ~14 秒 | 启发式路由跳过不必要的 Planning LLM 调用 |
| 多意图串行 | 顺序执行多个意图 | `asyncio.gather` 并行执行独立意图 |
| A2A 调用阻塞 | `send_task_async` 实际是阻塞方法 | `run_in_executor` + `asyncio.run()` 包装 |
| 子进程卡死 | stdout/stderr PIPE 缓冲区满 | stdout→DEVNULL，stderr 后台线程排空 |
| MCP 工具重复获取 | 每次请求 HTTP GET /tools | `_cached_tools` 全局缓存，只获取一次 |
| ReAct Thought 多余 | 每步额外一次 LLM 推理 | 使用 `create_agent`（LangGraph），省略 Thought 直接行动 |

### Q12：如果让你继续优化这个项目，你会怎么做？

**A**：
1. **工具结果缓存**：相同查询短时间内的 MCP 工具结果缓存（Redis），避免重复查数据库
2. **流式 Tool Calling**：当前 A2A 代理等工具结果全部返回后才生成回复，可改为边查边输出
3. **A2A 代理健康检查**：主服务定期 ping 子代理，故障时自动降级（返回友好提示而非超时等待）
4. **意图识别优化**：用小模型（如 qwen-turbo）做意图分类，大模型只做最终回复生成，降低延迟和成本
5. **多轮对话管理**：当前短期记忆无限增长，应增加滑动窗口或摘要压缩机制

### Q13：查询火车票一直超时无响应，你是怎么排查和解决的？

**A**：这是项目中最复杂的排查过程，涉及多个层面，最终定位到 **3 个叠加 bug**。

**现象**：输入"8.1 北京到成都的火车票"，前端卡住 3 分钟无响应，后台日志显示 A2A 调用超时：
```
HTTPConnectionPool(host='127.0.0.1', port=5006): Read timed out. (read timeout=120)
```

**排查过程**（逐层排除）：

| 排查步骤 | 怀疑点 | 验证方法 | 结论 |
|---------|--------|---------|------|
| ① MySQL 连接 | Docker 容器没启动？连接不上？ | 直接用 pymysql 查询 MySQL | 连接正常（0.031s），查询正常（0.001s），**排除** |
| ② 连接池耗尽 | pool_size=5 不够？连接没归还？ | 检查 `_execute_query` 的 finally 块 | 有 `conn.close()` 归还连接，**排除** |
| ③ MCP 工具调用 | MCP Server 8001 不响应？ | curl `http://127.0.0.1:8001/tools` | 工具列表正常返回，**排除** |
| ④ LangChain Agent | Agent 执行太慢？ | 绕过 A2A，直接运行 `agent_executor.invoke()` | **13.2 秒完成**，Agent 本身没问题 |
| ⑤ A2A HTTP 调用 | Flask Server 卡住？ | 单独启动 ticket_server，curl 测试 | 16.2 秒完成，**单独运行正常** |
| ⑥ 子进程环境 | api_server.py 的子进程有问题？ | 对比单独运行 vs 子进程运行 | **定位到子进程才出问题** |

**最终定位到 3 个叠加 bug**：

**Bug 1：`send_task_async` 阻塞事件循环**
```python
# ❌ 错误写法：send_task_async 名字带 async 但实际是阻塞方法
raw_response = await agent.send_task_async(task)  # 直接 await 会卡死事件循环

# ✅ 正确写法：用 run_in_executor 包装（参考项目原始代码）
raw_response = await asyncio.get_event_loop().run_in_executor(
    None, lambda: asyncio.run(agent.send_task_async(task))
)
```
原因：`python_a2a` 库的 `send_task_async` 内部使用 `requests`（同步 HTTP），并非真正的异步协程。直接 `await` 会阻塞整个事件循环。

**Bug 2：子进程 PIPE 缓冲区满**
```python
# ❌ 错误写法：stdout/stderr 都 PIPE 但从不读取
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# ✅ 正确写法：stdout 丢弃，stderr 后台线程排空
proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
threading.Thread(target=_drain_pipe, args=(proc,), daemon=True).start()
```
原因：AgentExecutor 的 `verbose=True` 产生大量日志输出，PIPE 缓冲区（Windows ~4KB）被填满后，子进程在 `print()` 时阻塞挂起。

**Bug 3：Docker 容器启动顺序**
```
正确顺序：docker compose up -d → python api_server.py（自动拉起 6 个子服务）
```
原因：api_server.py 启动子服务时，如果 MySQL/Milvus 容器未就绪，MCP 层连接失败，A2A 层查询超时。

**经验总结**：
- **名字带 async 不一定是真异步**：要查看底层实现，确认是否使用 `aiohttp`/`httpx` 等异步库
- **subprocess PIPE 必须排空**：不读取的 PIPE 会导致子进程阻塞，生产环境应使用 DEVNULL 或日志文件
- **分层排查法**：从底层（MySQL）→ 中间层（MCP）→ 上层（A2A）逐层验证，快速缩小问题范围
