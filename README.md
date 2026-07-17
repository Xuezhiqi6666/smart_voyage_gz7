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
│:5005   │  │:5006   │  │:5007   │   (LLM + 工具调用)
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
| 代理框架 | LangChain Agent + Tool Calling |
| 代理通信 | python_a2a（A2A 协议） |
| 工具服务 | FastMCP（MCP 协议） |
| 数据库 | MySQL 8.0（Docker，端口 3307） |
| 向量库 | Milvus（旅游团语义搜索） |
| 天气数据 | 和风天气 API |
| Web 框架 | FastAPI + Uvicorn |
