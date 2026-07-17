"""
MCP 工具转换模块
================

解决的问题：
    python_a2a 的 to_langchain_tool() 在转换 MCP 工具时，创建的是无 args_schema 的
    普通 Tool 对象。LangChain 会将其视为"单参数工具"（single-input tool），
    当 LLM 传入多个参数（尤其是可选参数）时，会报错：
        "Too many arguments to single-input tool"

    本模块通过以下方式修复：
    1. 从 MCP Server 的 /tools 端点获取工具的 JSON Schema 参数定义
    2. 动态构建 Pydantic BaseModel 作为 args_schema（区分必填/可选字段）
    3. 使用 LangChain 的 StructuredTool 替代普通 Tool
    4. 支持 string / integer / number / boolean 四种参数类型

使用方式：
    # 替换原来的 to_langchain_tool
    from utils.mcp_tools import to_structured_langchain_tool
    tools = to_structured_langchain_tool("http://127.0.0.1:8001")
"""

import requests
from typing import Optional

from pydantic import create_model
from langchain_core.tools import StructuredTool


# JSON Schema 类型 → Python 类型的映射
_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


def _build_args_schema(tool_name: str, parameters: dict):
    """
    从 MCP 工具的 JSON Schema 参数定义，动态构建 Pydantic BaseModel

    参数：
        tool_name: 工具名称，用于生成模型类名
        parameters: MCP 返回的 JSON Schema 格式参数定义，格式如下：
            {
                "type": "object",
                "properties": {
                    "departure_city": {"type": "string", "description": "出发城市"},
                    "cabin_type":     {"type": "string", "description": "舱位类型"}
                },
                "required": ["departure_city"]  # 不在其中的字段视为可选
            }

    返回值：
        动态生成的 Pydantic BaseModel 子类

    示例：
        对于 query_flight 工具，生成的模型等价于：
            class QueryFlightInput(BaseModel):
                departure_city: str
                arrival_city: str
                date: str
                cabin_type: Optional[str] = None  # 可选参数
    """
    properties = parameters.get("properties", {})
    required_fields = set(parameters.get("required", []))

    field_definitions = {}
    for field_name, field_info in properties.items():
        field_type_str = field_info.get("type", "string")
        python_type = _TYPE_MAP.get(field_type_str, str)

        if field_name in required_fields:
            # 必填字段：直接使用 Python 类型，无默认值
            field_definitions[field_name] = (python_type, ...)
        else:
            # 可选字段：包装为 Optional，默认值为 None
            field_definitions[field_name] = (Optional[python_type], None)

    # 动态创建 Pydantic 模型，类名取工具名首字母大写 + "Input"
    model_name = "".join(word.capitalize() for word in tool_name.split("_")) + "Input"
    return create_model(model_name, **field_definitions)


def to_structured_langchain_tool(mcp_url: str, tool_name: str = None):
    """
    将 MCP Server 的工具转换为带完整参数 Schema 的 LangChain StructuredTool

    与 to_langchain_tool() 的区别：
    ┌────────────────────────────┬──────────────────────────────────────┐
    │ to_langchain_tool()        │ to_structured_langchain_tool()       │
    ├────────────────────────────┼──────────────────────────────────────┤
    │ 创建普通 Tool              │ 创建 StructuredTool                  │
    │ 无 args_schema             │ 有 args_schema（Pydantic 模型）       │
    │ 单参数模式                  │ 多参数 + 可选参数                    │
    │ 可选参数会报错               │ 可选参数正常工作                     │
    └────────────────────────────┴──────────────────────────────────────┘

    参数：
        mcp_url: MCP Server 的地址，如 "http://127.0.0.1:8001"
        tool_name: 可选，只转换指定名称的工具；为 None 时转换全部工具

    返回值：
        - tool_name 不为 None 时：返回单个 StructuredTool
        - tool_name 为 None 时：返回 StructuredTool 列表

    异常：
        RuntimeError: MCP Server 连接失败或工具列表获取失败
    """
    # 步骤1：从 MCP Server 获取工具列表
    try:
        resp = requests.get(f"{mcp_url}/tools", timeout=10)
        resp.raise_for_status()
        available_tools = resp.json()
    except Exception as e:
        raise RuntimeError(f"无法从 MCP Server ({mcp_url}) 获取工具列表: {e}")

    # 步骤2：按名称过滤（如果指定了 tool_name）
    if tool_name is not None:
        available_tools = [t for t in available_tools if t.get("name") == tool_name]
        if not available_tools:
            raise RuntimeError(f"工具 '{tool_name}' 在 MCP Server 中不存在")

    # 步骤3：逐个转换为 StructuredTool
    langchain_tools = []
    for tool_info in available_tools:
        name = tool_info.get("name", "unnamed_tool")
        description = tool_info.get("description", f"MCP Tool: {name}")
        parameters = tool_info.get("parameters", {})

        # 构建 Pydantic 参数模型（区分必填 / 可选字段）
        args_schema = _build_args_schema(name, parameters)

        # 创建工具调用函数（闭包捕获 mcp_url 和 tool_name，避免循环引用问题）
        def make_func(url, tname):
            def tool_func(**kwargs):
                """调用 MCP Server 的工具端点，将参数以 JSON 方式 POST 过去"""
                response = requests.post(
                    f"{url}/tools/{tname}",
                    json=kwargs,
                    timeout=30
                )
                if response.status_code != 200:
                    return f"Error: HTTP {response.status_code} - {response.text}"

                result = response.json()

                # 处理错误响应
                if "error" in result:
                    return f"Error: {result['error']}"

                # 处理标准 MCP 内容格式
                if "content" in result:
                    content = result.get("content", [])
                    if content and isinstance(content, list) and "text" in content[0]:
                        return content[0]["text"]

                return str(result)
            return tool_func

        func = make_func(mcp_url, name)

        # 创建 StructuredTool（带 args_schema，支持多参数 + 可选参数）
        lc_tool = StructuredTool(
            name=name,
            description=description,
            func=func,
            args_schema=args_schema,
        )
        langchain_tools.append(lc_tool)

    # 返回单个工具或工具列表
    if tool_name is not None and len(langchain_tools) == 1:
        return langchain_tools[0]

    return langchain_tools
