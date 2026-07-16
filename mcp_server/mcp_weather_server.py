# ============================================================
# MCP 天气服务模块
# 功能：提供天气查询的 MCP（Model Context Protocol）工具接口
# 支持两种数据源：MySQL 数据库 / 和风天气 API（通过配置切换）
# 启动命令：python mcp_server/mcp_weather_server.py
# 访问地址：http://127.0.0.1:8002/mcp
#
# 调用流程图：
#
#   AI 智能体（如 Claude）
#        │
#        ▼
#   ┌─────────────────────────────┐
#   │  MCP Server (端口 8002)      │
#   │  create_weather_mcp_server() │
#   │  FastMCP("WeatherTools")     │
#   └──────────────┬──────────────┘
#                  │
#                  ▼
#           query_weather(city, start_date, end_date)
#                  │
#        ┌─────────┴─────────┐
#        │ 检查 weather_source│
#        └─────────┬─────────┘
#                  │
#        ┌─────────┴─────────┐
#        │                   │
#     "api"               其他（"db"）
#        │                   │
#        ▼                   ▼
#   ┌──────────────┐  ┌──────────────────┐
#   │ fetch_weather│  │ WeatherService   │
#   │ _from_api()  │  │ .query_weather() │
#   └──────┬───────┘  └────────┬─────────┘
#          │                   │
#          ▼                   ▼
#   ① geo接口:城市→ID     SQL查询 weather_data 表
#          │           SELECT * FROM weather_data
#          ▼            WHERE city = ? AND
#   ② 天气API:用ID获取数据    fx_date BETWEEN ? AND ?
#          │                   │
#          ▼                   ▼
#   ③ 过滤日期范围         遍历结果集（字典格式）
#   ④ 字段名映射           处理 date/Decimal 类型
#      tempMax→temp_max      │
#      textDay→text_day      ▼
#                     json.dumps() 序列化为 JSON 字符串
#          │                   │
#          └─────────┬─────────┘
#                    ▼
#           JSON 字符串返回给 AI 智能体
#
#   返回格式示例：
#   {
#     "status": "success",
#     "data": [
#       {
#         "city": "成都",
#         "fx_date": "2026-04-29",
#         "temp_max": 28,
#         "temp_min": 18,
#         "text_day": "多云",
#         "text_night": "晴",
#         "humidity": 65,
#         "wind_dir_day": "南风",
#         "wind_scale_day": "3",
#         "precip": 0.0
#       }
#     ]
#   }
#
# 数据源切换：修改 config.py 中的 weather_source 配置
#   weather_source = "api"  → 使用和风天气实时 API
#   weather_source = "db"   → 使用本地 MySQL 数据库
# ============================================================


import mysql.connector  # MySQL 数据库驱动，用于连接和查询本地天气数据
from mysql.connector.pooling import MySQLConnectionPool  # 连接池，复用数据库连接避免频繁创建/销毁
import json  # JSON 序列化/反序列化，将 Python 字典转为 JSON 字符串返回
from datetime import date, datetime, timedelta  # 日期类型，处理数据库返回的时间字段
from decimal import Decimal  # 高精度数字类型，处理数据库返回的数值字段
from python_a2a import FastMCP, create_fastapi_app  # MCP 框架：FastMCP 用于定义工具，create_fastapi_app 转为 FastAPI 服务
import uvicorn  # ASGI 服务器，用于运行 FastAPI 应用
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import Config  # 项目配置类，读取数据库/API 等配置信息
from create_logger import logger  # 日志记录器，统一输出运行日志
from utils.format import DateEncoder, default_encoder  # 自定义 JSON 编码器，处理 date/Decimal 等特殊类型
import requests  # HTTP 请求库，用于调用和风天气 API

# 加载全局配置对象
conf = Config()

def normalize_date(date_str: str, default: str = None) -> str:
    """
    规范化日期参数：空值使用默认值，非空则校验格式

    参数：
        date_str: 日期字符串，可能为空或格式不正确
        default: 空值时的默认值，格式 "YYYY-MM-DD"；不传则默认为当天

    返回：
        合法的日期字符串，格式 "YYYY-MM-DD"

    异常：
        ValueError: 当日期非空但格式不合法时抛出
    """
    if not date_str or not date_str.strip():
        # 空值时使用调用方指定的默认值，未指定则用当天
        if default is None:
            default = date.today().strftime("%Y-%m-%d")
        logger.info(f"日期参数为空，自动填充默认值：{default}")
        return default

    # 非空则校验格式，格式错误抛异常由调用方处理
    datetime.strptime(date_str.strip(), "%Y-%m-%d")
    return date_str.strip()


class WeatherService:
    """
    天气服务类：负责从 MySQL 数据库查询天气数据
    工作原理：在初始化时创建数据库连接池，后续查询从池中取连接，用完归还，避免频繁创建/销毁连接
    """

    def __init__(self):
        # 创建 MySQL 连接池，复用连接，避免高并发下频繁创建/销毁连接
        if conf.weather_source == 'database':
            self.pool = MySQLConnectionPool(
                pool_name=conf.pool_name,  # 连接池名称，用于标识该池
                pool_size=conf.pool_size,  # 连接池大小（最大连接数），从 config.py 读取
                host=conf.host,  # 数据库主机地址
                port=conf.port,  # 数据库端口号
                user=conf.user,  # 数据库用户名
                password=conf.password,  # 数据库密码
                database=conf.database  # 数据库名称
            )

    def query_weather(self, city: str, start_date: str, end_date: str) -> str:
        """
        参数化查询数据库天气数据，返回 JSON 字符串

        参数：
            city: 城市名称，如 "成都"
            start_date: 开始日期，格式 "YYYY-MM-DD"
            end_date: 结束日期，格式 "YYYY-MM-DD"

        返回：
            JSON 字符串，包含 status（状态）和 data（天气数据列表）
        """
        # 从连接池获取连接，用完在 finally 中归还，保证连接不会泄漏
        conn = None
        try:
            # TODO 和直接使用mysql-conn的区别在于这里是从池子里拿出来的
            conn = self.pool.get_connection()  # 从池中取出一个可用连接
            cursor = conn.cursor(dictionary=True)  # dictionary=True 使结果为字典而非元组

            # 参数化 SQL 查询：使用 %s 占位符防止 SQL 注入攻击
            # 查询 weather_data 表中指定城市和时间范围内的天气记录
            sql = ("SELECT city, fx_date, temp_max, temp_min, text_day, text_night, "
                   "humidity, wind_dir_day, wind_scale_day, precip FROM weather_data "
                   "WHERE city = %s AND fx_date BETWEEN %s AND %s ORDER BY fx_date")
            cursor.execute(sql, [city, start_date, end_date])

            results = cursor.fetchall()
            cursor.close()

            # 处理特殊类型字段：date/Decimal 等类型无法直接 JSON 序列化，需要转换
            for result in results:
                for key, value in result.items():
                    if isinstance(value, (date, datetime, timedelta, Decimal)):
                        result[key] = default_encoder(value)

            return json.dumps(
                {"status": "success", "data": results} if results
                else {"status": "no_data", "message": "未找到天气数据，请确认城市和日期。"},
                cls=DateEncoder, ensure_ascii=False
            )
        except Exception as e:
            logger.error(f"天气查询错误: {str(e)}")
            return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)
        finally:
            # 无论成功还是异常，都要将连接归还给连接池，否则池会被耗尽
            if conn is not None:
                conn.close()  # 连接池模式下 close() 不会真正关闭，而是归还到池中


def fetch_weather_from_api(city: str, start_date: str, end_date: str) -> str:
    """
    从和风天气 API 获取天气数据

    工作流程：
        1. 先用城市名调用 geo 接口获取 location_id
        2. 再用 location_id 调用天气预报接口获取数据
        3. 过滤出指定日期范围内的数据，格式化后返回

    参数：
        city: 城市名称，如 "成都"
        start_date: 开始日期，格式 "YYYY-MM-DD"
        end_date: 结束日期，格式 "YYYY-MM-DD"

    返回：
        JSON 字符串，格式与数据库查询一致
    """
    logger.info(f"从和风天气API获取数据: {city}, {start_date} ~ {end_date}")

    # 设置请求头，传入和风天气 API Key（认证凭证）
    headers = {
        "X-QW-Api-Key": conf.weather_api_key
    }

    # 第一步：城市名 → 地理位置 ID（和风天气需要先查城市 ID）
    geo_url = f"https://{conf.weather_api_host}/geo/v2/city/lookup?location={city}"
    geo_resp = requests.get(geo_url, headers=headers)
    geo_data = geo_resp.json()

    # 如果 API 没有找到该城市，直接返回错误信息
    if not geo_data.get("location"):
        return json.dumps({"status": "no_data", "message": f"未找到城市：{city}"}, ensure_ascii=False)
    # print(f"geo_data: {geo_data}")
    # 取第一个匹配的城市 ID
    location_id = geo_data["location"][0]["id"]

    # 第二步：用 location_id 获取天气预报数据
    weather_url = f"{conf.weather_base_url}?location={location_id}"
    weather_resp = requests.get(weather_url, headers=headers)
    weather_data = weather_resp.json()

    # 第三步：过滤日期范围，并将 API 返回的字段映射为数据库同名字段
    results = []
    for day in weather_data.get("daily", []):
        # 字符串比较即可，因为日期格式都是 "YYYY-MM-DD"
        if start_date <= day.get("fxDate", "") <= end_date:
            results.append({
                "city": city,  # 城市名
                "fx_date": day.get("fxDate"),  # 预报日期
                "temp_max": int(day.get("tempMax", 0)),  # 最高温度（转为整数）
                "temp_min": int(day.get("tempMin", 0)),  # 最低温度（转为整数）
                "text_day": day.get("textDay", ""),  # 白天天气描述
                "text_night": day.get("textNight", ""),  # 夜间天气描述
                "humidity": int(day.get("humidity", 0)),  # 湿度百分比
                "wind_dir_day": day.get("windDirDay", ""),  # 白天风向
                "wind_scale_day": day.get("windScaleDay", ""),  # 白天风力等级
                "precip": float(day.get("precip", 0))  # 降水量（保留浮点）
            })

    # 根据是否有结果返回对应格式
    if results:
        return json.dumps({"status": "success", "data": results}, ensure_ascii=False)
    return json.dumps({"status": "no_data", "message": "未找到天气数据。"}, ensure_ascii=False)


def create_weather_mcp_server():
    """
    创建并启动天气 MCP 服务器

    MCP（Model Context Protocol）：一种协议，允许 AI 模型（如 Claude）通过标准接口调用外部工具
    本模块将天气查询功能封装为 MCP 工具，供 AI 智能体调用
    """
    # 创建 FastMCP 实例，相当于定义一个工具服务
    # TODO 1.构建FastMCP实例
    weather_mcp = FastMCP(
        name="WeatherTools",  # 工具服务名称，AI 调用时会看到这个名称
    )

    # 实例化数据库查询服务
    service = WeatherService()

    # TODO 2.基于MCP实例对象，使用装饰器的方式注册可用工具
    # TODO description非常重要，尤其是日期的格式，需要明确指定。不然调用API可能会失败
    # 使用 @tool 装饰器将函数注册为 MCP 工具
    # name: 工具的唯一标识，AI 通过这个名字调用
    # description: 工具说明，AI 会阅读这段描述来判断是否使用该工具
    @weather_mcp.tool(
        name="query_weather",
        description="查询天气数据，参数：city(城市), start_date(开始日期YYYY-MM-DD), end_date(结束日期YYYY-MM-DD)"
    )
    def query_weather(city: str, start_date: str, end_date: str) -> str:
        """
        MCP 工具入口函数：根据配置选择从数据库或 API 获取天气数据
        入口统一校验日期参数，防止空值或非法格式导致 SQL 错误

        参数：
            city: 城市名称，如 "成都"
            start_date: 开始日期，格式 "YYYY-MM-DD"
            end_date: 结束日期，格式 "YYYY-MM-DD"

        返回：
            JSON 字符串，包含天气数据
        """
        # 规范化日期参数：开始日期空值补当天，结束日期空值补当天往后29天，格式错误返回提示
        try:
            today = date.today()
            start_date = normalize_date(start_date, default=today.strftime("%Y-%m-%d"))
            end_date = normalize_date(end_date, default=(today + timedelta(days=29)).strftime("%Y-%m-%d"))
        except ValueError as e:
            return json.dumps(
                {"status": "error", "message": f"日期格式不正确：{e}，请使用 YYYY-MM-DD 格式，如 2026-05-01"},
                ensure_ascii=False
            )

        logger.info(f"执行天气查询: {city}, {start_date} ~ {end_date}")

        # 根据 config.py 中的 weather_source 配置决定数据源：
        # - "api": 调用和风天气实时 API
        # - 其他值（如 "db"）: 查询本地 MySQL 数据库
        if conf.weather_source == "api":
            return fetch_weather_from_api(city, start_date, end_date)
        elif conf.weather_source == "database":
            return service.query_weather(city, start_date, end_date)
        else:
            logger.error("查询配置信息错误{}", conf.weather_source)
            return json.dumps({"status": "error", "message": "配置错误"}, ensure_ascii=False)

    # 打印服务启动信息到日志
    logger.info("=== 天气MCP服务器信息 ===")
    logger.info(f"名称: {weather_mcp.name}")

    return weather_mcp


if __name__ == '__main__':
    # 调试用法：直接调用 API 测试，不启动 MCP 服务
    # api = fetch_weather_from_api(city="成都", start_date="2026-04-28", end_date="2026-05-01")
    # print(api)

    # 启动 MCP 天气服务器
    weather_mcp = create_weather_mcp_server()
    print("服务器已启动，请访问 http://127.0.0.1:8002/mcp")
    app = create_fastapi_app(weather_mcp)
    uvicorn.run(app, host="0.0.0.0", port=8002)
