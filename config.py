"""
需求：管理SmartVoyage项目的配置信息，包括大模型、数据库、日志等配置
思路步骤：
1. 加载环境变量
2. 定义项目根目录路径
3. 创建Config类管理所有配置项
4. 配置大模型参数（API地址、密钥、模型名称）
5. 配置数据库参数（主机、用户名、密码、数据库名）
6. 配置日志文件路径
7. 配置票务查询接口地址
8. 配置意图映射字典
9. 实现根据环境获取不同数据库配置的方法
"""

import os
import logging
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# 项目根目录
project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))

#定义配置文件
class Config:

    def __init__(self):
        # 大模型配置
        self.base_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        # TODO 请确认模型名称是否正确，用来帅锅的（这里没做，以后再做）
        self.model_name = 'qwen3.7-max'

        # 数据库配置
        self.host = 'localhost'
        self.port = 3307  # 与 docker-compose.yml 的端口映射保持一致（宿主机 3307 → 容器 3306）
        # 业务账号由 docker-compose 的 MYSQL_USER/MYSQL_PASSWORD 自动创建，无需手动调用 ensure_mysql_user
        # TODO 实际工作中，一般一个业务创建一个账号。不要使用root，风险太大
        self.user = 'smart_voyage'
        self.password = '123456'  # 不是root
        self.database = 'travel_rag'

        # 数据库连接池配置
        self.pool_name = "smart_voyage_pool"  # 连接池名称
        self.pool_size = 5  # 连接池大小（最大连接数）

        # 日志配置
        self.log_file = os.path.join(project_root, 'logs', 'app.log')

        # # 票务查询的12306接口地址
        # self.url_123 = ""

        # 路由：意图和对应的Agent的对应关系
        self.intent = {
            "weather": "WeatherAssistant", # 查天气
            "flight": "TicketAssistant", # 查票务-机票
            "train": "TicketAssistant", # 查票务-火车票
            "concert": "TicketAssistant", # 查票务-演唱会
            "order": "TicketAssistant", # 查票务-查订单
            "car_rental": "TripAssistant", # 查旅行服务-租车
            "tour_group": "TripAssistant", # 查旅行服务-旅游团
            "insurance": "TripAssistant", # 查旅行服务-保险
            "trip_order": "TripAssistant", # 查旅行服务-订单
        }

        self.temperature = 0.1

        # 天气数据源配置
        # 默认："database"
        # 可选值："database"（从数据库获取） / "api"（直接从和风API获取）
        self.weather_source = "api"

        # 和风天气 API 配置
        self.test_api = "test_api"
        self.weather_api_key = "dfd7d1516bb148e897644c2c88cc805b"
        self.weather_api_host = "jv359h7m2j.re.qweatherapi.com"
        self.weather_base_url = f"https://{self.weather_api_host}/v7/weather/30d"
        self.weather_timezone = "Asia/Shanghai"

        # 天气监控城市列表（城市名 -> 城市代码）
        self.weather_city_codes = {
            "北京": "101010100",
            "成都": "101270101"
        }

        # 天气定时更新调度时间（北京时间）
        self.weather_schedule_time = "01:00"

        # Milvus 向量数据库配置
        self.milvus_host = "localhost"
        self.milvus_port = 19530
        self.tour_group_collection = "tour_groups"

        # Qwen Embedding API 配置
        self.embedding_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
        self.embedding_dim = 1024

def ensure_mysql_user(host='localhost', port=3307, root_user='root', root_password='123456',
                      target_user='smart_voyage', target_password='123456',
                      target_database='travel_rag'):
    """
    确保MySQL中存在指定用户，不存在则自动创建并授权。
    使用root账号连接，检查并创建业务用户。
    注意：若已在 docker-compose.yml 中配置 MYSQL_USER/MYSQL_PASSWORD，此函数可跳过调用。

    :param host: MySQL主机地址
    :param port: MySQL端口（需与 docker-compose 映射一致）
    :param root_user: 用于管理的管理员账号（默认root）
    :param root_password: 管理员账号密码
    :param target_user: 需要确保存在的业务账号
    :param target_password: 业务账号密码
    :param target_database: 业务账号需要访问的数据库
    """
    import mysql.connector
    from mysql.connector import Error

    try:
        # 用root账号连接MySQL（显式指定 port，避免与 docker-compose 端口不一致）
        conn = mysql.connector.connect(
            host=host,
            port=port,
            user=root_user,
            password=root_password
        )
        cursor = conn.cursor()

        # 检查目标用户是否存在（同时检查 localhost 和 % 两种host）
        cursor.execute(
            "SELECT COUNT(*) FROM mysql.user WHERE user = %s",
            (target_user,)
        )
        user_exists = cursor.fetchone()[0] > 0

        # 确保目标数据库存在
        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS `{target_database}` "
            f"DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
        logger.info(f"已确保数据库 {target_database} 存在")

        if not user_exists:
            # 创建用户
            cursor.execute(
                f"CREATE USER '{target_user}'@'localhost' IDENTIFIED BY %s",
                (target_password,)
            )
            logger.info(f"已创建MySQL用户: {target_user}@localhost")

            # 授权：对目标数据库拥有全部权限
            cursor.execute(
                f"GRANT ALL PRIVILEGES ON `{target_database}`.* TO '{target_user}'@'localhost'"
            )
            cursor.execute("FLUSH PRIVILEGES")
            logger.info(f"已授权用户 {target_user} 访问数据库 {target_database}")
        else:
            logger.info(f"MySQL用户 {target_user} 已存在，跳过创建")

        cursor.close()
        conn.close()

    except Error as e:
        logger.warning(f"自动创建MySQL用户失败: {e}")
        logger.warning("请手动创建用户，或检查root账号密码是否正确")
        raise


if __name__ == '__main__':
    print(Config().log_file)
    print(Config().database)
    ensure_mysql_user()
