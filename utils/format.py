"""
金融领域，金额的数据类型 Decimal(18,6),表示18位总长度，其中6位是小数，不能直接转换为 float/double

"""

import json
from datetime import date, datetime, timedelta
from decimal import Decimal


def default_encoder(obj):
    """格式化单个对象，将非标准类型转换为JSON兼容格式"""
    if isinstance(obj, datetime):
        return obj.strftime('%Y-%m-%d %H:%M:%S') # 格式化datetime对象为字符串YYYY-MM-DD HH:MM:SS
    if isinstance(obj, date):
        return obj.strftime('%Y-%m-%d') # 格式化date对象为字符串YYYY-MM-DD
    if isinstance(obj, timedelta):
        return str(obj) # 格式化timedelta对象为字符串
    if isinstance(obj, Decimal):
        return float(obj) # 将Decimal对象转换为float
    return obj

# 自定义JSON编码器类，处理非标准类型序列化
class DateEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (date, datetime)):
            return obj.strftime('%Y-%m-%d %H:%M:%S') if isinstance(obj, datetime) else obj.strftime('%Y-%m-%d')
        if isinstance(obj, timedelta):
            return str(obj)
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)