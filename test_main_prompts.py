"""
main_prompts.py 测试脚本
测试所有 Prompt 模板的结构正确性和 LLM 调用效果
"""
import sys, io, os, json, re
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from main_prompts import SmartVoyagePrompts
from langchain_openai import ChatOpenAI
from config import Config

conf = Config()
llm = ChatOpenAI(model=conf.model_name, base_url=conf.base_url, api_key=conf.api_key, temperature=0.1)

passed = 0
failed = 0

def check(name, condition, detail=''):
    global passed, failed
    if condition:
        passed += 1
        print(f'  PASS: {name}')
    else:
        failed += 1
        print(f'  FAIL: {name} {detail}')

def clean_json(text):
    """清理 LLM 返回的 JSON（去掉可能的代码块标记）"""
    return re.sub(r'^```json\s*|\s*```$', '', text).strip()

# ==================== 测试1：模板结构验证 ====================
print('='*60)
print('测试1：模板结构验证（变量名正确性）')
print('='*60)

p = SmartVoyagePrompts.intent_prompt()
check('intent_prompt 变量完整',
      {'user_profile', 'task_context', 'conversation_history', 'query'}.issubset(set(p.input_variables)),
      f'实际: {p.input_variables}')

p = SmartVoyagePrompts.summarize_weather_prompt()
check('summarize_weather 变量', {'query', 'raw_response'}.issubset(set(p.input_variables)))

p = SmartVoyagePrompts.summarize_ticket_prompt()
check('summarize_ticket 变量', {'query', 'raw_response'}.issubset(set(p.input_variables)))

p = SmartVoyagePrompts.attraction_prompt()
check('attraction_prompt 变量', {'query', 'weather_info'}.issubset(set(p.input_variables)))

p = SmartVoyagePrompts.planning_prompt()
check('planning_prompt 变量', {'conversation_history', 'query', 'intents', 'user_queries'}.issubset(set(p.input_variables)))

p = SmartVoyagePrompts.react_prompt()
check('react_prompt 变量', {'available_tools', 'plan_steps', 'observations', 'current_step', 'step_description', 'query'}.issubset(set(p.input_variables)))

p = SmartVoyagePrompts.react_summary_prompt()
check('react_summary 变量', {'query', 'all_observations'}.issubset(set(p.input_variables)))
print()

# ==================== 测试2：intent_prompt 单意图 ====================
print('='*60)
print('测试2：intent_prompt 单意图识别（机票）')
print('='*60)
chain = SmartVoyagePrompts.intent_prompt() | llm
resp = chain.invoke({
    'user_profile': '无已知的用户偏好',
    'task_context': '{}',
    'conversation_history': '',
    'query': '帮我查一下明天北京到上海的机票'
}).content.strip()
resp = clean_json(resp)
try:
    result = json.loads(resp)
    check('返回有效 JSON', True)
    check('包含 intents', 'intents' in result)
    check('识别为 flight', 'flight' in result.get('intents', []), f'实际: {result.get("intents")}')
    check('包含 user_queries', 'user_queries' in result)
    queries = result.get('user_queries', {})
    flight_q = queries.get('flight', '')
    check('改写包含北京', '北京' in flight_q, f'实际: {flight_q}')
    check('改写包含上海', '上海' in flight_q, f'实际: {flight_q}')
except json.JSONDecodeError as e:
    check('返回有效 JSON', False, f'解析失败: {resp[:200]}')
print()

# ==================== 测试3：intent_prompt 多意图 ====================
print('='*60)
print('测试3：intent_prompt 多意图识别')
print('='*60)
resp2 = chain.invoke({
    'user_profile': 'seat_type: 二等座',
    'task_context': '{}',
    'conversation_history': '',
    'query': '我想8月1日从北京去成都，帮我查一下天气和火车票'
}).content.strip()
resp2 = clean_json(resp2)
try:
    result2 = json.loads(resp2)
    intents = result2.get('intents', [])
    check('返回有效 JSON', True)
    check('识别出 weather', 'weather' in intents, f'实际: {intents}')
    check('识别出 train', 'train' in intents, f'实际: {intents}')
    check('follow_up_message 为空（不需要追问）', result2.get('follow_up_message', '') == '',
          f'实际: {result2.get("follow_up_message")}')
except json.JSONDecodeError:
    check('返回有效 JSON', False, f'解析失败: {resp2[:200]}')
print()

# ==================== 测试4：intent_prompt out_of_scope ====================
print('='*60)
print('测试4：intent_prompt out_of_scope')
print('='*60)
resp3 = chain.invoke({
    'user_profile': '无已知的用户偏好',
    'task_context': '{}',
    'conversation_history': '',
    'query': '你好，你是谁？'
}).content.strip()
resp3 = clean_json(resp3)
try:
    result3 = json.loads(resp3)
    check('识别为 out_of_scope', 'out_of_scope' in result3.get('intents', []), f'实际: {result3.get("intents")}')
    check('follow_up_message 非空', len(result3.get('follow_up_message', '')) > 0,
          f'实际: {result3.get("follow_up_message", "")}')
except json.JSONDecodeError:
    check('返回有效 JSON', False, f'解析失败: {resp3[:200]}')
print()

# ==================== 测试5：planning_prompt ====================
print('='*60)
print('测试5：planning_prompt 任务规划')
print('='*60)
plan_chain = SmartVoyagePrompts.planning_prompt() | llm
plan_resp = plan_chain.invoke({
    'conversation_history': '',
    'query': '帮我查北京到上海的机票，再看下那边天气，推荐几个景点',
    'intents': json.dumps(['flight', 'weather', 'attraction'], ensure_ascii=False),
    'user_queries': json.dumps({
        'flight': '北京到上海机票',
        'weather': '上海天气',
        'attraction': '上海景点推荐'
    }, ensure_ascii=False)
}).content.strip()
plan_resp = clean_json(plan_resp)
try:
    plan = json.loads(plan_resp)
    check('返回有效 JSON', True)
    check('包含 need_plan', 'need_plan' in plan)
    check('包含 steps', 'steps' in plan)
    steps = plan.get('steps', [])
    check('步骤数 >= 3', len(steps) >= 3, f'实际步骤数: {len(steps)}')
    check('每步有 intent', all('intent' in s for s in steps))
    check('每步有 depends_on', all('depends_on' in s for s in steps))
    check('每步有 action', all('action' in s for s in steps))
except json.JSONDecodeError:
    check('返回有效 JSON', False, f'解析失败: {plan_resp[:200]}')
print()

# ==================== 测试6：attraction_prompt ====================
print('='*60)
print('测试6：attraction_prompt 景点推荐')
print('='*60)
attr_chain = SmartVoyagePrompts.attraction_prompt() | llm
attr_resp = attr_chain.invoke({
    'query': '推荐几个北京景点',
    'weather_info': '北京明天晴，气温25-35度'
}).content.strip()
check('景点推荐非空', len(attr_resp) > 50, f'长度: {len(attr_resp)}')
check('包含北京相关', any(kw in attr_resp for kw in ['北京', '故宫', '长城', '天安门', '颐和园']),
      f'回复前100字: {attr_resp[:100]}')
check('包含天气建议', any(kw in attr_resp for kw in ['晴', '防晒', '气温', '天气']),
      f'未包含天气相关信息')
print()

# ==================== 测试7：react_summary_prompt ====================
print('='*60)
print('测试7：react_summary_prompt 汇总回复')
print('='*60)
summary_chain = SmartVoyagePrompts.react_summary_prompt() | llm
summary_resp = summary_chain.invoke({
    'query': '帮我查北京到上海的机票和上海天气',
    'all_observations': '步骤1 (查机票): MU5101 经济舱1280元 余票89\n步骤2 (查天气): 上海明天晴 25-32度'
}).content.strip()
check('汇总回复非空', len(summary_resp) > 30, f'长度: {len(summary_resp)}')
check('汇总包含机票信息', any(kw in summary_resp for kw in ['机票', 'MU', '1280', '航班']),
      f'回复: {summary_resp[:150]}')
check('汇总包含天气信息', any(kw in summary_resp for kw in ['天气', '晴', '25', '32', '温度']),
      f'回复: {summary_resp[:150]}')
print()

# ==================== 测试8：summarize_weather_prompt ====================
print('='*60)
print('测试8：summarize_weather_prompt 天气总结')
print('='*60)
weather_sum_chain = SmartVoyagePrompts.summarize_weather_prompt() | llm
weather_sum_resp = weather_sum_chain.invoke({
    'query': '北京明天天气',
    'raw_response': json.dumps({
        "status": "success",
        "data": [{"city": "北京", "fx_date": "2026-07-18", "temp_max": 33, "temp_min": 22,
                   "text_day": "晴", "text_night": "多云", "humidity": 45, "wind_dir_day": "东南风"}]
    }, ensure_ascii=False)
}).content.strip()
check('天气总结非空', len(weather_sum_resp) > 30, f'长度: {len(weather_sum_resp)}')
check('包含温度信息', any(kw in weather_sum_resp for kw in ['33', '22', '温度']),
      f'回复: {weather_sum_resp[:150]}')
print()

# ==================== 测试9：summarize_ticket_prompt ====================
print('='*60)
print('测试9：summarize_ticket_prompt 票务总结')
print('='*60)
ticket_sum_chain = SmartVoyagePrompts.summarize_ticket_prompt() | llm
ticket_sum_resp = ticket_sum_chain.invoke({
    'query': '北京到上海的机票',
    'raw_response': json.dumps({
        "status": "success",
        "data": [{"flight_no": "MU5101", "departure_time": "08:30", "cabin_type": "经济舱",
                   "price": 1280, "remaining_seats": 89}]
    }, ensure_ascii=False)
}).content.strip()
check('票务总结非空', len(ticket_sum_resp) > 30, f'长度: {len(ticket_sum_resp)}')
check('包含航班信息', any(kw in ticket_sum_resp for kw in ['MU5101', '机票', '1280', '航班']),
      f'回复: {ticket_sum_resp[:150]}')
print()

# ==================== 结果 ====================
print('='*60)
print(f'测试结果: {passed} 通过, {failed} 失败')
print('='*60)
