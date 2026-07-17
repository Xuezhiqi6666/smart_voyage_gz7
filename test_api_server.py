"""
api_server.py 全面测试
覆盖所有 REST API 端点
"""
import sys, io, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import requests

BASE = "http://127.0.0.1:8080"
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

# ==================== 测试1：GET / 首页 ====================
print('='*60)
print('测试1：GET / 首页')
print('='*60)
try:
    r = requests.get(f'{BASE}/', timeout=5)
    check('HTTP 200', r.status_code == 200, f'实际: {r.status_code}')
    check('返回 HTML', 'html' in r.text[:200].lower() or 'smart' in r.text[:500].lower(),
          f'前100字: {r.text[:100]}')
except Exception as e:
    check('首页可访问', False, str(e))
print()

# ==================== 测试2：GET /api/agents 代理列表 ====================
print('='*60)
print('测试2：GET /api/agents 代理列表')
print('='*60)
try:
    r = requests.get(f'{BASE}/api/agents', timeout=10)
    data = r.json()
    check('HTTP 200', r.status_code == 200)
    check('status=success', data.get('status') == 'success')
    agents = data.get('data', [])
    check('返回代理列表', len(agents) >= 3, f'实际数量: {len(agents)}')
    agent_names = [a.get('name', '') for a in agents]
    check('包含 WeatherAssistant', 'WeatherAssistant' in agent_names, f'实际: {agent_names}')
    check('包含 TicketAssistant', 'TicketAssistant' in agent_names, f'实际: {agent_names}')
    check('包含 TripAssistant', 'TripAssistant' in agent_names, f'实际: {agent_names}')
    for a in agents:
        print(f'  代理: {a["name"]} - {a.get("description","")[:40]}... (技能: {len(a.get("skills",[]))}个)')
except Exception as e:
    check('代理列表可访问', False, str(e))
print()

# ==================== 测试3：POST /api/memory/clear 清空记忆 ====================
print('='*60)
print('测试3：POST /api/memory/clear 清空记忆')
print('='*60)
try:
    r = requests.post(f'{BASE}/api/memory/clear', timeout=5)
    data = r.json()
    check('HTTP 200', r.status_code == 200)
    check('status=success', data.get('status') == 'success')
    check('返回确认消息', '清空' in data.get('message', ''), f'实际: {data.get("message")}')
except Exception as e:
    check('清空记忆', False, str(e))
print()

# ==================== 测试4：GET /api/memory 记忆状态 ====================
print('='*60)
print('测试4：GET /api/memory 记忆状态（清空后）')
print('='*60)
try:
    r = requests.get(f'{BASE}/api/memory', timeout=5)
    data = r.json()
    check('HTTP 200', r.status_code == 200)
    check('status=success', data.get('status') == 'success')
    mem = data.get('data', {})
    check('short_term_messages 为空', len(mem.get('short_term_messages', [])) == 0,
          f'实际: {len(mem.get("short_term_messages", []))}')
    check('user_profile 为空', len(mem.get('user_profile', {})) == 0)
except Exception as e:
    check('记忆状态', False, str(e))
print()

# ==================== 测试5：POST /api/memory/profile 更新偏好 ====================
print('='*60)
print('测试5：POST /api/memory/profile 更新用户偏好')
print('='*60)
try:
    r = requests.post(f'{BASE}/api/memory/profile',
                      json={"profile": {"seat_type": "二等座", "cabin_type": "经济舱"}},
                      timeout=5)
    data = r.json()
    check('HTTP 200', r.status_code == 200)
    check('status=success', data.get('status') == 'success')
    # 验证偏好已生效
    r2 = requests.get(f'{BASE}/api/memory', timeout=5)
    mem2 = r2.json().get('data', {})
    profile = mem2.get('user_profile', {})
    check('seat_type 已保存', profile.get('seat_type') == '二等座', f'实际: {profile}')
    check('cabin_type 已保存', profile.get('cabin_type') == '经济舱', f'实际: {profile}')
    print(f'  偏好: {profile}')
except Exception as e:
    check('更新偏好', False, str(e))
print()

# ==================== 测试6：POST /api/chat 天气查询 ====================
print('='*60)
print('测试6：POST /api/chat 天气查询')
print('='*60)
try:
    t0 = time.time()
    r = requests.post(f'{BASE}/api/chat',
                      json={"message": "北京今天天气怎么样？"},
                      timeout=120)
    t1 = time.time()
    data = r.json()
    print(f'  耗时: {t1-t0:.1f}s')
    check('HTTP 200', r.status_code == 200)
    check('status=success', data.get('status') == 'success')
    msg = data.get('message', '')
    check('包含北京', '北京' in msg, f'回复: {msg[:100]}')
    check('包含天气信息', any(kw in msg for kw in ['温度', '°', '天气', '晴', '阴', '雨', '湿度']),
          f'回复: {msg[:100]}')
    print(f'  回复: {msg[:200]}')
except Exception as e:
    check('天气查询', False, str(e))
print()

# ==================== 测试7：POST /api/chat 火车票查询 ====================
print('='*60)
print('测试7：POST /api/chat 火车票查询')
print('='*60)
try:
    t0 = time.time()
    r = requests.post(f'{BASE}/api/chat',
                      json={"message": "查一下8月1日从北京到成都的火车票"},
                      timeout=120)
    t1 = time.time()
    data = r.json()
    print(f'  耗时: {t1-t0:.1f}s')
    check('HTTP 200', r.status_code == 200)
    msg = data.get('message', '')
    check('包含车次', any(kw in msg for kw in ['G351', 'G349', 'G89', '车次', '列车']),
          f'回复: {msg[:100]}')
    check('包含价格', any(kw in msg for kw in ['¥', '￥', '796', '元']),
          f'回复: {msg[:100]}')
    print(f'  回复: {msg[:200]}')
except Exception as e:
    check('火车票查询', False, str(e))
print()

# ==================== 测试8：POST /api/chat out_of_scope ====================
print('='*60)
print('测试8：POST /api/chat out_of_scope')
print('='*60)
try:
    t0 = time.time()
    r = requests.post(f'{BASE}/api/chat',
                      json={"message": "你好"},
                      timeout=30)
    t1 = time.time()
    data = r.json()
    print(f'  耗时: {t1-t0:.1f}s')
    check('HTTP 200', r.status_code == 200)
    msg = data.get('message', '')
    check('包含助手介绍', any(kw in msg for kw in ['旅行', '助手', '天气', '机票', '帮']),
          f'回复: {msg[:100]}')
    print(f'  回复: {msg[:200]}')
except Exception as e:
    check('out_of_scope', False, str(e))
print()

# ==================== 测试9：GET /api/memory 对话后记忆 ====================
print('='*60)
print('测试9：GET /api/memory 对话后记忆状态')
print('='*60)
try:
    r = requests.get(f'{BASE}/api/memory', timeout=5)
    mem = r.json().get('data', {})
    msgs = mem.get('short_term_messages', [])
    entities = mem.get('entity_history', [])
    profile = mem.get('user_profile', {})
    check('短期消息有记录', len(msgs) > 0, f'数量: {len(msgs)}')
    check('实体历史有记录', len(entities) > 0, f'数量: {len(entities)}')
    check('用户偏好保留', profile.get('seat_type') == '二等座', f'偏好: {profile}')
    print(f'  短期消息: {len(msgs)} 条')
    print(f'  实体历史: {len(entities)} 条')
    print(f'  用户偏好: {profile}')
except Exception as e:
    check('对话后记忆', False, str(e))
print()

# ==================== 测试10：POST /api/chat/stream SSE流式 ====================
print('='*60)
print('测试10：POST /api/chat/stream SSE流式输出')
print('='*60)
try:
    t0 = time.time()
    r = requests.post(f'{BASE}/api/chat/stream',
                      json={"message": "北京今天天气"},
                      stream=True, timeout=120)
    check('HTTP 200', r.status_code == 200)
    check('Content-Type SSE', 'event-stream' in r.headers.get('content-type', ''),
          f'实际: {r.headers.get("content-type")}')
    chunks = []
    full_text = ''
    for line in r.iter_lines(decode_unicode=True):
        if line and line.startswith('data: '):
            payload = line[6:]  # 去掉 "data: " 前缀
            if payload == '[DONE]':
                break
            try:
                chunk_data = json.loads(payload)
                chunk_text = chunk_data.get('chunk', '')
                chunks.append(chunk_text)
                full_text += chunk_text
            except json.JSONDecodeError:
                pass
    t1 = time.time()
    print(f'  耗时: {t1-t0:.1f}s')
    print(f'  SSE片段数: {len(chunks)}')
    print(f'  完整回复: {full_text[:200]}')
    check('收到SSE片段', len(chunks) > 0, f'片段数: {len(chunks)}')
    check('包含天气信息', any(kw in full_text for kw in ['天气', '温度', '°', '晴', '阴', '北京']))
except Exception as e:
    check('SSE流式', False, str(e))
print()

# ==================== 结果 ====================
print('='*60)
print(f'测试结果: {passed} 通过, {failed} 失败')
print('='*60)
