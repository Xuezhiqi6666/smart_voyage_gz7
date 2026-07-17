"""
chat_service.py 全面测试
覆盖：单意图、多意图、out_of_scope、追问、景点推荐、记忆系统、流式输出
"""
import sys, io, os, time, asyncio
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from chat_service import ChatService

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

async def run_tests():
    global passed, failed

    service = ChatService()
    service.clear_memory()

    # ==================== 测试1：单意图 - 天气 ====================
    print('='*60)
    print('测试1：单意图 - 天气查询')
    print('='*60)
    t0 = time.time()
    resp = await service.chat('北京今天天气怎么样？')
    t1 = time.time()
    print(f'  耗时: {t1-t0:.1f}s')
    print(f'  回复: {resp[:200]}')
    check('状态码成功', len(resp) > 20)
    check('包含北京', '北京' in resp)
    check('包含天气信息', any(kw in resp for kw in ['温度', '°', '天气', '晴', '阴', '雨']))
    print()

    # ==================== 测试2：单意图 - 火车票 ====================
    print('='*60)
    print('测试2：单意图 - 火车票查询')
    print('='*60)
    t0 = time.time()
    resp = await service.chat('查一下8月1日从北京到成都的火车票')
    t1 = time.time()
    print(f'  耗时: {t1-t0:.1f}s')
    print(f'  回复: {resp[:200]}')
    check('包含车次信息', any(kw in resp for kw in ['G351', 'G349', 'G89', '车次', '列车']))
    check('包含价格', any(kw in resp for kw in ['¥', '￥', '796', '元']))
    check('包含座位类型', any(kw in resp for kw in ['二等座', '一等座', '商务座']))
    print()

    # ==================== 测试3：多意图 - 天气+火车票 ====================
    print('='*60)
    print('测试3：多意图 - 天气+火车票并行')
    print('='*60)
    t0 = time.time()
    resp = await service.chat('我想8月1日从北京去成都，帮我查一下天气和火车票')
    t1 = time.time()
    print(f'  耗时: {t1-t0:.1f}s')
    print(f'  回复: {resp[:300]}')
    check('包含天气信息', any(kw in resp for kw in ['天气', '温度', '°', '晴', '雨', '湿度']))
    check('包含票务信息', any(kw in resp for kw in ['车次', '列车', 'G3', '二等座', '¥', '￥']))
    print()

    # ==================== 测试4：out_of_scope ====================
    print('='*60)
    print('测试4：out_of_scope - 闲聊')
    print('='*60)
    t0 = time.time()
    resp = await service.chat('你好，你是谁？')
    t1 = time.time()
    print(f'  耗时: {t1-t0:.1f}s')
    print(f'  回复: {resp[:200]}')
    check('包含自我介绍', any(kw in resp for kw in ['旅行', '助手', '天气', '机票', '火车', 'SmartVoyage', '帮']))
    check('不包含查询数据', not any(kw in resp for kw in ['¥', '车次', '温度']))
    print()

    # ==================== 测试5：景点推荐 ====================
    print('='*60)
    print('测试5：景点推荐')
    print('='*60)
    t0 = time.time()
    resp = await service.chat('推荐几个北京的景点')
    t1 = time.time()
    print(f'  耗时: {t1-t0:.1f}s')
    print(f'  回复: {resp[:300]}')
    check('包含景点', any(kw in resp for kw in ['故宫', '长城', '天安门', '颐和园', '景点', '北京']))
    check('回复长度合理', len(resp) > 50, f'长度: {len(resp)}')
    print()

    # ==================== 测试6：租车查询 ====================
    print('='*60)
    print('测试6：租车查询（TripAssistant）')
    print('='*60)
    t0 = time.time()
    resp = await service.chat('有没有去丽江的旅游团')
    t1 = time.time()
    print(f'  耗时: {t1-t0:.1f}s')
    print(f'  回复: {resp[:300]}')
    check('包含旅游团信息', any(kw in resp for kw in ['旅游团', '丽江', '天数', '价格', '元', '游']))
    print()

    # ==================== 测试7：保险查询 ====================
    print('='*60)
    print('测试7：保险查询（TripAssistant）')
    print('='*60)
    t0 = time.time()
    resp = await service.chat('有什么旅行保险可以买')
    t1 = time.time()
    print(f'  耗时: {t1-t0:.1f}s')
    print(f'  回复: {resp[:300]}')
    check('包含保险信息', any(kw in resp for kw in ['保险', '保障', '意外', '元/份', '赔付']))
    print()

    # ==================== 测试8：记忆系统 - 多轮对话 ====================
    print('='*60)
    print('测试8：记忆系统 - 多轮对话上下文')
    print('='*60)
    service.clear_memory()
    # 第一轮：设定上下文
    resp1 = await service.chat('我打算8月1日从北京去成都')
    check('第一轮回复正常', len(resp1) > 10)
    # 第二轮：引用上下文（只说"天气"，期望能从历史中推断北京/成都）
    resp2 = await service.chat('那边天气怎么样')
    check('第二轮回复正常', len(resp2) > 10)
    check('第二轮包含天气信息', any(kw in resp2 for kw in ['天气', '温度', '°', '晴', '雨', '阴', '湿度', '成都', '北京']))
    print(f'  第一轮: {resp1[:100]}')
    print(f'  第二轮: {resp2[:100]}')
    print()

    # ==================== 测试9：记忆状态检查 ====================
    print('='*60)
    print('测试9：记忆状态检查')
    print('='*60)
    mem_state = service.get_memory_state()
    check('short_term_messages 非空', len(mem_state['short_term_messages']) > 0,
          f'实际: {len(mem_state["short_term_messages"])}')
    check('entity_history 有记录', len(mem_state['entity_history']) > 0,
          f'实际: {len(mem_state["entity_history"])}')
    print(f'  短期消息数: {len(mem_state["short_term_messages"])}')
    print(f'  实体历史数: {len(mem_state["entity_history"])}')
    print()

    # ==================== 测试10：流式输出 ====================
    print('='*60)
    print('测试10：流式输出 chat_stream()')
    print('='*60)
    service.clear_memory()
    t0 = time.time()
    chunks = []
    async for chunk in service.chat_stream('北京今天天气'):
        chunks.append(chunk)
    t1 = time.time()
    full_resp = ''.join(chunks)
    print(f'  耗时: {t1-t0:.1f}s')
    print(f'  片段数: {len(chunks)}')
    print(f'  回复: {full_resp[:200]}')
    check('流式输出有内容', len(full_resp) > 20)
    check('包含天气信息', any(kw in full_resp for kw in ['天气', '温度', '°', '晴', '阴', '雨', '北京']))
    print()

    # ==================== 结果 ====================
    print('='*60)
    print(f'测试结果: {passed} 通过, {failed} 失败')
    print('='*60)

asyncio.run(run_tests())
