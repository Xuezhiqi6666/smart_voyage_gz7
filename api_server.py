"""
需求：SmartVoyage FastAPI后端服务器，提供REST API接口
"""
import asyncio
import json
import os
import sys
import subprocess
import signal
import threading
import time
import atexit
import socket
import re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

from chat_service import ChatService

app = FastAPI(title="SmartVoyage API", description="基于A2A的旅行智能助手")

# 全局服务实例
chat_service = ChatService()


class ChatRequest(BaseModel):
    message: str


class ProfileRequest(BaseModel):
    profile: dict


@app.get("/")
async def index():
    """返回前端页面"""
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """发送消息，获取回复"""
    response = await chat_service.chat(request.message)
    return {"status": "success", "message": response}


async def sse_generator(message: str):
    """SSE 生成器，逐字流式返回回复，处理期间发送心跳保活"""
    queue = asyncio.Queue()

    async def producer():
        """运行 chat_stream，将结果逐块放入队列"""
        try:
            async for text_chunk in chat_service.chat_stream(message):
                await queue.put(text_chunk)
        finally:
            await queue.put(None)  # 结束信号

    # 在后台运行 producer
    producer_task = asyncio.create_task(producer())

    while True:
        try:
            # 等待下一个 chunk，最多等 10 秒
            chunk = await asyncio.wait_for(queue.get(), timeout=10)
            if chunk is None:  # producer 完成
                break
            yield f"data: {json.dumps({'chunk': chunk}, ensure_ascii=False)}\n\n"
        except asyncio.TimeoutError:
            # 心跳：SSE 注释行（以 : 开头），客户端忽略但保持连接不断
            yield ": heartbeat\n\n"

    # 确保 producer 已完成
    await producer_task

    # 发送结束标记
    yield "data: [DONE]\n\n"


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """发送消息，流式获取回复（SSE）"""
    return StreamingResponse(sse_generator(request.message), media_type="text/event-stream")


@app.get("/api/memory")
async def get_memory():
    """获取记忆状态"""
    return {"status": "success", "data": chat_service.get_memory_state()}


@app.post("/api/memory/clear")
async def clear_memory():
    """清空记忆"""
    chat_service.clear_memory()
    return {"status": "success", "message": "记忆已清空"}


@app.post("/api/memory/profile")
async def update_profile(request: ProfileRequest):
    """更新用户偏好"""
    chat_service.update_user_profile(request.profile)
    return {"status": "success", "message": "用户偏好已更新"}


@app.get("/api/agents")
async def get_agents():
    """获取代理卡片信息"""
    return {"status": "success", "data": chat_service.get_agent_cards()}


# ==================== 子服务自动管理 ====================
# 启动 api_server.py 时自动拉起所有 MCP 工具服务和 A2A 代理服务，
# 退出时自动清理所有子进程，避免残留僵尸进程。

_SUB_PROCESSES = []
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# (服务名称, 相对脚本路径) —— 顺序：先 MCP 工具层，再 A2A 代理层
_SUB_SERVICES = [
    ("MCP-Ticket  :8001", "mcp_server/mcp_ticket_server.py"),
    ("MCP-Weather :8002", "mcp_server/mcp_weather_server.py"),
    ("MCP-Trip    :8003", "mcp_server/mcp_trip_server.py"),
    ("A2A-Weather :5005", "a2a_server/weather_server.py"),
    ("A2A-Ticket  :5006", "a2a_server/ticket_server.py"),
    ("A2A-Trip    :5007", "a2a_server/trip_server.py"),
]


def _drain_pipe(proc, name):
    """后台线程：持续排空子进程的 stderr，防止管道缓冲区满导致子进程阻塞"""
    try:
        for line in proc.stderr:
            pass  # 丢弃输出，防止管道堆积
    except Exception:
        pass  # 进程退出时 pipe 关闭，忽略异常


def _start_sub_services():
    """启动所有子服务进程，先启动 MCP 工具层，再启动 A2A 代理层"""
    python_exe = sys.executable  # 使用当前 Python 解释器（与 conda 环境一致）

    # 分层启动：MCP 工具层先启动（A2A 代理层依赖它们）
    mcp_services = [(n, s) for n, s in _SUB_SERVICES if s.startswith("mcp_server/")]
    a2a_services = [(n, s) for n, s in _SUB_SERVICES if s.startswith("a2a_server/")]

    for layer_name, services in [("MCP工具层", mcp_services), ("A2A代理层", a2a_services)]:
        print(f"\n  --- {layer_name} ---")
        for name, script in services:
            script_path = os.path.join(_PROJECT_DIR, script)
            if not os.path.exists(script_path):
                print(f"  [跳过] {name} - 文件不存在: {script_path}")
                continue
            # 创建新进程组（Windows 上 CREATE_NEW_PROCESS_GROUP），方便批量终止
            creationflags = 0
            if sys.platform == "win32":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
            proc = subprocess.Popen(
                [python_exe, script_path],
                cwd=_PROJECT_DIR,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                creationflags=creationflags,
            )
            # 启动后台线程排空 stderr，防止管道缓冲区满导致子进程阻塞
            threading.Thread(target=_drain_pipe, args=(proc, name), daemon=True).start()
            _SUB_PROCESSES.append((name, proc))
            print(f"  [启动] {name}  (PID: {proc.pid})")
        # MCP 层启动后等待一段时间，确保 A2A 层能连接到 MCP
        wait = 3 if layer_name == "MCP工具层" else 2
        time.sleep(wait)

    # 健康检查：确认所有子进程仍在运行，且端口已在监听
    print(f"\n  --- 健康检查 ---")
    failed = []
    for name, proc in _SUB_PROCESSES:
        # 从服务名中提取端口号（如 "MCP-Ticket  :8001" → 8001）
        port_match = re.search(r':(\d+)', name)
        port = int(port_match.group(1)) if port_match else None

        if proc.poll() is not None:  # 进程已退出
            print(f"  [失败] {name} (退出码: {proc.returncode})")
            failed.append(name)
        elif port:
            # 进程存活，检查端口是否就绪（最多重试 10 次，每次等 1 秒）
            port_ready = False
            for _ in range(10):
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(1)
                    result = sock.connect_ex(("127.0.0.1", port))
                    sock.close()
                    if result == 0:
                        port_ready = True
                        break
                except Exception:
                    pass
                time.sleep(1)
            if port_ready:
                print(f"  [正常] {name}  (PID: {proc.pid}, 端口 {port} 已监听)")
            else:
                print(f"  [警告] {name}  (PID: {proc.pid}, 端口 {port} 未监听，可能仍在初始化)")
                failed.append(name)
        else:
            print(f"  [正常] {name}  (PID: {proc.pid})")

    if failed:
        try:
            print(f"\n  [WARN] {len(failed)} 个子服务启动失败或未就绪: {', '.join(failed)}")
            print(f"    相关功能可能不可用，但主服务仍会运行。\n")
        except UnicodeEncodeError:
            print(f"\n  [WARN] {len(failed)} sub-services failed: {', '.join(failed)}\n")
    else:
        print(f"  所有子服务运行正常（共 {len(_SUB_PROCESSES)} 个）\n")


def _stop_sub_services():
    """停止所有子服务进程"""
    if not _SUB_PROCESSES:
        return
    print("\n正在停止子服务...")
    for name, proc in _SUB_PROCESSES:
        if proc.poll() is None:  # 进程仍在运行
            try:
                proc.terminate()
                proc.wait(timeout=3)
                print(f"  [停止] {name}")
            except subprocess.TimeoutExpired:
                proc.kill()
                print(f"  [强制停止] {name}")
            except Exception as e:
                print(f"  [停止失败] {name}: {e}")
    _SUB_PROCESSES.clear()


if __name__ == "__main__":
    print("=" * 50)
    print("SmartVoyage 正在启动子服务...")
    print("=" * 50)
    _start_sub_services()

    # 注册退出时的清理回调
    atexit.register(_stop_sub_services)

    def _signal_handler(signum, frame):
        _stop_sub_services()
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    print("启动主 API 服务器 (端口 8080)...")
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="info")
