# server.py — FastAPI：将 main_process 封装为 REST / WebSocket，供树莓派、ROS 等端云分离调用
from __future__ import annotations

import base64
import json
import re
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import llm_api
from llm_api import clear_memory, get_response
from vision import no_input_vision, process_image


def _conversation_history() -> List[Dict[str, Any]]:
    """与 llm_api 内部 deque 同步（只读快照）。"""
    return [dict(m) for m in llm_api._message_history]


def _decode_image_bytes(raw: bytes) -> Optional[np.ndarray]:
    if not raw:
        return None
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _decode_base64_image(b64: Optional[str]) -> Optional[np.ndarray]:
    """支持裸 base64 或 data:image/...;base64, 前缀。"""
    if not b64 or not str(b64).strip():
        return None
    s = str(b64).strip()
    if "base64," in s:
        s = s.split("base64,", 1)[1]
    s = re.sub(r"\s+", "", s)
    try:
        raw = base64.standard_b64decode(s)
    except Exception:
        return None
    return _decode_image_bytes(raw)


def main_process(image: Optional[np.ndarray], user_text: str) -> Dict[str, Any]:
    """
    与 Gradio 主流程对齐（无麦克风）：VLM 画面描述 + 用户文本 → LLM 决策。
    image: BGR numpy，可为 None。
    """
    if image is None:
        vision_result = no_input_vision()
    else:
        vision_result = process_image(image)

    desc = (vision_result.get("description") or "").strip()
    llm_result, _ = get_response(
        "neutral",
        user_text,
        enable_tts=False,
        vision_desc=desc,
    )

    return {
        "robot": {
            "emotion": llm_result.get("emotion"),
            "action": llm_result.get("action"),
            "reply": llm_result.get("reply"),
        },
        "pipeline": {
            "perception": llm_result.get("perception"),
            "decision": llm_result.get("decision"),
            "execution": llm_result.get("execution"),
        },
        "vision": {
            "success": vision_result.get("success"),
            "description": vision_result.get("description"),
            "is_fallback": vision_result.get("is_fallback"),
        },
        "history": _conversation_history(),
    }


# ---------------------------------------------------------------------------
app = FastAPI(
    title="暖暖多模态 API",
    version="2.0.0",
    description="REST + WebSocket：硬件 POST 或长连接即可获取 robot 决策，实现端云分离。",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatJsonBody(BaseModel):
    """JSON 体：适合 ROS / HTTP 客户端直接序列化。"""

    text: str = Field("", description="用户说的话（可与麦克风 STT 对齐）")
    image_base64: Optional[str] = Field(
        None, description="可选 JPEG/PNG 的 base64；可带 data:image/jpeg;base64, 前缀"
    )


@app.get("/")
async def root():
    return {
        "service": "emotion_aware_system",
        "endpoints": {
            "health": "GET /health",
            "chat_multipart": "POST /api/chat  (form: text, image?)",
            "chat_json": "POST /api/chat/json  (JSON: text, image_base64?)",
            "memory_clear": "POST /api/memory/clear",
            "websocket": "WS /ws/chat  (JSON 文本帧)",
        },
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/memory/clear")
async def api_memory_clear():
    """清空服务端对话记忆（与 Gradio「清除记忆」一致）。"""
    clear_memory()
    return {"ok": True}


@app.post("/api/chat")
async def api_chat(
    text: str = Form(""),
    image: Optional[UploadFile] = File(None),
):
    """
    multipart/form-data：`text` + 可选 `image` 文件。
    树莓派示例：`curl -F "text=你好" -F "image=@cap.jpg" http://<host>:8000/api/chat`
    """
    frame = None
    if image is not None:
        raw = await image.read()
        frame = _decode_image_bytes(raw)
    return main_process(frame, text)


@app.post("/api/chat/json")
async def api_chat_json(body: ChatJsonBody):
    """
    application/json：纯 HTTP 即可集成 ROS / Python requests，无需 multipart。
    """
    frame = _decode_base64_image(body.image_base64)
    return main_process(frame, body.text)


@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    """
    WebSocket：每帧一条 JSON 文本消息，格式与 ChatJsonBody 相同；
    服务端以 JSON 返回与 REST 相同的 `main_process` 结构，便于长会话或降低连接开销。
    """
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json(
                    {"error": "invalid_json", "hint": "send JSON: {text, image_base64?}"}
                )
                continue
            text = (data.get("text") or "") if isinstance(data, dict) else ""
            b64 = data.get("image_base64") if isinstance(data, dict) else None
            frame = _decode_base64_image(b64)
            out = main_process(frame, str(text))
            await websocket.send_json(out)
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
