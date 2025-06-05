"""
WebSocket通信モジュール
リアルタイムのラベリング進捗共有
"""

from typing import Any

from flask import request
from flask_socketio import SocketIO, emit, join_room, leave_room
from loguru import logger

# SocketIOインスタンス（app.pyで初期化される）
socketio = None

# 接続中のクライアント管理
connected_clients: dict[str, dict[str, Any]] = {}


def init_socketio(app):
    """SocketIOを初期化"""
    global socketio
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

    # イベントハンドラーを登録
    register_handlers()

    return socketio


def register_handlers():
    """WebSocketイベントハンドラーを登録"""

    @socketio.on("connect")
    def handle_connect():
        """クライアント接続時"""
        client_id = request.sid
        connected_clients[client_id] = {"session_id": None, "user_name": None, "connected_at": None}
        logger.info(f"クライアント接続: {client_id}")
        emit("connected", {"client_id": client_id})

    @socketio.on("disconnect")
    def handle_disconnect():
        """クライアント切断時"""
        client_id = request.sid
        if client_id in connected_clients:
            session_id = connected_clients[client_id].get("session_id")
            if session_id:
                leave_room(session_id)
                emit(
                    "user_left",
                    {
                        "client_id": client_id,
                        "user_name": connected_clients[client_id].get("user_name"),
                    },
                    room=session_id,
                )
            del connected_clients[client_id]
        logger.info(f"クライアント切断: {client_id}")

    @socketio.on("join_session")
    def handle_join_session(data):
        """セッションに参加"""
        client_id = request.sid
        session_id = data.get("session_id")
        user_name = data.get("user_name", "Anonymous")

        if client_id in connected_clients:
            # 以前のルームから退出
            old_session = connected_clients[client_id].get("session_id")
            if old_session:
                leave_room(old_session)

            # 新しいルームに参加
            join_room(session_id)
            connected_clients[client_id].update({"session_id": session_id, "user_name": user_name})

            # 他のユーザーに通知
            emit(
                "user_joined",
                {"client_id": client_id, "user_name": user_name},
                room=session_id,
                exclude_self=True,
            )

            # 現在の参加者リストを送信
            participants = get_session_participants(session_id)
            emit("participants_update", {"participants": participants}, room=session_id)

            logger.info(f"セッション参加: {user_name} -> {session_id}")

    @socketio.on("leave_session")
    def handle_leave_session():
        """セッションから退出"""
        client_id = request.sid
        if client_id in connected_clients:
            session_id = connected_clients[client_id].get("session_id")
            if session_id:
                leave_room(session_id)
                emit(
                    "user_left",
                    {
                        "client_id": client_id,
                        "user_name": connected_clients[client_id].get("user_name"),
                    },
                    room=session_id,
                )
                connected_clients[client_id]["session_id"] = None

    @socketio.on("frame_update")
    def handle_frame_update(data):
        """フレーム更新の通知"""
        client_id = request.sid
        session_id = connected_clients.get(client_id, {}).get("session_id")

        if session_id:
            emit(
                "frame_updated",
                {
                    "frame_number": data.get("frame_number"),
                    "timestamp": data.get("timestamp"),
                    "updated_by": connected_clients[client_id].get("user_name"),
                },
                room=session_id,
                exclude_self=True,
            )

    @socketio.on("label_update")
    def handle_label_update(data):
        """ラベル更新の通知"""
        client_id = request.sid
        session_id = connected_clients.get(client_id, {}).get("session_id")

        if session_id:
            emit(
                "label_updated",
                {
                    "frame_number": data.get("frame_number"),
                    "player": data.get("player"),
                    "tile_index": data.get("tile_index"),
                    "label": data.get("label"),
                    "updated_by": connected_clients[client_id].get("user_name"),
                },
                room=session_id,
                exclude_self=True,
            )

    @socketio.on("progress_update")
    def handle_progress_update(data):
        """進捗更新の通知"""
        client_id = request.sid
        session_id = connected_clients.get(client_id, {}).get("session_id")

        if session_id:
            emit(
                "progress_updated",
                {
                    "total_frames": data.get("total_frames"),
                    "labeled_frames": data.get("labeled_frames"),
                    "current_frame": data.get("current_frame"),
                    "updated_by": connected_clients[client_id].get("user_name"),
                },
                room=session_id,
            )

    @socketio.on("hand_area_update")
    def handle_hand_area_update(data):
        """手牌領域更新の通知"""
        client_id = request.sid
        session_id = connected_clients.get(client_id, {}).get("session_id")

        if session_id:
            emit(
                "hand_area_updated",
                {
                    "player": data.get("player"),
                    "area": data.get("area"),
                    "updated_by": connected_clients[client_id].get("user_name"),
                },
                room=session_id,
                exclude_self=True,
            )

    @socketio.on("request_sync")
    def handle_request_sync(data):
        """同期リクエスト"""
        client_id = request.sid
        session_id = connected_clients.get(client_id, {}).get("session_id")

        if session_id:
            # セッションの最新状態を要求元に送信
            emit(
                "sync_response",
                {
                    "session_id": session_id,
                    "participants": get_session_participants(session_id),
                    "requested_by": client_id,
                },
                to=client_id,
            )

    @socketio.on("chat_message")
    def handle_chat_message(data):
        """チャットメッセージ"""
        client_id = request.sid
        session_id = connected_clients.get(client_id, {}).get("session_id")

        if session_id:
            emit(
                "chat_message",
                {
                    "message": data.get("message"),
                    "sender": connected_clients[client_id].get("user_name"),
                    "timestamp": data.get("timestamp"),
                },
                room=session_id,
            )

    @socketio.on("cursor_move")
    def handle_cursor_move(data):
        """カーソル位置の共有（協調作業用）"""
        client_id = request.sid
        session_id = connected_clients.get(client_id, {}).get("session_id")

        if session_id:
            emit(
                "cursor_moved",
                {
                    "client_id": client_id,
                    "user_name": connected_clients[client_id].get("user_name"),
                    "x": data.get("x"),
                    "y": data.get("y"),
                    "element": data.get("element"),
                },
                room=session_id,
                exclude_self=True,
            )


def get_session_participants(session_id: str) -> list:
    """セッションの参加者リストを取得"""
    participants = []
    for client_id, client_info in connected_clients.items():
        if client_info.get("session_id") == session_id:
            participants.append(
                {
                    "client_id": client_id,
                    "user_name": client_info.get("user_name"),
                    "connected_at": client_info.get("connected_at"),
                }
            )
    return participants


def broadcast_to_session(session_id: str, event: str, data: dict[str, Any]):
    """セッション内の全クライアントにブロードキャスト"""
    if socketio:
        socketio.emit(event, data, room=session_id)


def send_to_client(client_id: str, event: str, data: dict[str, Any]):
    """特定のクライアントに送信"""
    if socketio:
        socketio.emit(event, data, to=client_id)


def get_active_sessions() -> dict[str, list]:
    """アクティブなセッション一覧を取得"""
    sessions = {}
    for client_info in connected_clients.values():
        session_id = client_info.get("session_id")
        if session_id:
            if session_id not in sessions:
                sessions[session_id] = []
            sessions[session_id].append(
                {
                    "user_name": client_info.get("user_name"),
                    "connected_at": client_info.get("connected_at"),
                }
            )
    return sessions
