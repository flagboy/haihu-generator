/**
 * ラベリングWebSocketクライアント
 */
export class LabelingWebSocket {
    constructor() {
        this.socket = null;
        this.sessionId = null;
        this.userName = null;
        this.eventHandlers = new Map();
        this.connected = false;
    }

    /**
     * WebSocket接続を初期化
     */
    connect(url = '') {
        if (this.socket && this.connected) {
            console.warn('WebSocket already connected');
            return;
        }

        // Socket.IOクライアントを使用
        this.socket = io(url || window.location.origin, {
            transports: ['websocket', 'polling']
        });

        this._registerDefaultHandlers();
        this.connected = true;
    }

    /**
     * デフォルトのイベントハンドラーを登録
     */
    _registerDefaultHandlers() {
        // 接続イベント
        this.socket.on('connect', () => {
            console.log('WebSocket connected', this.socket.id);
            this._emit('connected', { clientId: this.socket.id });
        });

        // 切断イベント
        this.socket.on('disconnect', (reason) => {
            console.log('WebSocket disconnected', reason);
            this.connected = false;
            this._emit('disconnected', { reason });
        });

        // エラーイベント
        this.socket.on('error', (error) => {
            console.error('WebSocket error', error);
            this._emit('error', { error });
        });

        // カスタムイベント
        this._setupCustomEventHandlers();
    }

    /**
     * カスタムイベントハンドラーを設定
     */
    _setupCustomEventHandlers() {
        // ユーザー参加通知
        this.socket.on('user_joined', (data) => {
            this._emit('userJoined', data);
        });

        // ユーザー退出通知
        this.socket.on('user_left', (data) => {
            this._emit('userLeft', data);
        });

        // 参加者リスト更新
        this.socket.on('participants_update', (data) => {
            this._emit('participantsUpdate', data);
        });

        // フレーム更新通知
        this.socket.on('frame_updated', (data) => {
            this._emit('frameUpdated', data);
        });

        // ラベル更新通知
        this.socket.on('label_updated', (data) => {
            this._emit('labelUpdated', data);
        });

        // 進捗更新通知
        this.socket.on('progress_updated', (data) => {
            this._emit('progressUpdated', data);
        });

        // 手牌領域更新通知
        this.socket.on('hand_area_updated', (data) => {
            this._emit('handAreaUpdated', data);
        });

        // 同期レスポンス
        this.socket.on('sync_response', (data) => {
            this._emit('syncResponse', data);
        });

        // チャットメッセージ
        this.socket.on('chat_message', (data) => {
            this._emit('chatMessage', data);
        });

        // カーソル移動
        this.socket.on('cursor_moved', (data) => {
            this._emit('cursorMoved', data);
        });
    }

    /**
     * セッションに参加
     */
    joinSession(sessionId, userName = 'Anonymous') {
        this.sessionId = sessionId;
        this.userName = userName;

        this.socket.emit('join_session', {
            session_id: sessionId,
            user_name: userName
        });
    }

    /**
     * セッションから退出
     */
    leaveSession() {
        if (this.sessionId) {
            this.socket.emit('leave_session');
            this.sessionId = null;
        }
    }

    /**
     * フレーム更新を通知
     */
    notifyFrameUpdate(frameNumber, timestamp) {
        this.socket.emit('frame_update', {
            frame_number: frameNumber,
            timestamp: timestamp
        });
    }

    /**
     * ラベル更新を通知
     */
    notifyLabelUpdate(frameNumber, player, tileIndex, label) {
        this.socket.emit('label_update', {
            frame_number: frameNumber,
            player: player,
            tile_index: tileIndex,
            label: label
        });
    }

    /**
     * 進捗更新を通知
     */
    notifyProgressUpdate(totalFrames, labeledFrames, currentFrame) {
        this.socket.emit('progress_update', {
            total_frames: totalFrames,
            labeled_frames: labeledFrames,
            current_frame: currentFrame
        });
    }

    /**
     * 手牌領域更新を通知
     */
    notifyHandAreaUpdate(player, area) {
        this.socket.emit('hand_area_update', {
            player: player,
            area: area
        });
    }

    /**
     * 同期をリクエスト
     */
    requestSync() {
        this.socket.emit('request_sync', {
            session_id: this.sessionId
        });
    }

    /**
     * チャットメッセージを送信
     */
    sendChatMessage(message) {
        this.socket.emit('chat_message', {
            message: message,
            timestamp: new Date().toISOString()
        });
    }

    /**
     * カーソル位置を共有
     */
    shareCursorPosition(x, y, element = null) {
        this.socket.emit('cursor_move', {
            x: x,
            y: y,
            element: element
        });
    }

    /**
     * イベントハンドラーを登録
     */
    on(event, handler) {
        if (!this.eventHandlers.has(event)) {
            this.eventHandlers.set(event, new Set());
        }
        this.eventHandlers.get(event).add(handler);
    }

    /**
     * イベントハンドラーを削除
     */
    off(event, handler) {
        if (this.eventHandlers.has(event)) {
            this.eventHandlers.get(event).delete(handler);
        }
    }

    /**
     * イベントを発火
     */
    _emit(event, data) {
        if (this.eventHandlers.has(event)) {
            this.eventHandlers.get(event).forEach(handler => {
                try {
                    handler(data);
                } catch (error) {
                    console.error(`Error in event handler for ${event}:`, error);
                }
            });
        }
    }

    /**
     * 接続を切断
     */
    disconnect() {
        if (this.socket) {
            this.leaveSession();
            this.socket.disconnect();
            this.socket = null;
            this.connected = false;
        }
    }

    /**
     * 接続状態を取得
     */
    isConnected() {
        return this.connected && this.socket && this.socket.connected;
    }

    /**
     * 再接続
     */
    reconnect() {
        if (this.socket) {
            this.socket.connect();
        }
    }
}

// シングルトンインスタンスをエクスポート
export const labelingWebSocket = new LabelingWebSocket();
