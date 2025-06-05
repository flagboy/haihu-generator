/**
 * 統合されたラベリングアプリケーション
 */

import { LabelingAPIClient } from './modules/api/LabelingAPIClient.js';
import { LabelingWebSocket } from './modules/websocket/LabelingWebSocket.js';
import { CanvasManager } from './modules/canvas/CanvasManager.js';
import { HandAreaManager } from './modules/managers/HandAreaManager.js';
import { AnnotationManager } from './modules/managers/AnnotationManager.js';
import { AppState } from './modules/state/AppState.js';
import { KeyboardShortcutManager } from './modules/utils/KeyboardShortcutManager.js';
import { NotificationManager } from './modules/utils/NotificationManager.js';

class LabelingApp {
    constructor() {
        // 状態管理
        this.state = new AppState();

        // API/WebSocketクライアント
        this.api = new LabelingAPIClient();
        this.websocket = new LabelingWebSocket();

        // マネージャー
        this.canvas = null;
        this.handAreaManager = null;
        this.annotationManager = null;
        this.keyboardManager = new KeyboardShortcutManager();
        this.notifications = new NotificationManager();

        // セッション情報
        this.sessionId = null;
        this.currentFrame = 0;
        this.totalFrames = 0;

        // UI要素
        this.elements = {};
    }

    /**
     * アプリケーションを初期化
     */
    async init() {
        try {
            // UI要素を取得
            this._getUIElements();

            // キャンバスを初期化
            this._initCanvas();

            // WebSocketを接続
            this._initWebSocket();

            // イベントリスナーを設定
            this._setupEventListeners();

            // キーボードショートカットを設定
            this._setupKeyboardShortcuts();

            // 既存のセッション一覧を取得
            await this._loadSessions();

            this.notifications.success('ラベリングシステムを初期化しました');
        } catch (error) {
            console.error('初期化エラー:', error);
            this.notifications.error('初期化に失敗しました: ' + error.message);
        }
    }

    /**
     * UI要素を取得
     */
    _getUIElements() {
        this.elements = {
            // キャンバス
            canvas: document.getElementById('labeling-canvas'),

            // 動画制御
            videoSelector: document.getElementById('video-selector'),
            sessionSelector: document.getElementById('session-selector'),
            createSessionBtn: document.getElementById('create-session-btn'),

            // フレーム制御
            frameSlider: document.getElementById('frame-slider'),
            frameNumber: document.getElementById('frame-number'),
            prevFrameBtn: document.getElementById('prev-frame-btn'),
            nextFrameBtn: document.getElementById('next-frame-btn'),

            // プレイヤー選択
            playerSelector: document.getElementById('player-selector'),

            // 手牌領域設定
            setHandAreaBtn: document.getElementById('set-hand-area-btn'),
            autoDetectBtn: document.getElementById('auto-detect-btn'),

            // ラベリング
            tileButtons: document.querySelectorAll('.tile-button'),
            autoLabelBtn: document.getElementById('auto-label-btn'),

            // 進捗表示
            progressBar: document.getElementById('progress-bar'),
            progressText: document.getElementById('progress-text'),

            // その他
            exportBtn: document.getElementById('export-btn'),
            participantsList: document.getElementById('participants-list')
        };
    }

    /**
     * キャンバスを初期化
     */
    _initCanvas() {
        this.canvas = new CanvasManager(this.elements.canvas);
        this.handAreaManager = new HandAreaManager(this.state, this.canvas, this.api);
        this.annotationManager = new AnnotationManager(this.state, this.api);
    }

    /**
     * WebSocketを初期化
     */
    _initWebSocket() {
        this.websocket.connect();

        // WebSocketイベントハンドラー
        this.websocket.on('connected', () => {
            console.log('WebSocket接続完了');
        });

        this.websocket.on('frameUpdated', (data) => {
            if (data.frame_number !== this.currentFrame) {
                this.notifications.info(`${data.updated_by}がフレーム${data.frame_number}を更新しました`);
            }
        });

        this.websocket.on('labelUpdated', (data) => {
            this._handleRemoteLabelUpdate(data);
        });

        this.websocket.on('progressUpdated', (data) => {
            this._updateProgress(data.labeled_frames, data.total_frames);
        });

        this.websocket.on('participantsUpdate', (data) => {
            this._updateParticipantsList(data.participants);
        });
    }

    /**
     * イベントリスナーを設定
     */
    _setupEventListeners() {
        // セッション作成
        this.elements.createSessionBtn?.addEventListener('click', () => this._createSession());

        // セッション選択
        this.elements.sessionSelector?.addEventListener('change', (e) => {
            if (e.target.value) {
                this._loadSession(e.target.value);
            }
        });

        // フレーム制御
        this.elements.prevFrameBtn?.addEventListener('click', () => this._navigateFrame(-1));
        this.elements.nextFrameBtn?.addEventListener('click', () => this._navigateFrame(1));
        this.elements.frameSlider?.addEventListener('input', (e) => {
            this._loadFrame(parseInt(e.target.value));
        });

        // プレイヤー選択
        this.elements.playerSelector?.addEventListener('change', (e) => {
            this.state.currentPlayer = e.target.value;
            this._refreshTiles();
        });

        // 手牌領域設定
        this.elements.setHandAreaBtn?.addEventListener('click', () => {
            this.handAreaManager.startAreaSelection(this.state.currentPlayer);
        });

        this.elements.autoDetectBtn?.addEventListener('click', () => {
            this._autoDetectHandAreas();
        });

        // 牌ボタン
        this.elements.tileButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                const label = e.target.dataset.tile;
                this._labelCurrentTile(label);
            });
        });

        // 自動ラベリング
        this.elements.autoLabelBtn?.addEventListener('click', () => {
            this._autoLabelCurrentFrame();
        });

        // エクスポート
        this.elements.exportBtn?.addEventListener('click', () => {
            this._exportAnnotations();
        });
    }

    /**
     * キーボードショートカットを設定
     */
    _setupKeyboardShortcuts() {
        // 数字キー: 牌の数字
        for (let i = 1; i <= 9; i++) {
            this.keyboardManager.register(i.toString(), () => {
                this._labelWithNumber(i);
            });
        }

        // 牌種切り替え
        this.keyboardManager.register('q', () => this._setTileType('m')); // 萬子
        this.keyboardManager.register('w', () => this._setTileType('p')); // 筒子
        this.keyboardManager.register('e', () => this._setTileType('s')); // 索子

        // 字牌
        this.keyboardManager.register('a', () => this._labelCurrentTile('1z')); // 東
        this.keyboardManager.register('s', () => this._labelCurrentTile('2z')); // 南
        this.keyboardManager.register('d', () => this._labelCurrentTile('3z')); // 西
        this.keyboardManager.register('f', () => this._labelCurrentTile('4z')); // 北
        this.keyboardManager.register('g', () => this._labelCurrentTile('5z')); // 白
        this.keyboardManager.register('h', () => this._labelCurrentTile('6z')); // 發
        this.keyboardManager.register('j', () => this._labelCurrentTile('7z')); // 中

        // ナビゲーション
        this.keyboardManager.register('ArrowLeft', () => this._navigateTile(-1));
        this.keyboardManager.register('ArrowRight', () => this._navigateTile(1));
        this.keyboardManager.register('ArrowUp', () => this._navigateFrame(-1));
        this.keyboardManager.register('ArrowDown', () => this._navigateFrame(1));

        // その他
        this.keyboardManager.register('Enter', () => this._confirmCurrentFrame());
        this.keyboardManager.register('Escape', () => this._cancelCurrentOperation());
    }

    /**
     * セッション一覧を読み込み
     */
    async _loadSessions() {
        try {
            const sessions = await this.api.listSessions();

            // セレクトボックスを更新
            if (this.elements.sessionSelector) {
                this.elements.sessionSelector.innerHTML = '<option value="">セッションを選択...</option>';
                sessions.forEach(session => {
                    const option = document.createElement('option');
                    option.value = session.session_id;
                    option.textContent = `${session.video_path} (${session.updated_at})`;
                    this.elements.sessionSelector.appendChild(option);
                });
            }
        } catch (error) {
            console.error('セッション一覧の取得に失敗:', error);
        }
    }

    /**
     * 新規セッションを作成
     */
    async _createSession() {
        const videoPath = this.elements.videoSelector?.value;
        if (!videoPath) {
            this.notifications.warning('動画を選択してください');
            return;
        }

        try {
            const result = await this.api.createSession(videoPath);
            this.sessionId = result.session_id;

            // WebSocketでセッションに参加
            const userName = prompt('ユーザー名を入力してください:', 'ユーザー');
            this.websocket.joinSession(this.sessionId, userName || 'Anonymous');

            // セッション情報を更新
            this.totalFrames = result.video_info.frame_count;
            this._updateProgress(0, this.totalFrames);

            // フレームスライダーを設定
            if (this.elements.frameSlider) {
                this.elements.frameSlider.max = this.totalFrames - 1;
                this.elements.frameSlider.value = 0;
            }

            // 最初のフレームを読み込み
            await this._loadFrame(0);

            this.notifications.success('新規セッションを作成しました');
        } catch (error) {
            console.error('セッション作成エラー:', error);
            this.notifications.error('セッション作成に失敗しました: ' + error.message);
        }
    }

    /**
     * セッションを読み込み
     */
    async _loadSession(sessionId) {
        try {
            const session = await this.api.getSession(sessionId);
            this.sessionId = sessionId;

            // WebSocketでセッションに参加
            const userName = prompt('ユーザー名を入力してください:', 'ユーザー');
            this.websocket.joinSession(sessionId, userName || 'Anonymous');

            // セッション情報を更新
            this.totalFrames = session.progress.total_frames;
            this.currentFrame = session.progress.current_frame;
            this._updateProgress(session.progress.labeled_frames, this.totalFrames);

            // 手牌領域を読み込み
            const handAreas = await this.api.getHandAreas(sessionId);
            this.state.handAreas = handAreas;

            // 現在のフレームを読み込み
            await this._loadFrame(this.currentFrame);

            this.notifications.success('セッションを読み込みました');
        } catch (error) {
            console.error('セッション読み込みエラー:', error);
            this.notifications.error('セッション読み込みに失敗しました: ' + error.message);
        }
    }

    /**
     * フレームを読み込み
     */
    async _loadFrame(frameNumber) {
        if (!this.sessionId) return;

        try {
            // フレーム画像を取得
            const frameBlob = await this.api.getFrame(this.sessionId, frameNumber);
            const frameUrl = URL.createObjectURL(frameBlob);

            // キャンバスに表示
            await this.canvas.loadImage(frameUrl);

            // 手牌領域を描画
            this.handAreaManager.drawAllAreas();

            // 現在のフレーム情報を更新
            this.currentFrame = frameNumber;
            if (this.elements.frameNumber) {
                this.elements.frameNumber.textContent = `${frameNumber + 1} / ${this.totalFrames}`;
            }
            if (this.elements.frameSlider) {
                this.elements.frameSlider.value = frameNumber;
            }

            // 牌を分割して表示
            await this._refreshTiles();

            // WebSocketで通知
            this.websocket.notifyFrameUpdate(frameNumber, Date.now());

        } catch (error) {
            console.error('フレーム読み込みエラー:', error);
            this.notifications.error('フレーム読み込みに失敗しました');
        }
    }

    /**
     * 牌を更新
     */
    async _refreshTiles() {
        if (!this.sessionId) return;

        const player = this.state.currentPlayer;

        try {
            // 牌を分割
            const result = await this.api.splitTiles(this.sessionId, this.currentFrame, player);

            // 牌の位置情報を保存
            this.state.currentTiles = result.positions;
            this.state.currentTileIndex = 0;

            // 最初の牌を選択
            this._selectTile(0);

        } catch (error) {
            console.error('牌分割エラー:', error);
        }
    }

    /**
     * フレームをナビゲート
     */
    _navigateFrame(direction) {
        const newFrame = Math.max(0, Math.min(this.totalFrames - 1, this.currentFrame + direction));
        if (newFrame !== this.currentFrame) {
            this._loadFrame(newFrame);
        }
    }

    /**
     * 牌をナビゲート
     */
    _navigateTile(direction) {
        const tiles = this.state.currentTiles;
        if (!tiles || tiles.length === 0) return;

        const newIndex = Math.max(0, Math.min(tiles.length - 1, this.state.currentTileIndex + direction));
        this._selectTile(newIndex);
    }

    /**
     * 牌を選択
     */
    _selectTile(index) {
        const tiles = this.state.currentTiles;
        if (!tiles || index < 0 || index >= tiles.length) return;

        this.state.currentTileIndex = index;

        // キャンバスでハイライト
        const tile = tiles[index];
        this.canvas.highlightRegion(tile.x, tile.y, tile.w, tile.h);
    }

    /**
     * 現在の牌にラベルを付ける
     */
    async _labelCurrentTile(label) {
        if (!this.sessionId || !this.state.currentTiles) return;

        const tileIndex = this.state.currentTileIndex;
        const tile = this.state.currentTiles[tileIndex];

        // ラベルを設定
        tile.label = label;

        // WebSocketで通知
        this.websocket.notifyLabelUpdate(
            this.currentFrame,
            this.state.currentPlayer,
            tileIndex,
            label
        );

        // 次の牌へ移動
        if (tileIndex < this.state.currentTiles.length - 1) {
            this._navigateTile(1);
        }
    }

    /**
     * 数字でラベル付け
     */
    _labelWithNumber(number) {
        const tileType = this.state.currentTileType || 'm';
        this._labelCurrentTile(`${number}${tileType}`);
    }

    /**
     * 牌種を設定
     */
    _setTileType(type) {
        this.state.currentTileType = type;
        this.notifications.info(`牌種: ${type === 'm' ? '萬子' : type === 'p' ? '筒子' : '索子'}`);
    }

    /**
     * 現在のフレームを確定
     */
    async _confirmCurrentFrame() {
        if (!this.sessionId || !this.state.currentTiles) return;

        try {
            // ラベル付けされた牌を収集
            const labeledTiles = this.state.currentTiles
                .filter(tile => tile.label)
                .map((tile, index) => ({
                    index,
                    label: tile.label,
                    x: tile.x,
                    y: tile.y,
                    w: tile.w,
                    h: tile.h,
                    confidence: 1.0
                }));

            // アノテーションを保存
            await this.api.addAnnotation(
                this.sessionId,
                this.currentFrame,
                this.state.currentPlayer,
                labeledTiles
            );

            // 進捗を更新
            const progress = await this.api.getSession(this.sessionId);
            this._updateProgress(progress.progress.labeled_frames, this.totalFrames);

            // 次のフレームへ
            this._navigateFrame(1);

            this.notifications.success('フレームを保存しました');
        } catch (error) {
            console.error('保存エラー:', error);
            this.notifications.error('保存に失敗しました');
        }
    }

    /**
     * 自動ラベリング
     */
    async _autoLabelCurrentFrame() {
        // TODO: AI自動ラベリング機能の実装
        this.notifications.info('自動ラベリング機能は開発中です');
    }

    /**
     * 手牌領域を自動検出
     */
    async _autoDetectHandAreas() {
        // TODO: 自動検出機能の実装
        this.notifications.info('自動検出機能は開発中です');
    }

    /**
     * アノテーションをエクスポート
     */
    async _exportAnnotations() {
        if (!this.sessionId) return;

        try {
            const format = prompt('エクスポート形式を選択してください (coco/yolo/tenhou):', 'coco');
            if (!format) return;

            const data = await this.api.exportAnnotations(this.sessionId, format);

            // ダウンロード
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `annotations_${this.sessionId}_${format}.json`;
            a.click();

            this.notifications.success(`${format}形式でエクスポートしました`);
        } catch (error) {
            console.error('エクスポートエラー:', error);
            this.notifications.error('エクスポートに失敗しました');
        }
    }

    /**
     * 進捗を更新
     */
    _updateProgress(labeled, total) {
        const percentage = total > 0 ? (labeled / total * 100) : 0;

        if (this.elements.progressBar) {
            this.elements.progressBar.style.width = percentage + '%';
        }
        if (this.elements.progressText) {
            this.elements.progressText.textContent = `${labeled} / ${total} (${percentage.toFixed(1)}%)`;
        }
    }

    /**
     * 参加者リストを更新
     */
    _updateParticipantsList(participants) {
        if (!this.elements.participantsList) return;

        this.elements.participantsList.innerHTML = '';
        participants.forEach(p => {
            const li = document.createElement('li');
            li.textContent = p.user_name;
            this.elements.participantsList.appendChild(li);
        });
    }

    /**
     * リモートラベル更新を処理
     */
    _handleRemoteLabelUpdate(data) {
        // 他のユーザーがラベルを更新した場合の処理
        if (data.frame_number === this.currentFrame &&
            data.player === this.state.currentPlayer) {
            // 現在表示中のフレームが更新された場合は再読み込み
            this._refreshTiles();
        }
    }

    /**
     * 現在の操作をキャンセル
     */
    _cancelCurrentOperation() {
        // 手牌領域設定モードをキャンセル
        if (this.handAreaManager.isSelecting) {
            this.handAreaManager.cancelSelection();
        }
    }
}

// アプリケーションを起動
document.addEventListener('DOMContentLoaded', () => {
    const app = new LabelingApp();
    app.init();

    // グローバルに公開（デバッグ用）
    window.labelingApp = app;
});
