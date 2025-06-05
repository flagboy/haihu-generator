/**
 * アプリケーション状態管理クラス
 */
export class AppState {
    constructor() {
        // キャンバス関連
        this.canvas = null;
        this.ctx = null;
        this.currentImage = null;
        this.zoomLevel = 1;
        this.panX = 0;
        this.panY = 0;

        // 動画・フレーム関連
        this.currentVideo = null;
        this.currentFrame = null;
        this.frameList = [];
        this.currentFrameIndex = -1;

        // アノテーション関連
        this.annotations = [];
        this.selectedTileType = null;

        // 描画状態
        this.isDrawing = false;
        this.startX = 0;
        this.startY = 0;
        this.currentBbox = null;

        // セッション関連
        this.sessionId = this._generateSessionId();

        // プレイヤー・手牌領域関連
        this.currentPlayer = 'bottom';
        this.handAreas = {
            bottom: null,
            top: null,
            left: null,
            right: null
        };
        this.isSettingHandArea = false;

        // イベントリスナー
        this._listeners = {};
    }

    /**
     * セッションIDを生成
     */
    _generateSessionId() {
        return `labeling_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    /**
     * 状態を更新
     */
    update(key, value) {
        const oldValue = this[key];
        this[key] = value;
        this._notify('stateChanged', { key, oldValue, newValue: value });
    }

    /**
     * 複数の状態を一括更新
     */
    updateMultiple(updates) {
        Object.entries(updates).forEach(([key, value]) => {
            this.update(key, value);
        });
    }

    /**
     * キャンバスを設定
     */
    setCanvas(canvas, ctx) {
        this.canvas = canvas;
        this.ctx = ctx;
        this._notify('canvasSet', { canvas, ctx });
    }

    /**
     * 現在のビデオを設定
     */
    setCurrentVideo(video) {
        this.currentVideo = video;
        this._notify('videoChanged', { video });
    }

    /**
     * 現在のフレームを設定
     */
    setCurrentFrame(frame, index) {
        this.currentFrame = frame;
        this.currentFrameIndex = index;
        this._notify('frameChanged', { frame, index });
    }

    /**
     * フレームリストを設定
     */
    setFrameList(frames) {
        this.frameList = frames;
        this._notify('frameListChanged', { frames });
    }

    /**
     * アノテーションを追加
     */
    addAnnotation(annotation) {
        this.annotations.push(annotation);
        this._notify('annotationAdded', { annotation });
    }

    /**
     * アノテーションを削除
     */
    removeAnnotation(index) {
        const removed = this.annotations.splice(index, 1)[0];
        this._notify('annotationRemoved', { annotation: removed, index });
    }

    /**
     * すべてのアノテーションをクリア
     */
    clearAnnotations() {
        this.annotations = [];
        this._notify('annotationsCleared');
    }

    /**
     * アノテーションを更新
     */
    updateAnnotation(index, updates) {
        if (index >= 0 && index < this.annotations.length) {
            Object.assign(this.annotations[index], updates);
            this._notify('annotationUpdated', { index, updates });
        }
    }

    /**
     * 手牌領域を設定
     */
    setHandArea(player, area) {
        this.handAreas[player] = area;
        this._notify('handAreaChanged', { player, area });
    }

    /**
     * 現在のプレイヤーを設定
     */
    setCurrentPlayer(player) {
        this.currentPlayer = player;
        this._notify('currentPlayerChanged', { player });
    }

    /**
     * ズームレベルを設定
     */
    setZoom(level) {
        this.zoomLevel = Math.max(0.1, Math.min(5, level));
        this._notify('zoomChanged', { zoomLevel: this.zoomLevel });
    }

    /**
     * パン位置を設定
     */
    setPan(x, y) {
        this.panX = x;
        this.panY = y;
        this._notify('panChanged', { panX: x, panY: y });
    }

    /**
     * リセット
     */
    reset() {
        this.zoomLevel = 1;
        this.panX = 0;
        this.panY = 0;
        this.annotations = [];
        this.currentBbox = null;
        this.isDrawing = false;
        this._notify('stateReset');
    }

    /**
     * イベントリスナーを登録
     */
    on(event, callback) {
        if (!this._listeners[event]) {
            this._listeners[event] = [];
        }
        this._listeners[event].push(callback);
    }

    /**
     * イベントリスナーを削除
     */
    off(event, callback) {
        if (!this._listeners[event]) return;

        const index = this._listeners[event].indexOf(callback);
        if (index > -1) {
            this._listeners[event].splice(index, 1);
        }
    }

    /**
     * イベントを通知
     */
    _notify(event, data) {
        if (!this._listeners[event]) return;

        this._listeners[event].forEach(callback => {
            try {
                callback(data);
            } catch (error) {
                console.error(`Error in event listener for ${event}:`, error);
            }
        });
    }

    /**
     * 現在の状態をシリアライズ
     */
    serialize() {
        return {
            sessionId: this.sessionId,
            currentVideo: this.currentVideo,
            currentFrameIndex: this.currentFrameIndex,
            annotations: this.annotations,
            handAreas: this.handAreas,
            currentPlayer: this.currentPlayer,
            selectedTileType: this.selectedTileType,
            zoomLevel: this.zoomLevel,
            panX: this.panX,
            panY: this.panY
        };
    }

    /**
     * 状態を復元
     */
    deserialize(data) {
        if (data.sessionId) this.sessionId = data.sessionId;
        if (data.currentVideo) this.currentVideo = data.currentVideo;
        if (data.currentFrameIndex !== undefined) this.currentFrameIndex = data.currentFrameIndex;
        if (data.annotations) this.annotations = data.annotations;
        if (data.handAreas) this.handAreas = data.handAreas;
        if (data.currentPlayer) this.currentPlayer = data.currentPlayer;
        if (data.selectedTileType) this.selectedTileType = data.selectedTileType;
        if (data.zoomLevel !== undefined) this.zoomLevel = data.zoomLevel;
        if (data.panX !== undefined) this.panX = data.panX;
        if (data.panY !== undefined) this.panY = data.panY;

        this._notify('stateRestored', data);
    }
}

// シングルトンインスタンスをエクスポート
export const appState = new AppState();
