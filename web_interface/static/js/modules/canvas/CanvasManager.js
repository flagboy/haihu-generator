/**
 * キャンバス管理クラス
 */
export class CanvasManager {
    constructor(canvasElement, appState) {
        this.canvas = canvasElement;
        this.ctx = canvasElement.getContext('2d');
        this.appState = appState;

        // キャンバス設定
        this.setupCanvas();

        // イベントハンドラー
        this._onDrawCallback = null;
        this._onClickCallback = null;
    }

    /**
     * キャンバスの初期設定
     */
    setupCanvas() {
        // アンチエイリアスを有効化
        this.ctx.imageSmoothingEnabled = true;
        this.ctx.imageSmoothingQuality = 'high';

        // 状態を同期
        this.appState.setCanvas(this.canvas, this.ctx);
    }

    /**
     * キャンバスをリサイズ
     */
    resize(containerWidth, containerHeight) {
        if (!this.appState.currentImage) return;

        const image = this.appState.currentImage;
        const imageAspect = image.width / image.height;
        const containerAspect = containerWidth / containerHeight;

        if (imageAspect > containerAspect) {
            this.canvas.width = containerWidth;
            this.canvas.height = containerWidth / imageAspect;
        } else {
            this.canvas.width = containerHeight * imageAspect;
            this.canvas.height = containerHeight;
        }

        this.redraw();
    }

    /**
     * 画像を読み込み
     */
    async loadImage(src) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
                this.appState.currentImage = img;
                // コンテナサイズに合わせてリサイズ
                const container = this.canvas.parentElement;
                this.resize(container.clientWidth, Math.min(600, container.clientHeight));
                resolve(img);
            };
            img.onerror = reject;
            img.src = src;
        });
    }

    /**
     * キャンバスを再描画
     */
    redraw() {
        if (!this.appState.currentImage) return;

        const { zoomLevel, panX, panY } = this.appState;

        // キャンバスをクリア
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // 画像を描画
        this.ctx.save();
        this.ctx.scale(zoomLevel, zoomLevel);
        this.ctx.translate(panX, panY);
        this.ctx.drawImage(
            this.appState.currentImage,
            0, 0,
            this.canvas.width / zoomLevel,
            this.canvas.height / zoomLevel
        );
        this.ctx.restore();

        // カスタム描画コールバック
        if (this._onDrawCallback) {
            this._onDrawCallback(this.ctx);
        }
    }

    /**
     * バウンディングボックスを描画
     */
    drawBoundingBox(bbox, color = '#007bff', lineWidth = 2, fillAlpha = 0) {
        const { zoomLevel, panX, panY } = this.appState;

        this.ctx.save();
        this.ctx.scale(zoomLevel, zoomLevel);
        this.ctx.translate(panX, panY);

        // 塗りつぶし（オプション）
        if (fillAlpha > 0) {
            this.ctx.fillStyle = color + Math.round(fillAlpha * 255).toString(16).padStart(2, '0');
            this.ctx.fillRect(bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1);
        }

        // 枠線
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = lineWidth / zoomLevel;
        this.ctx.strokeRect(bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1);

        this.ctx.restore();
    }

    /**
     * ラベルを描画
     */
    drawLabel(bbox, text, color = '#007bff', fontSize = 14) {
        const { zoomLevel, panX, panY } = this.appState;

        this.ctx.save();
        this.ctx.scale(zoomLevel, zoomLevel);
        this.ctx.translate(panX, panY);

        const scaledFontSize = fontSize / zoomLevel;
        this.ctx.font = `${scaledFontSize}px Arial`;

        // テキストサイズを測定
        const textMetrics = this.ctx.measureText(text);
        const textWidth = textMetrics.width;
        const textHeight = scaledFontSize;

        // 背景を描画
        this.ctx.fillStyle = color;
        this.ctx.fillRect(
            bbox.x1,
            bbox.y1 - textHeight - 4,
            textWidth + 8,
            textHeight + 4
        );

        // テキストを描画
        this.ctx.fillStyle = '#ffffff';
        this.ctx.fillText(text, bbox.x1 + 4, bbox.y1 - 4);

        this.ctx.restore();
    }

    /**
     * 領域を描画（手牌領域など）
     */
    drawArea(area, label, color = '#00ff00', alpha = 0.3) {
        const { zoomLevel, panX, panY } = this.appState;

        this.ctx.save();
        this.ctx.scale(zoomLevel, zoomLevel);
        this.ctx.translate(panX, panY);

        const x = area.x * this.canvas.width / zoomLevel;
        const y = area.y * this.canvas.height / zoomLevel;
        const w = area.w * this.canvas.width / zoomLevel;
        const h = area.h * this.canvas.height / zoomLevel;

        // 領域を塗りつぶし
        this.ctx.fillStyle = color + Math.round(alpha * 255).toString(16).padStart(2, '0');
        this.ctx.fillRect(x, y, w, h);

        // 枠線
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 2 / zoomLevel;
        this.ctx.strokeRect(x, y, w, h);

        // ラベル
        if (label) {
            this.ctx.fillStyle = color;
            this.ctx.font = `${16 / zoomLevel}px Arial`;
            this.ctx.fillText(label, x + 5, y + 20 / zoomLevel);
        }

        this.ctx.restore();
    }

    /**
     * 座標変換：キャンバス座標 → 画像座標
     */
    canvasToImageCoords(canvasX, canvasY) {
        const { zoomLevel, panX, panY } = this.appState;
        return {
            x: canvasX / zoomLevel - panX,
            y: canvasY / zoomLevel - panY
        };
    }

    /**
     * 座標変換：画像座標 → キャンバス座標
     */
    imageToCanvasCoords(imageX, imageY) {
        const { zoomLevel, panX, panY } = this.appState;
        return {
            x: (imageX + panX) * zoomLevel,
            y: (imageY + panY) * zoomLevel
        };
    }

    /**
     * 座標変換：正規化座標 → 画像座標
     */
    normalizedToImageCoords(normalizedX, normalizedY) {
        return {
            x: normalizedX * this.canvas.width,
            y: normalizedY * this.canvas.height
        };
    }

    /**
     * ズーム
     */
    zoom(factor, centerX = null, centerY = null) {
        const oldZoom = this.appState.zoomLevel;
        const newZoom = Math.max(0.1, Math.min(5, oldZoom * factor));

        if (centerX !== null && centerY !== null) {
            // マウス位置を中心にズーム
            const rect = this.canvas.getBoundingClientRect();
            const canvasX = centerX - rect.left;
            const canvasY = centerY - rect.top;

            // ズーム前の画像座標
            const imgCoords = this.canvasToImageCoords(canvasX, canvasY);

            // ズームレベルを更新
            this.appState.setZoom(newZoom);

            // ズーム後も同じ画像座標がマウス位置に来るようにパンを調整
            const newCanvasCoords = this.imageToCanvasCoords(imgCoords.x, imgCoords.y);
            const panAdjustX = (canvasX - newCanvasCoords.x) / newZoom;
            const panAdjustY = (canvasY - newCanvasCoords.y) / newZoom;

            this.appState.setPan(
                this.appState.panX + panAdjustX,
                this.appState.panY + panAdjustY
            );
        } else {
            // 中心を基準にズーム
            this.appState.setZoom(newZoom);
        }

        this.redraw();
    }

    /**
     * パン（移動）
     */
    pan(deltaX, deltaY) {
        const { zoomLevel } = this.appState;
        this.appState.setPan(
            this.appState.panX + deltaX / zoomLevel,
            this.appState.panY + deltaY / zoomLevel
        );
        this.redraw();
    }

    /**
     * ズームとパンをリセット
     */
    resetView() {
        this.appState.setZoom(1);
        this.appState.setPan(0, 0);
        this.redraw();
    }

    /**
     * 描画コールバックを設定
     */
    onDraw(callback) {
        this._onDrawCallback = callback;
    }

    /**
     * クリックイベントを処理
     */
    handleClick(event) {
        const rect = this.canvas.getBoundingClientRect();
        const canvasX = event.clientX - rect.left;
        const canvasY = event.clientY - rect.top;
        const imgCoords = this.canvasToImageCoords(canvasX, canvasY);

        if (this._onClickCallback) {
            this._onClickCallback(imgCoords, event);
        }
    }

    /**
     * クリックコールバックを設定
     */
    onClick(callback) {
        this._onClickCallback = callback;
    }

    /**
     * キャンバスをエクスポート
     */
    exportImage(format = 'png', quality = 0.9) {
        return new Promise((resolve) => {
            this.canvas.toBlob((blob) => {
                resolve(blob);
            }, `image/${format}`, quality);
        });
    }

    /**
     * スクリーンショットを取得
     */
    getScreenshot() {
        return this.canvas.toDataURL('image/png');
    }

    /**
     * キャンバスをクリア
     */
    clear() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }

    /**
     * 破棄処理
     */
    destroy() {
        this.clear();
        this._onDrawCallback = null;
        this._onClickCallback = null;
    }
}
