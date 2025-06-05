/**
 * 手牌領域管理クラス
 */
export class HandAreaManager {
    constructor(appState, canvasManager, apiClient) {
        this.appState = appState;
        this.canvasManager = canvasManager;
        this.apiClient = apiClient;

        // 手牌領域の表示設定
        this.areaColors = {
            bottom: '#00ff00',
            top: '#ffff00',
            left: '#00ffff',
            right: '#ff00ff'
        };

        this.playerNames = {
            bottom: '自分',
            top: '対面',
            left: '左',
            right: '右'
        };
    }

    /**
     * 手牌領域を設定
     */
    setArea(player, area) {
        if (!this.isValidPlayer(player)) {
            throw new Error(`Invalid player: ${player}`);
        }

        this.appState.setHandArea(player, area);
    }

    /**
     * 手牌領域を取得
     */
    getArea(player) {
        return this.appState.handAreas[player];
    }

    /**
     * すべての手牌領域を取得
     */
    getAllAreas() {
        return { ...this.appState.handAreas };
    }

    /**
     * 有効なプレイヤーかチェック
     */
    isValidPlayer(player) {
        return ['bottom', 'top', 'left', 'right'].includes(player);
    }

    /**
     * 手牌領域を描画
     */
    drawAreas() {
        Object.entries(this.appState.handAreas).forEach(([player, area]) => {
            if (area) {
                const isCurrentPlayer = player === this.appState.currentPlayer;
                const color = this.areaColors[player];
                const alpha = isCurrentPlayer ? 0.3 : 0.1;
                const label = this.playerNames[player];

                this.canvasManager.drawArea(area, label, color, alpha);
            }
        });
    }

    /**
     * 手牌領域をハイライト
     */
    highlightArea(player) {
        const area = this.getArea(player);
        if (area) {
            const color = this.areaColors[player];
            this.canvasManager.drawArea(area, this.playerNames[player], color, 0.5);
        }
    }

    /**
     * マウス位置の手牌領域を検出
     */
    detectAreaAtPoint(x, y) {
        // 正規化座標に変換
        const normalizedX = x / this.canvasManager.canvas.width;
        const normalizedY = y / this.canvasManager.canvas.height;

        for (const [player, area] of Object.entries(this.appState.handAreas)) {
            if (area && this.isPointInArea(normalizedX, normalizedY, area)) {
                return player;
            }
        }

        return null;
    }

    /**
     * 点が領域内にあるかチェック
     */
    isPointInArea(x, y, area) {
        return x >= area.x &&
               x <= area.x + area.w &&
               y >= area.y &&
               y <= area.y + area.h;
    }

    /**
     * 手牌領域を正規化（0-1の範囲に収める）
     */
    normalizeArea(area) {
        return {
            x: Math.max(0, Math.min(1, area.x)),
            y: Math.max(0, Math.min(1, area.y)),
            w: Math.max(0, Math.min(1 - area.x, area.w)),
            h: Math.max(0, Math.min(1 - area.y, area.h))
        };
    }

    /**
     * 手牌領域の妥当性をチェック
     */
    validateArea(area) {
        if (!area || typeof area !== 'object') {
            return false;
        }

        const required = ['x', 'y', 'w', 'h'];
        for (const prop of required) {
            if (typeof area[prop] !== 'number' || area[prop] < 0 || area[prop] > 1) {
                return false;
            }
        }

        // 最小サイズチェック
        return area.w > 0.01 && area.h > 0.01;
    }

    /**
     * 手牌領域をサーバーに保存
     */
    async saveAreas() {
        try {
            const frameSize = this.appState.currentImage
                ? [this.appState.currentImage.width, this.appState.currentImage.height]
                : null;

            const response = await this.apiClient.setHandAreas(frameSize, this.appState.handAreas);
            return response;
        } catch (error) {
            console.error('Failed to save hand areas:', error);
            throw error;
        }
    }

    /**
     * 手牌領域をサーバーから読み込み
     */
    async loadAreas() {
        try {
            const response = await this.apiClient.getHandAreas();

            if (response.regions) {
                Object.entries(response.regions).forEach(([player, area]) => {
                    if (this.validateArea(area)) {
                        this.setArea(player, area);
                    }
                });
            }

            return response;
        } catch (error) {
            console.error('Failed to load hand areas:', error);
            throw error;
        }
    }

    /**
     * 手牌領域から牌を抽出
     */
    extractHandRegion(image, player) {
        const area = this.getArea(player);
        if (!area || !image) {
            return null;
        }

        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');

        const x = Math.floor(area.x * image.width);
        const y = Math.floor(area.y * image.height);
        const w = Math.floor(area.w * image.width);
        const h = Math.floor(area.h * image.height);

        canvas.width = w;
        canvas.height = h;

        ctx.drawImage(image, x, y, w, h, 0, 0, w, h);

        return canvas;
    }

    /**
     * 手牌領域のプレビューを生成
     */
    generatePreview(player, maxWidth = 200) {
        const handCanvas = this.extractHandRegion(this.appState.currentImage, player);
        if (!handCanvas) {
            return null;
        }

        // リサイズ
        const scale = Math.min(1, maxWidth / handCanvas.width);
        const previewCanvas = document.createElement('canvas');
        const ctx = previewCanvas.getContext('2d');

        previewCanvas.width = handCanvas.width * scale;
        previewCanvas.height = handCanvas.height * scale;

        ctx.drawImage(handCanvas, 0, 0, previewCanvas.width, previewCanvas.height);

        return previewCanvas.toDataURL();
    }

    /**
     * デフォルトの手牌領域を設定
     */
    setDefaultAreas() {
        const defaults = {
            bottom: { x: 0.15, y: 0.75, w: 0.7, h: 0.15 },
            top: { x: 0.15, y: 0.1, w: 0.7, h: 0.15 },
            left: { x: 0.05, y: 0.3, w: 0.15, h: 0.4 },
            right: { x: 0.8, y: 0.3, w: 0.15, h: 0.4 }
        };

        Object.entries(defaults).forEach(([player, area]) => {
            this.setArea(player, area);
        });
    }

    /**
     * 手牌領域をクリア
     */
    clearAreas() {
        ['bottom', 'top', 'left', 'right'].forEach(player => {
            this.appState.setHandArea(player, null);
        });
    }

    /**
     * 手牌領域を自動検出（実験的）
     */
    async autoDetectAreas(image) {
        // TODO: 画像解析による自動検出の実装
        // 現時点ではデフォルト値を返す
        this.setDefaultAreas();
        return this.getAllAreas();
    }

    /**
     * 手牌領域設定をエクスポート
     */
    exportConfig() {
        return {
            areas: this.getAllAreas(),
            frameSize: this.appState.currentImage
                ? {
                    width: this.appState.currentImage.width,
                    height: this.appState.currentImage.height
                }
                : null,
            timestamp: new Date().toISOString()
        };
    }

    /**
     * 手牌領域設定をインポート
     */
    importConfig(config) {
        if (!config || !config.areas) {
            throw new Error('Invalid configuration format');
        }

        Object.entries(config.areas).forEach(([player, area]) => {
            if (this.isValidPlayer(player) && this.validateArea(area)) {
                this.setArea(player, area);
            }
        });
    }
}
