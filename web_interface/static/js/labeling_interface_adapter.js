/**
 * ラベリングインターフェースアダプター
 *
 * 拡張ショートカットとバッチラベリング機能を既存のインターフェースに統合
 */

class LabelingInterfaceAdapter {
    constructor(existingInterface) {
        this.interface = existingInterface;
        this.selectedBoxes = new Set();
        this.clipboard = null;
        this.quickLabelMode = null;
        this.zoom = 1.0;
        this.gridVisible = false;
        this.labelsVisible = true;
        this.overlayVisible = true;

        // 拡張機能の初期化
        this.initializeEnhancements();
    }

    initializeEnhancements() {
        // ショートカットマネージャーの初期化
        this.shortcutManager = new EnhancedShortcutManager(this);

        // クイックラベリングモードの初期化
        this.quickLabelMode = new QuickLabelingMode(this);

        // 既存のキャンバスを取得
        this.canvas = this.interface.canvas || document.getElementById('labeling-canvas');

        // 追加のUI要素を作成
        this.createAdditionalUI();
    }

    createAdditionalUI() {
        // ステータスバーの作成
        const statusBar = document.createElement('div');
        statusBar.id = 'labeling-status-bar';
        statusBar.style.cssText = `
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            height: 30px;
            background: #333;
            color: white;
            display: flex;
            align-items: center;
            padding: 0 20px;
            font-size: 14px;
            z-index: 1000;
        `;

        statusBar.innerHTML = `
            <span>フレーム: <span id="status-frame">0/0</span></span>
            <span style="margin-left: 20px;">選択: <span id="status-selected">0</span></span>
            <span style="margin-left: 20px;">ズーム: <span id="status-zoom">100%</span></span>
            <span style="margin-left: auto;">Hキーでヘルプ</span>
        `;

        document.body.appendChild(statusBar);

        // グリッドオーバーレイの作成
        this.createGridOverlay();
    }

    createGridOverlay() {
        const gridCanvas = document.createElement('canvas');
        gridCanvas.id = 'grid-overlay';
        gridCanvas.style.cssText = `
            position: absolute;
            top: 0;
            left: 0;
            pointer-events: none;
            display: none;
        `;

        if (this.canvas && this.canvas.parentElement) {
            this.canvas.parentElement.appendChild(gridCanvas);
            this.gridCanvas = gridCanvas;
        }
    }

    // ナビゲーション関数
    nextFrame() {
        if (this.interface.nextFrame) {
            this.interface.nextFrame();
        } else if (this.interface.frameIndex !== undefined) {
            this.interface.frameIndex++;
            this.interface.loadFrame(this.interface.frameIndex);
        }
        this.updateStatus();
    }

    previousFrame() {
        if (this.interface.previousFrame) {
            this.interface.previousFrame();
        } else if (this.interface.frameIndex !== undefined) {
            this.interface.frameIndex--;
            this.interface.loadFrame(this.interface.frameIndex);
        }
        this.updateStatus();
    }

    goToFirstFrame() {
        if (this.interface.goToFirstFrame) {
            this.interface.goToFirstFrame();
        } else {
            this.interface.frameIndex = 0;
            this.interface.loadFrame(0);
        }
        this.updateStatus();
    }

    goToLastFrame() {
        if (this.interface.goToLastFrame) {
            this.interface.goToLastFrame();
        } else if (this.interface.totalFrames) {
            this.interface.frameIndex = this.interface.totalFrames - 1;
            this.interface.loadFrame(this.interface.frameIndex);
        }
        this.updateStatus();
    }

    togglePlayPause() {
        if (this.interface.togglePlayPause) {
            this.interface.togglePlayPause();
        }
    }

    // ラベリング操作
    confirmCurrentBox() {
        if (this.interface.confirmCurrentBox) {
            this.interface.confirmCurrentBox();
        } else if (this.interface.currentBox) {
            this.interface.saveAnnotation(this.interface.currentBox);
            this.interface.currentBox = null;
        }
    }

    cancelCurrentBox() {
        if (this.interface.cancelCurrentBox) {
            this.interface.cancelCurrentBox();
        } else {
            this.interface.currentBox = null;
            this.interface.redraw();
        }
    }

    deleteSelectedBox() {
        if (this.selectedBoxes.size > 0) {
            // 複数選択の削除
            this.selectedBoxes.forEach(boxId => {
                this.interface.deleteAnnotation(boxId);
            });
            this.selectedBoxes.clear();
        } else if (this.interface.deleteSelectedBox) {
            this.interface.deleteSelectedBox();
        }
        this.updateStatus();
    }

    // コピー＆ペースト
    copySelectedBox() {
        if (this.selectedBoxes.size > 0) {
            // 複数選択のコピー
            this.clipboard = [];
            this.selectedBoxes.forEach(boxId => {
                const annotation = this.interface.getAnnotation(boxId);
                if (annotation) {
                    this.clipboard.push(annotation);
                }
            });
        } else if (this.interface.selectedBox) {
            const annotation = this.interface.getAnnotation(this.interface.selectedBox);
            if (annotation) {
                this.clipboard = [annotation];
            }
        }

        if (this.clipboard && this.clipboard.length > 0) {
            this.showMessage(`${this.clipboard.length}個のボックスをコピーしました`);
        }
    }

    pasteBox() {
        if (!this.clipboard || this.clipboard.length === 0) {
            this.showMessage('クリップボードが空です');
            return;
        }

        // 現在のフレームにペースト
        this.clipboard.forEach(annotation => {
            const newAnnotation = {
                ...annotation,
                frame_id: this.interface.currentFrameId || this.interface.frameIndex,
                id: `ann_${Date.now()}_${Math.random()}`
            };
            this.interface.addAnnotation(newAnnotation);
        });

        this.showMessage(`${this.clipboard.length}個のボックスを貼り付けました`);
        this.interface.redraw();
    }

    copyPreviousFrame() {
        const previousFrameIndex = (this.interface.frameIndex || 0) - 1;
        if (previousFrameIndex < 0) {
            this.showMessage('前のフレームがありません');
            return;
        }

        // 前フレームのアノテーションを取得
        const previousAnnotations = this.interface.getFrameAnnotations(previousFrameIndex);
        if (previousAnnotations && previousAnnotations.length > 0) {
            // 現在のフレームにコピー
            previousAnnotations.forEach(annotation => {
                const newAnnotation = {
                    ...annotation,
                    frame_id: this.interface.currentFrameId || this.interface.frameIndex,
                    id: `ann_${Date.now()}_${Math.random()}`
                };
                this.interface.addAnnotation(newAnnotation);
            });

            this.showMessage(`前フレームから${previousAnnotations.length}個のボックスをコピーしました`);
            this.interface.redraw();
        } else {
            this.showMessage('前フレームにアノテーションがありません');
        }
    }

    // 牌の選択
    selectTileType(type) {
        if (this.interface.selectTileType) {
            this.interface.selectTileType(type);
        }

        // クイックラベリングモード用に記憶
        if (this.quickLabelMode) {
            this.quickLabelMode.lastTileType = type;
        }

        this.showMessage(`${type}を選択しました`);
    }

    selectTileNumber(number) {
        if (this.interface.selectTileNumber) {
            this.interface.selectTileNumber(number);
        }

        // クイックラベリングモード用に記憶
        if (this.quickLabelMode) {
            this.quickLabelMode.lastTileNumber = number;
        }

        this.showMessage(`${number}を選択しました`);
    }

    toggleRedDora() {
        if (this.interface.toggleRedDora) {
            this.interface.toggleRedDora();
        } else {
            // 現在の5を赤ドラに変換
            this.interface.isRedDora = !this.interface.isRedDora;
            this.showMessage(`赤ドラ: ${this.interface.isRedDora ? 'ON' : 'OFF'}`);
        }
    }

    selectBackTile() {
        if (this.interface.selectBackTile) {
            this.interface.selectBackTile();
        } else {
            this.interface.currentTileType = 'back';
            this.showMessage('裏面牌を選択しました');
        }
    }

    // 表示制御
    toggleGrid() {
        this.gridVisible = !this.gridVisible;
        if (this.gridCanvas) {
            this.gridCanvas.style.display = this.gridVisible ? 'block' : 'none';
            if (this.gridVisible) {
                this.drawGrid();
            }
        }
        this.showMessage(`グリッド: ${this.gridVisible ? 'ON' : 'OFF'}`);
    }

    toggleLabels() {
        this.labelsVisible = !this.labelsVisible;
        this.interface.showLabels = this.labelsVisible;
        this.interface.redraw();
        this.showMessage(`ラベル: ${this.labelsVisible ? 'ON' : 'OFF'}`);
    }

    toggleOverlay() {
        this.overlayVisible = !this.overlayVisible;
        this.interface.showOverlay = this.overlayVisible;
        this.interface.redraw();
        this.showMessage(`オーバーレイ: ${this.overlayVisible ? 'ON' : 'OFF'}`);
    }

    // バッチ操作
    selectAllBoxes() {
        const annotations = this.interface.getCurrentFrameAnnotations();
        if (annotations) {
            annotations.forEach(ann => {
                this.selectedBoxes.add(ann.id);
            });
            this.updateSelection();
            this.showMessage(`${this.selectedBoxes.size}個のボックスを選択しました`);
        }
    }

    deselectAllBoxes() {
        this.selectedBoxes.clear();
        this.updateSelection();
        this.showMessage('選択を解除しました');
    }

    deleteAllBoxes() {
        if (confirm('現在のフレームのすべてのボックスを削除しますか？')) {
            const annotations = this.interface.getCurrentFrameAnnotations();
            if (annotations) {
                annotations.forEach(ann => {
                    this.interface.deleteAnnotation(ann.id);
                });
                this.showMessage('すべてのボックスを削除しました');
                this.interface.redraw();
            }
        }
    }

    // クイックアクション
    quickLabelMode() {
        if (this.quickLabelMode) {
            this.quickLabelMode.toggle();
        }
    }

    switchToNextUnlabeled() {
        // 次の未ラベルフレームを検索
        const currentIndex = this.interface.frameIndex || 0;
        const totalFrames = this.interface.totalFrames || 0;

        for (let i = currentIndex + 1; i < totalFrames; i++) {
            const annotations = this.interface.getFrameAnnotations(i);
            if (!annotations || annotations.length === 0) {
                this.interface.frameIndex = i;
                this.interface.loadFrame(i);
                this.showMessage(`未ラベルフレーム ${i} に移動しました`);
                return;
            }
        }

        this.showMessage('未ラベルフレームが見つかりません');
    }

    switchToPreviousUnlabeled() {
        // 前の未ラベルフレームを検索
        const currentIndex = this.interface.frameIndex || 0;

        for (let i = currentIndex - 1; i >= 0; i--) {
            const annotations = this.interface.getFrameAnnotations(i);
            if (!annotations || annotations.length === 0) {
                this.interface.frameIndex = i;
                this.interface.loadFrame(i);
                this.showMessage(`未ラベルフレーム ${i} に移動しました`);
                return;
            }
        }

        this.showMessage('未ラベルフレームが見つかりません');
    }

    saveProgress() {
        if (this.interface.saveProgress) {
            this.interface.saveProgress();
        } else {
            // 自動保存をトリガー
            this.interface.saveAnnotations();
        }
        this.showMessage('進捗を保存しました');
    }

    // ズーム機能
    zoomIn() {
        this.zoom = Math.min(this.zoom * 1.2, 5.0);
        this.applyZoom();
        this.showMessage(`ズーム: ${Math.round(this.zoom * 100)}%`);
    }

    zoomOut() {
        this.zoom = Math.max(this.zoom / 1.2, 0.2);
        this.applyZoom();
        this.showMessage(`ズーム: ${Math.round(this.zoom * 100)}%`);
    }

    resetZoom() {
        this.zoom = 1.0;
        this.applyZoom();
        this.showMessage('ズームをリセットしました');
    }

    applyZoom() {
        if (this.canvas) {
            this.canvas.style.transform = `scale(${this.zoom})`;
            this.canvas.style.transformOrigin = 'center center';
        }
        this.updateStatus();
    }

    // プレイヤー選択
    selectPlayer(playerNumber) {
        if (this.interface.selectPlayer) {
            this.interface.selectPlayer(playerNumber);
        } else {
            this.interface.currentPlayer = playerNumber;
        }
        this.showMessage(`プレイヤー ${playerNumber} を選択しました`);
    }

    // ユーティリティ関数
    showMessage(message) {
        if (this.interface.showMessage) {
            this.interface.showMessage(message);
        } else {
            // 簡易的なメッセージ表示
            const messageDiv = document.createElement('div');
            messageDiv.style.cssText = `
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: rgba(0, 0, 0, 0.8);
                color: white;
                padding: 20px 40px;
                border-radius: 10px;
                font-size: 18px;
                z-index: 10000;
            `;
            messageDiv.textContent = message;
            document.body.appendChild(messageDiv);

            setTimeout(() => {
                messageDiv.remove();
            }, 2000);
        }
    }

    isInputFocused() {
        const activeElement = document.activeElement;
        return activeElement && (
            activeElement.tagName === 'INPUT' ||
            activeElement.tagName === 'TEXTAREA' ||
            activeElement.tagName === 'SELECT'
        );
    }

    updateStatus() {
        const frameSpan = document.getElementById('status-frame');
        const selectedSpan = document.getElementById('status-selected');
        const zoomSpan = document.getElementById('status-zoom');

        if (frameSpan) {
            const current = (this.interface.frameIndex || 0) + 1;
            const total = this.interface.totalFrames || 0;
            frameSpan.textContent = `${current}/${total}`;
        }

        if (selectedSpan) {
            selectedSpan.textContent = this.selectedBoxes.size;
        }

        if (zoomSpan) {
            zoomSpan.textContent = `${Math.round(this.zoom * 100)}%`;
        }
    }

    updateSelection() {
        // 選択状態の視覚的更新
        const allBoxes = document.querySelectorAll('.annotation-box');
        allBoxes.forEach(box => {
            if (this.selectedBoxes.has(box.dataset.id)) {
                box.classList.add('selected');
            } else {
                box.classList.remove('selected');
            }
        });

        this.updateStatus();
    }

    addToSelection(element) {
        const id = element.dataset.id;
        if (id) {
            if (this.selectedBoxes.has(id)) {
                this.selectedBoxes.delete(id);
            } else {
                this.selectedBoxes.add(id);
            }
            this.updateSelection();
        }
    }

    drawGrid() {
        if (!this.gridCanvas || !this.canvas) return;

        const ctx = this.gridCanvas.getContext('2d');
        const width = this.canvas.width;
        const height = this.canvas.height;

        this.gridCanvas.width = width;
        this.gridCanvas.height = height;

        ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
        ctx.lineWidth = 1;

        // グリッドの描画
        const gridSize = 50;

        for (let x = 0; x < width; x += gridSize) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
        }

        for (let y = 0; y < height; y += gridSize) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }
    }

    // インターフェース互換性のためのメソッド
    getFrame(frameIndex) {
        if (this.interface.getFrame) {
            return this.interface.getFrame(frameIndex);
        }
        return null;
    }

    getFrameShape(frameId) {
        if (this.interface.getFrameShape) {
            return this.interface.getFrameShape(frameId);
        }
        return [this.canvas.height, this.canvas.width];
    }

    getFrameAnnotations(frameIndex) {
        if (this.interface.getFrameAnnotations) {
            return this.interface.getFrameAnnotations(frameIndex);
        }
        return [];
    }

    getCurrentFrameAnnotations() {
        const frameIndex = this.interface.frameIndex || 0;
        return this.getFrameAnnotations(frameIndex);
    }

    getAnnotation(annotationId) {
        if (this.interface.getAnnotation) {
            return this.interface.getAnnotation(annotationId);
        }
        return null;
    }

    addAnnotation(annotation) {
        if (this.interface.addAnnotation) {
            this.interface.addAnnotation(annotation);
        }
    }

    deleteAnnotation(annotationId) {
        if (this.interface.deleteAnnotation) {
            this.interface.deleteAnnotation(annotationId);
        }
    }

    createBox(bbox, tileType, tileNumber) {
        const annotation = {
            id: `ann_${Date.now()}_${Math.random()}`,
            frame_id: this.interface.currentFrameId || this.interface.frameIndex,
            bbox: [bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height],
            tile_type: tileType,
            tile_id: `${tileNumber}${tileType[0]}`,
            created_at: new Date().toISOString()
        };

        this.addAnnotation(annotation);
        this.interface.redraw();

        return annotation;
    }
}

// エクスポート
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { LabelingInterfaceAdapter };
}
