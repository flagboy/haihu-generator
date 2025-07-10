/**
 * 拡張版アノテーションシステム - クライアントサイドJS
 * 類似フレームスキップ機能を含む効率的な教師データ作成
 */

class EnhancedAnnotationSystem {
    constructor() {
        // キャンバス関連
        this.canvas = document.getElementById('mainCanvas');
        this.ctx = this.canvas.getContext('2d');

        // 状態管理
        this.currentFrame = 0;
        this.totalFrames = 1000;
        this.annotations = [];
        this.selectedObjects = [];
        this.currentTool = 'select';
        this.zoom = 1.0;

        // 類似度チェック
        this.similarityCheckEnabled = true;
        this.skipSimilarFrames = true;
        this.similarityThreshold = 0.95;
        this.frameHistory = [];
        this.skippedFrames = new Set();

        // ドラッグ関連
        this.isDragging = false;
        this.dragStart = { x: 0, y: 0 };

        // 履歴管理
        this.history = [];
        this.historyIndex = -1;

        // セッション情報
        this.sessionId = null;
        this.videoId = null;

        // 初期化
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupKeyboardShortcuts();
        this.initTileSelector();
        this.loadSession();
    }

    // セッション管理
    async loadSession() {
        try {
            const urlParams = new URLSearchParams(window.location.search);
            this.videoId = urlParams.get('video_id');

            if (!this.videoId) {
                this.showToast('動画IDが指定されていません', 'error');
                return;
            }

            // セッション作成
            const response = await fetch('/api/annotation/session', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    video_id: this.videoId,
                    annotator: 'user'
                })
            });

            const data = await response.json();
            if (data.success) {
                this.sessionId = data.session.session_id;
                this.totalFrames = data.session.total_frames;

                // スマートフレーム選択を実行
                await this.performSmartFrameSelection();

                // 最初のフレームをロード
                this.loadFrame(0);
            }
        } catch (error) {
            console.error('セッション作成エラー:', error);
            this.showToast('セッションの作成に失敗しました', 'error');
        }
    }

    // スマートフレーム選択
    async performSmartFrameSelection() {
        this.showLoading('効率的なフレームを選択中...');

        try {
            const response = await fetch('/api/annotation/smart-select', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    video_id: this.videoId,
                    mode: 'adaptive'
                })
            });

            const data = await response.json();
            if (data.success) {
                // 選択されなかったフレームをスキップリストに追加
                for (let i = 0; i < this.totalFrames; i++) {
                    if (!data.selected_frames.includes(i)) {
                        this.skippedFrames.add(i);
                    }
                }

                this.showToast(
                    `${data.total_selected}フレームを選択しました (${(data.selection_rate * 100).toFixed(1)}%)`,
                    'success'
                );
            }
        } catch (error) {
            console.error('スマート選択エラー:', error);
        } finally {
            this.hideLoading();
        }
    }

    // フレーム読み込み
    async loadFrame(frameNumber, checkSimilarity = true) {
        this.showLoading();

        try {
            // 類似度チェック（必要な場合）
            if (checkSimilarity && this.similarityCheckEnabled && !this.skippedFrames.has(frameNumber)) {
                const similarity = await this.checkFrameSimilarity(frameNumber);
                if (similarity && similarity.should_skip) {
                    this.showSimilarityWarning(similarity);

                    // 自動スキップが有効な場合
                    if (this.skipSimilarFrames) {
                        await this.skipFrame(frameNumber, 'similar_to_previous');
                        // 次の非スキップフレームを探す
                        const nextFrame = this.findNextUnskippedFrame(frameNumber);
                        if (nextFrame !== null) {
                            await this.loadFrame(nextFrame, false);
                            return;
                        }
                    }
                }
            }

            // フレームデータ取得
            const response = await fetch(`/api/annotation/frame/${frameNumber}`);
            const frameData = await response.json();

            this.currentFrame = frameNumber;
            await this.displayFrame(frameData);
            this.updateProgress();
            this.updateNavigationButtons();

            // フレーム履歴に追加
            this.frameHistory.push({
                frameNumber: frameNumber,
                timestamp: Date.now()
            });
            if (this.frameHistory.length > 100) {
                this.frameHistory.shift();
            }

        } catch (error) {
            console.error('フレーム読み込みエラー:', error);
            this.showToast('フレームの読み込みに失敗しました', 'error');
        } finally {
            this.hideLoading();
        }
    }

    // 類似度チェック
    async checkFrameSimilarity(frameNumber) {
        try {
            const response = await fetch('/api/annotation/check-similarity', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    frame_number: frameNumber,
                    video_id: this.videoId
                })
            });

            return await response.json();
        } catch (error) {
            console.error('類似度チェックエラー:', error);
            return null;
        }
    }

    // 類似度警告表示
    showSimilarityWarning(similarity) {
        const indicator = document.createElement('div');
        indicator.className = 'similarity-warning';
        indicator.innerHTML = `
            <div class="similarity-score ${this.getSimilarityClass(similarity.similarity_score)}">
                類似度: ${(similarity.similarity_score * 100).toFixed(1)}%
            </div>
            <div class="similarity-type">${this.getSimilarityTypeText(similarity.similarity_type)}</div>
            <div class="similarity-actions">
                <button onclick="annotationSystem.skipCurrentFrame()">スキップ</button>
                <button onclick="annotationSystem.dismissSimilarityWarning()">続行</button>
            </div>
        `;

        document.querySelector('.image-container').appendChild(indicator);

        // 3秒後に自動的に非表示
        setTimeout(() => {
            this.dismissSimilarityWarning();
        }, 3000);
    }

    dismissSimilarityWarning() {
        const warning = document.querySelector('.similarity-warning');
        if (warning) {
            warning.remove();
        }
    }

    getSimilarityClass(score) {
        if (score >= 0.95) return 'similarity-very-high';
        if (score >= 0.85) return 'similarity-high';
        return 'similarity-medium';
    }

    getSimilarityTypeText(type) {
        const texts = {
            'identical': 'ほぼ同一のフレーム',
            'very_similar': '非常に類似したフレーム',
            'similar': '類似したフレーム'
        };
        return texts[type] || type;
    }

    // フレームスキップ
    async skipFrame(frameNumber, reason = 'manual_skip') {
        try {
            const response = await fetch(`/api/annotation/skip-frame/${frameNumber}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    video_id: this.videoId,
                    reason: reason
                })
            });

            const data = await response.json();
            if (data.success) {
                this.skippedFrames.add(frameNumber);
                this.showToast('フレームをスキップしました', 'info');
            }
        } catch (error) {
            console.error('フレームスキップエラー:', error);
        }
    }

    async skipCurrentFrame() {
        await this.skipFrame(this.currentFrame, 'manual_skip');
        this.nextFrame();
    }

    // 次の非スキップフレームを探す
    findNextUnskippedFrame(currentFrame) {
        for (let i = currentFrame + 1; i < this.totalFrames; i++) {
            if (!this.skippedFrames.has(i)) {
                return i;
            }
        }
        return null;
    }

    // 前の非スキップフレームを探す
    findPreviousUnskippedFrame(currentFrame) {
        for (let i = currentFrame - 1; i >= 0; i--) {
            if (!this.skippedFrames.has(i)) {
                return i;
            }
        }
        return null;
    }

    // フレーム表示
    async displayFrame(frameData) {
        // 画像読み込み
        const img = new Image();
        img.onload = () => {
            this.canvas.width = img.width;
            this.canvas.height = img.height;
            this.ctx.drawImage(img, 0, 0);

            // アノテーション描画
            this.drawAnnotations();

            // スキップフレームの表示
            if (frameData.isSkipped || this.skippedFrames.has(this.currentFrame)) {
                this.drawSkippedOverlay(frameData.skipReason);
            }
        };
        img.src = frameData.imagePath;

        // UI更新
        document.getElementById('sceneType').textContent = frameData.sceneType || '不明';
        document.getElementById('roundInfo').textContent = frameData.roundInfo || '-';
        document.getElementById('dealerPosition').textContent = frameData.dealerPosition || '-';

        // スキップ状態の表示
        if (frameData.isSkipped || this.skippedFrames.has(this.currentFrame)) {
            document.body.classList.add('skipped-frame');
        } else {
            document.body.classList.remove('skipped-frame');
        }

        // アノテーション読み込み
        this.annotations = frameData.annotations || [];
        this.updateObjectList();
    }

    // スキップオーバーレイ描画
    drawSkippedOverlay(reason) {
        this.ctx.fillStyle = 'rgba(128, 128, 128, 0.3)';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        const text = this.getSkipReasonText(reason);
        const boxWidth = 400;
        const boxHeight = 60;
        const x = this.canvas.width / 2 - boxWidth / 2;
        const y = this.canvas.height / 2 - boxHeight / 2;

        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        this.ctx.fillRect(x, y, boxWidth, boxHeight);

        this.ctx.fillStyle = 'white';
        this.ctx.font = '20px sans-serif';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(text, this.canvas.width / 2, this.canvas.height / 2 + 7);
    }

    getSkipReasonText(reason) {
        const reasons = {
            'similar_to_previous': '前のフレームと類似',
            'manual_skip': '手動でスキップ',
            'auto_skip': '自動スキップ',
            'low_importance': '重要度が低い'
        };
        return reasons[reason] || 'スキップされたフレーム';
    }

    // ナビゲーション
    previousFrame() {
        const prevFrame = this.findPreviousUnskippedFrame(this.currentFrame);
        if (prevFrame !== null) {
            this.saveCurrentAnnotations();
            this.loadFrame(prevFrame);
        } else {
            this.showToast('最初のフレームです', 'info');
        }
    }

    nextFrame() {
        const nextFrame = this.findNextUnskippedFrame(this.currentFrame);
        if (nextFrame !== null) {
            this.saveCurrentAnnotations();
            this.loadFrame(nextFrame);
        } else {
            this.showToast('最後のフレームです', 'info');
        }
    }

    jumpToFrame() {
        const frameNumber = prompt('フレーム番号を入力:', this.currentFrame);
        if (frameNumber !== null) {
            const num = parseInt(frameNumber);
            if (num >= 0 && num < this.totalFrames) {
                this.saveCurrentAnnotations();
                this.loadFrame(num);
            }
        }
    }

    // 進捗更新
    updateProgress() {
        const actualProgress = (this.currentFrame + 1) / this.totalFrames * 100;
        const annotatedFrames = this.totalFrames - this.skippedFrames.size;
        const annotationProgress = annotatedFrames > 0 ?
            (this.frameHistory.length / annotatedFrames * 100) : 0;

        document.getElementById('progressFill').style.width = actualProgress + '%';
        document.getElementById('frameInfo').textContent =
            `フレーム ${this.currentFrame + 1} / ${this.totalFrames}`;

        // 効率統計の更新
        document.getElementById('totalFrames').textContent = this.totalFrames;
        document.getElementById('annotatedFrames').textContent = this.frameHistory.length;
        document.getElementById('skippedFrames').textContent = this.skippedFrames.size;
        document.getElementById('skipRate').textContent =
            `${(this.skippedFrames.size / this.totalFrames * 100).toFixed(1)}%`;
    }

    // ナビゲーションボタンの更新
    updateNavigationButtons() {
        const prevButton = document.getElementById('prevButton');
        const nextButton = document.getElementById('nextButton');

        if (prevButton) {
            prevButton.disabled = this.findPreviousUnskippedFrame(this.currentFrame) === null;
        }
        if (nextButton) {
            nextButton.disabled = this.findNextUnskippedFrame(this.currentFrame) === null;
        }
    }

    // アノテーション描画
    drawAnnotations() {
        this.annotations.forEach((anno, index) => {
            const isSelected = this.selectedObjects.includes(index);
            this.drawBoundingBox(anno.bbox, anno.type, anno.confidence, isSelected);
        });
    }

    drawBoundingBox(bbox, type, confidence, isSelected) {
        this.ctx.save();

        // 信頼度による色分け
        let color;
        if (confidence >= 0.8) color = '#4CAF50';
        else if (confidence >= 0.5) color = '#FF9800';
        else color = '#f44336';

        // 選択時は太く
        this.ctx.lineWidth = isSelected ? 3 : 2;
        this.ctx.strokeStyle = color;
        this.ctx.strokeRect(bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1);

        // ラベル表示
        this.ctx.fillStyle = color;
        this.ctx.fillRect(bbox.x1, bbox.y1 - 20, 60, 20);
        this.ctx.fillStyle = 'white';
        this.ctx.font = '12px sans-serif';
        this.ctx.fillText(`${type} ${(confidence * 100).toFixed(0)}%`, bbox.x1 + 5, bbox.y1 - 5);

        this.ctx.restore();
    }

    // 自動検出
    async autoDetect() {
        this.showLoading('自動検出中...');

        try {
            const response = await fetch(`/api/annotation/auto-detect/${this.currentFrame}`, {
                method: 'POST'
            });

            const data = await response.json();
            if (data.success) {
                this.annotations = data.annotations;
                this.drawAnnotations();
                this.updateObjectList();
                this.showToast(`${data.annotations.length}個の牌を検出しました`, 'success');
                this.addToHistory('auto-detect');
            }
        } catch (error) {
            this.showToast('自動検出に失敗しました', 'error');
        } finally {
            this.hideLoading();
        }
    }

    // 保存
    async saveAnnotations() {
        this.saveCurrentAnnotations();

        try {
            const response = await fetch('/api/annotation/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    videoId: this.videoId,
                    frameNumber: this.currentFrame,
                    annotations: this.annotations,
                    gameState: {
                        round: document.getElementById('roundSelect').value,
                        honba: parseInt(document.getElementById('honbaInput').value),
                        dealer: document.getElementById('dealerSelect').value,
                        remainingTiles: parseInt(document.getElementById('remainingTilesInput').value)
                    }
                })
            });

            const data = await response.json();
            if (data.success) {
                this.showToast('保存しました', 'success');
                this.updateLastSaved();
            }
        } catch (error) {
            this.showToast('保存に失敗しました', 'error');
        }
    }

    saveCurrentAnnotations() {
        // ローカルストレージに保存
        const frameData = {
            frameNumber: this.currentFrame,
            annotations: this.annotations,
            timestamp: new Date().toISOString()
        };

        let savedData = JSON.parse(localStorage.getItem('annotationData') || '{}');
        savedData[this.currentFrame] = frameData;
        localStorage.setItem('annotationData', JSON.stringify(savedData));
    }

    // UI設定
    setupSettings() {
        // 類似度設定
        const similarityToggle = document.getElementById('similarityCheckToggle');
        if (similarityToggle) {
            similarityToggle.checked = this.similarityCheckEnabled;
            similarityToggle.addEventListener('change', (e) => {
                this.similarityCheckEnabled = e.target.checked;
                localStorage.setItem('similarityCheckEnabled', this.similarityCheckEnabled);
            });
        }

        const skipToggle = document.getElementById('skipSimilarToggle');
        if (skipToggle) {
            skipToggle.checked = this.skipSimilarFrames;
            skipToggle.addEventListener('change', (e) => {
                this.skipSimilarFrames = e.target.checked;
                localStorage.setItem('skipSimilarFrames', this.skipSimilarFrames);
            });
        }

        const thresholdSlider = document.getElementById('similarityThreshold');
        if (thresholdSlider) {
            thresholdSlider.value = this.similarityThreshold * 100;
            thresholdSlider.addEventListener('input', (e) => {
                this.similarityThreshold = e.target.value / 100;
                document.getElementById('thresholdValue').textContent = `${e.target.value}%`;
                localStorage.setItem('similarityThreshold', this.similarityThreshold);
            });
        }
    }

    // ヘルパー関数
    showToast(message, type = 'info') {
        const toast = document.getElementById('toast');
        toast.textContent = message;
        toast.className = `toast ${type} show`;

        setTimeout(() => {
            toast.classList.remove('show');
        }, 3000);
    }

    showLoading(message = '読み込み中...') {
        const loading = document.getElementById('loading');
        const loadingText = loading.querySelector('p');
        loadingText.textContent = message;
        loading.style.display = 'block';
    }

    hideLoading() {
        document.getElementById('loading').style.display = 'none';
    }

    updateLastSaved() {
        const now = new Date();
        const timeStr = now.toLocaleTimeString('ja-JP', {
            hour: '2-digit',
            minute: '2-digit'
        });
        document.getElementById('lastSaved').textContent = `最終保存: ${timeStr}`;
    }

    // イベントリスナー設定
    setupEventListeners() {
        // キャンバスイベント
        this.canvas.addEventListener('mousedown', (e) => this.onMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this.onMouseMove(e));
        this.canvas.addEventListener('mouseup', (e) => this.onMouseUp(e));
        this.canvas.addEventListener('wheel', (e) => this.onWheel(e));

        // 設定
        this.setupSettings();
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // ナビゲーション
            if (e.key === 'a' || e.key === 'ArrowLeft') this.previousFrame();
            else if (e.key === 'd' || e.key === 'ArrowRight') this.nextFrame();
            else if (e.key === 'g') this.jumpToFrame();

            // アクション
            else if (e.key === 'r') this.autoDetect();
            else if (e.key === 's' && !e.ctrlKey) this.skipCurrentFrame();

            // 保存
            else if (e.ctrlKey && e.key === 's') {
                e.preventDefault();
                this.saveAnnotations();
            }
        });
    }

    // マウスイベント（簡略版）
    onMouseDown(e) {
        // 実装省略
    }

    onMouseMove(e) {
        // 実装省略
    }

    onMouseUp(e) {
        // 実装省略
    }

    onWheel(e) {
        // 実装省略
    }

    // その他のメソッド
    initTileSelector() {
        // 実装省略
    }

    updateObjectList() {
        // 実装省略
    }

    addToHistory(action) {
        // 実装省略
    }
}

// グローバルインスタンス
let annotationSystem;

// DOM読み込み完了時に初期化
document.addEventListener('DOMContentLoaded', () => {
    annotationSystem = new EnhancedAnnotationSystem();
});
