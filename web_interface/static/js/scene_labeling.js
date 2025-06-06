/**
 * 対局画面ラベリング用JavaScript
 */

class SceneLabelingApp {
    constructor() {
        this.sessionId = null;
        this.currentFrame = 0;
        this.totalFrames = 0;
        this.videoInfo = null;
        this.canvas = document.getElementById('frameCanvas');
        this.ctx = this.canvas.getContext('2d');

        this.initializeEventListeners();
    }

    initializeEventListeners() {
        // ファイル選択
        document.getElementById('videoFile').addEventListener('change', (e) => {
            this.handleFileSelect(e);
        });

        // セッション開始
        document.getElementById('startSession').addEventListener('click', () => {
            this.startSession();
        });

        // ナビゲーション
        document.getElementById('prevFrame').addEventListener('click', () => {
            this.navigateFrame(-1);
        });

        document.getElementById('nextFrame').addEventListener('click', () => {
            this.navigateFrame(1);
        });

        document.getElementById('frameSlider').addEventListener('input', (e) => {
            this.jumpToFrame(parseInt(e.target.value));
        });

        document.getElementById('jumpButton').addEventListener('click', () => {
            const frameNum = parseInt(document.getElementById('jumpToFrame').value);
            if (!isNaN(frameNum)) {
                this.jumpToFrame(frameNum);
            }
        });

        document.getElementById('jumpToUnlabeled').addEventListener('click', () => {
            this.jumpToNextUnlabeled();
        });

        document.getElementById('jumpToUncertain').addEventListener('click', () => {
            this.jumpToUncertainFrame();
        });

        // ラベリング
        document.getElementById('labelGame').addEventListener('click', () => {
            this.labelCurrentFrame(true);
        });

        document.getElementById('labelNonGame').addEventListener('click', () => {
            this.labelCurrentFrame(false);
        });

        // バッチラベリング
        document.getElementById('batchLabelGame').addEventListener('click', () => {
            this.batchLabel(true);
        });

        document.getElementById('batchLabelNonGame').addEventListener('click', () => {
            this.batchLabel(false);
        });

        // 自動ラベリング
        document.getElementById('autoLabel').addEventListener('click', () => {
            this.autoLabel();
        });

        // エクスポート
        document.getElementById('exportSegments').addEventListener('click', () => {
            this.exportSegments();
        });

        // モデル学習
        document.getElementById('showTrainingPanel').addEventListener('click', () => {
            this.showTrainingModal();
        });

        document.getElementById('prepareTraining').addEventListener('click', () => {
            this.prepareTraining();
        });

        document.getElementById('startTraining').addEventListener('click', () => {
            this.startTraining();
        });

        // キーボードショートカット
        document.addEventListener('keydown', (e) => {
            this.handleKeyPress(e);
        });
    }

    async handleFileSelect(event) {
        const file = event.target.files[0];
        if (!file) return;

        // ファイルをアップロード
        const formData = new FormData();
        formData.append('video', file);

        try {
            const response = await fetch('/api/upload_video', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const result = await response.json();
                this.videoPath = result.video_info.filepath;
                document.getElementById('startSession').disabled = false;
                showNotification('動画をアップロードしました', 'success');
            } else {
                showNotification('アップロードに失敗しました', 'error');
            }
        } catch (error) {
            console.error('Upload error:', error);
            showNotification('アップロードエラー', 'error');
        }
    }

    async startSession() {
        if (!this.videoPath) {
            showNotification('動画を選択してください', 'warning');
            return;
        }

        try {
            const response = await fetch('/api/scene_labeling/sessions', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    video_path: this.videoPath
                })
            });

            if (response.ok) {
                const result = await response.json();
                this.sessionId = result.session_id;
                this.videoInfo = result.video_info;
                this.totalFrames = result.video_info.total_frames;

                // UIを更新
                document.getElementById('totalFrames').textContent = this.totalFrames;
                document.getElementById('frameSlider').max = this.totalFrames - 1;
                document.getElementById('labelingArea').style.display = 'block';

                // 統計情報を更新
                this.updateStatistics(result.statistics);

                // 最初のフレームを読み込み
                await this.loadFrame(0);

                showNotification('セッションを開始しました', 'success');
            } else {
                showNotification('セッション開始に失敗しました', 'error');
            }
        } catch (error) {
            console.error('Session start error:', error);
            showNotification('セッション開始エラー', 'error');
        }
    }

    async loadFrame(frameNumber) {
        if (!this.sessionId) return;

        try {
            const response = await fetch(
                `/api/scene_labeling/sessions/${this.sessionId}/frame/${frameNumber}`
            );

            if (response.ok) {
                const result = await response.json();
                this.currentFrame = frameNumber;

                // フレームを表示
                const img = new Image();
                img.onload = () => {
                    // キャンバスサイズを調整
                    const scale = Math.min(
                        this.canvas.width / img.width,
                        this.canvas.height / img.height
                    );
                    const width = img.width * scale;
                    const height = img.height * scale;
                    const x = (this.canvas.width - width) / 2;
                    const y = (this.canvas.height - height) / 2;

                    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
                    this.ctx.drawImage(img, x, y, width, height);
                };
                img.src = result.image;

                // UI更新
                document.getElementById('currentFrame').textContent = frameNumber;
                document.getElementById('timestamp').textContent =
                    (frameNumber / this.videoInfo.fps).toFixed(2);
                document.getElementById('frameSlider').value = frameNumber;

                // AI推論結果を表示
                if (result.auto_result) {
                    document.getElementById('aiResult').textContent =
                        result.auto_result.is_game_scene ? '対局画面' : '非対局画面';
                    document.getElementById('aiConfidence').textContent =
                        (result.auto_result.confidence * 100).toFixed(1) + '%';
                }

                // 既存のラベルを表示
                if (result.label) {
                    this.showExistingLabel(result.label);
                } else {
                    this.clearLabelIndicator();
                }

            } else {
                showNotification('フレーム読み込みに失敗しました', 'error');
            }
        } catch (error) {
            console.error('Frame load error:', error);
            showNotification('フレーム読み込みエラー', 'error');
        }
    }

    async labelCurrentFrame(isGameScene) {
        if (!this.sessionId) return;

        try {
            const response = await fetch(`/api/scene_labeling/sessions/${this.sessionId}/label`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    frame_number: this.currentFrame,
                    is_game_scene: isGameScene,
                    annotator: 'manual'
                })
            });

            if (response.ok) {
                const result = await response.json();
                this.updateStatistics(result.statistics);
                this.showLabelIndicator(isGameScene);

                // 次のフレームへ自動移動
                setTimeout(() => {
                    this.navigateFrame(1);
                }, 100);
            } else {
                showNotification('ラベル付与に失敗しました', 'error');
            }
        } catch (error) {
            console.error('Label error:', error);
            showNotification('ラベル付与エラー', 'error');
        }
    }

    async batchLabel(isGameScene) {
        const startFrame = parseInt(document.getElementById('batchStart').value);
        const endFrame = parseInt(document.getElementById('batchEnd').value);

        if (isNaN(startFrame) || isNaN(endFrame)) {
            showNotification('開始・終了フレームを入力してください', 'warning');
            return;
        }

        try {
            const response = await fetch(
                `/api/scene_labeling/sessions/${this.sessionId}/batch_label`,
                {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        start_frame: startFrame,
                        end_frame: endFrame,
                        is_game_scene: isGameScene,
                        annotator: 'manual'
                    })
                }
            );

            if (response.ok) {
                const result = await response.json();
                this.updateStatistics(result.statistics);
                showNotification(
                    `${result.labeled_count}フレームにラベルを付与しました`,
                    'success'
                );

                // 現在のフレームを再読み込み
                await this.loadFrame(this.currentFrame);
            } else {
                showNotification('バッチラベリングに失敗しました', 'error');
            }
        } catch (error) {
            console.error('Batch label error:', error);
            showNotification('バッチラベリングエラー', 'error');
        }
    }

    async autoLabel() {
        const interval = parseInt(document.getElementById('sampleInterval').value);

        if (confirm(`${interval}フレーム間隔で自動ラベリングを実行しますか？`)) {
            try {
                const response = await fetch(
                    `/api/scene_labeling/sessions/${this.sessionId}/auto_label`,
                    {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            sample_interval: interval
                        })
                    }
                );

                if (response.ok) {
                    const result = await response.json();
                    this.updateStatistics(result.statistics);
                    showNotification(
                        `${result.labeled_count}フレームを自動ラベリングしました`,
                        'success'
                    );

                    // 現在のフレームを再読み込み
                    await this.loadFrame(this.currentFrame);
                } else {
                    showNotification('自動ラベリングに失敗しました', 'error');
                }
            } catch (error) {
                console.error('Auto label error:', error);
                showNotification('自動ラベリングエラー', 'error');
            }
        }
    }

    async jumpToNextUnlabeled() {
        try {
            const response = await fetch(
                `/api/scene_labeling/sessions/${this.sessionId}/next_unlabeled?start_from=${this.currentFrame}`
            );

            if (response.ok) {
                const result = await response.json();
                if (result.next_frame !== null) {
                    await this.loadFrame(result.next_frame);
                } else {
                    showNotification('未ラベルフレームがありません', 'info');
                }
            }
        } catch (error) {
            console.error('Jump error:', error);
        }
    }

    async jumpToUncertainFrame() {
        try {
            const response = await fetch(
                `/api/scene_labeling/sessions/${this.sessionId}/uncertainty_frame`
            );

            if (response.ok) {
                const result = await response.json();
                if (result.frame_number !== null) {
                    await this.loadFrame(result.frame_number);
                } else {
                    showNotification('不確実なフレームがありません', 'info');
                }
            }
        } catch (error) {
            console.error('Jump error:', error);
        }
    }

    async exportSegments() {
        try {
            const response = await fetch(
                `/api/scene_labeling/sessions/${this.sessionId}/segments`
            );

            if (response.ok) {
                const result = await response.json();

                // JSONをダウンロード
                const blob = new Blob([JSON.stringify(result, null, 2)], {
                    type: 'application/json'
                });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `segments_${this.sessionId}.json`;
                a.click();
                URL.revokeObjectURL(url);

                showNotification(
                    `${result.total_segments}個のセグメントをエクスポートしました`,
                    'success'
                );
            }
        } catch (error) {
            console.error('Export error:', error);
            showNotification('エクスポートエラー', 'error');
        }
    }

    navigateFrame(direction) {
        const newFrame = this.currentFrame + direction;
        if (newFrame >= 0 && newFrame < this.totalFrames) {
            this.loadFrame(newFrame);
        }
    }

    jumpToFrame(frameNumber) {
        if (frameNumber >= 0 && frameNumber < this.totalFrames) {
            this.loadFrame(frameNumber);
        }
    }

    handleKeyPress(event) {
        // 入力フィールドにフォーカスがある場合は無視
        if (event.target.tagName === 'INPUT') return;

        switch (event.key.toLowerCase()) {
            case 'g':
                event.preventDefault();
                this.labelCurrentFrame(true);
                break;
            case 'n':
                event.preventDefault();
                this.labelCurrentFrame(false);
                break;
            case 'arrowleft':
                event.preventDefault();
                this.navigateFrame(-1);
                break;
            case 'arrowright':
                event.preventDefault();
                this.navigateFrame(1);
                break;
            case ' ':
                event.preventDefault();
                this.jumpToNextUnlabeled();
                break;
            case 'u':
                event.preventDefault();
                this.jumpToUncertainFrame();
                break;
        }
    }

    updateStatistics(stats) {
        document.getElementById('statTotalFrames').textContent = stats.total_frames;
        document.getElementById('statLabeled').textContent = stats.labeled_frames;
        document.getElementById('statGameScenes').textContent = stats.game_scenes;
        document.getElementById('statNonGameScenes').textContent = stats.non_game_scenes;
        document.getElementById('statProgress').textContent =
            (stats.progress * 100).toFixed(1);

        // プログレスバーを更新
        const progressBar = document.getElementById('progressBar');
        progressBar.style.width = (stats.progress * 100) + '%';
        progressBar.textContent = (stats.progress * 100).toFixed(1) + '%';
    }

    showLabelIndicator(isGameScene) {
        // キャンバスの枠線を変更
        this.canvas.style.border = isGameScene ? '5px solid #28a745' : '5px solid #dc3545';
    }

    showExistingLabel(label) {
        this.showLabelIndicator(label.is_game_scene);

        // ラベル情報を表示
        const labelText = label.is_game_scene ? '対局画面' : '非対局画面';
        const annotatorText = label.annotator === 'auto' ? '(自動)' : '(手動)';

        // 既存のオーバーレイを削除
        const existingOverlay = document.getElementById('labelOverlay');
        if (existingOverlay) {
            existingOverlay.remove();
        }

        // 新しいオーバーレイを作成
        const overlay = document.createElement('div');
        overlay.id = 'labelOverlay';
        overlay.className = 'position-absolute top-0 start-0 m-2 badge';
        overlay.classList.add(label.is_game_scene ? 'bg-success' : 'bg-danger');
        overlay.textContent = `${labelText} ${annotatorText}`;
        this.canvas.parentElement.style.position = 'relative';
        this.canvas.parentElement.appendChild(overlay);
    }

    clearLabelIndicator() {
        this.canvas.style.border = '1px solid #dee2e6';
        const overlay = document.getElementById('labelOverlay');
        if (overlay) {
            overlay.remove();
        }
    }

    async showTrainingModal() {
        // モーダルを表示
        const modal = new bootstrap.Modal(document.getElementById('trainingModal'));
        modal.show();

        // データセット情報を読み込み
        await this.loadDatasetInfo();
    }

    async loadDatasetInfo() {
        try {
            const response = await fetch('/api/scene_training/datasets');
            if (response.ok) {
                const datasets = await response.json();

                // 各分割の情報を表示
                for (const split of ['train', 'val', 'test']) {
                    const info = datasets[split];
                    const element = document.getElementById(`${split}DatasetInfo`);

                    if (info.error) {
                        element.innerHTML = `<div class="text-danger">エラー: ${info.error}</div>`;
                    } else {
                        element.innerHTML = `
                            <div>総サンプル: ${info.total_samples}</div>
                            <div>対局: ${info.game_scenes}</div>
                            <div>非対局: ${info.non_game_scenes}</div>
                            <div>動画数: ${info.videos}</div>
                        `;
                    }
                }

                // データが十分あれば準備ボタンを有効化
                const hasEnoughData = datasets.train.total_samples > 0 &&
                                    datasets.val.total_samples > 0;
                document.getElementById('prepareTraining').disabled = !hasEnoughData;

            } else {
                showNotification('データセット情報の取得に失敗しました', 'error');
            }
        } catch (error) {
            console.error('Dataset info error:', error);
            showNotification('データセット情報取得エラー', 'error');
        }
    }

    async prepareTraining() {
        try {
            const response = await fetch('/api/scene_training/prepare', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (response.ok) {
                const result = await response.json();

                if (result.ready) {
                    document.getElementById('startTraining').disabled = false;
                    showNotification('学習データの準備が完了しました', 'success');
                } else {
                    showNotification('学習データが不足しています', 'warning');
                }
            } else {
                showNotification('データ準備に失敗しました', 'error');
            }
        } catch (error) {
            console.error('Prepare training error:', error);
            showNotification('データ準備エラー', 'error');
        }
    }

    async startTraining() {
        const epochs = parseInt(document.getElementById('trainingEpochs').value);
        const batchSize = parseInt(document.getElementById('trainingBatchSize').value);
        const learningRate = parseFloat(document.getElementById('trainingLearningRate').value);

        try {
            const response = await fetch('/api/scene_training/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    epochs: epochs,
                    batch_size: batchSize,
                    learning_rate: learningRate
                })
            });

            if (response.ok) {
                const result = await response.json();
                this.trainingSessionId = result.session_id;

                // 進捗表示を開始
                document.getElementById('trainingProgress').style.display = 'block';
                document.getElementById('startTraining').disabled = true;

                // WebSocketでセッションに参加
                socket.emit('join_session', { session_id: this.trainingSessionId });

                // 学習完了イベントをリッスン
                socket.on('scene_training_complete', (data) => {
                    if (data.session_id === this.trainingSessionId) {
                        this.onTrainingComplete(data.results);
                    }
                });

                socket.on('scene_training_error', (data) => {
                    if (data.session_id === this.trainingSessionId) {
                        this.onTrainingError(data.error);
                    }
                });

                showNotification('学習を開始しました', 'success');
            } else {
                const error = await response.json();
                showNotification(`学習開始に失敗: ${error.error}`, 'error');
            }
        } catch (error) {
            console.error('Start training error:', error);
            showNotification('学習開始エラー', 'error');
        }
    }

    onTrainingComplete(results) {
        document.getElementById('trainingProgressBar').style.width = '100%';
        document.getElementById('trainingProgressBar').textContent = '完了';

        const log = document.getElementById('trainingLog');
        log.innerHTML = `
            <div class="alert alert-success">
                <h6>学習完了</h6>
                <div>エポック数: ${results.epochs_trained}</div>
                <div>最良検証精度: ${(results.best_val_acc * 100).toFixed(2)}%</div>
                <div>最終学習精度: ${(results.final_train_acc * 100).toFixed(2)}%</div>
                <div>モデル保存先: ${results.paths.best_model}</div>
            </div>
        `;

        showNotification('モデル学習が完了しました', 'success');
        document.getElementById('startTraining').disabled = false;
    }

    onTrainingError(error) {
        const log = document.getElementById('trainingLog');
        log.innerHTML = `
            <div class="alert alert-danger">
                <h6>学習エラー</h6>
                <div>${error}</div>
            </div>
        `;

        showNotification('学習中にエラーが発生しました', 'error');
        document.getElementById('startTraining').disabled = false;
    }
}

// アプリケーションを初期化
document.addEventListener('DOMContentLoaded', () => {
    window.sceneLabelingApp = new SceneLabelingApp();
});
