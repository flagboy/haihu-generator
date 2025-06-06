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

        // 類似フレーム検出用
        this.previousFrameData = null;
        this.lastLabeledFrameData = null;  // 最後にラベル付けしたフレームのデータ
        this.similarityThreshold = 0.90;  // 90%以上類似している場合はスキップ
        this.autoSkipEnabled = true;
        this.skipCount = 0;
        this.isSkipping = false;  // スキップ処理中フラグ
        this.searchingUnlabeled = false;  // 未ラベル検索中フラグ
        this.skipJustEnded = false;  // スキップが終了したばかりかどうかのフラグ

        // エラーリトライ制御用
        this.frameLoadRetryCount = 0;
        this.maxRetries = 3;
        this.lastErrorFrame = -1;

        // デフォルトのビデオパスを設定
        const videoPathInput = document.getElementById('videoPath');
        if (videoPathInput && videoPathInput.value) {
            this.videoPath = videoPathInput.value;
            console.log('デフォルトの動画パスを設定:', this.videoPath);
        }

        this.initializeEventListeners();
    }

    initializeEventListeners() {
        // パス入力
        document.getElementById('videoPath').addEventListener('input', (e) => {
            this.videoPath = e.target.value.trim();
            console.log('動画パスが入力されました:', this.videoPath);
        });

        // アップロード済みファイル一覧
        document.getElementById('browseUploads').addEventListener('click', () => {
            this.showUploadedFiles();
        });

        // セッション開始
        document.getElementById('startSession').addEventListener('click', () => {
            this.startSession();
        });

        // 既存セッション読み込み
        const loadExistingBtn = document.getElementById('loadExisting');
        if (loadExistingBtn) {
            loadExistingBtn.addEventListener('click', () => {
                console.log('既存セッション読み込みボタンがクリックされました');
                this.showSessionList();
            });
        } else {
            console.error('loadExistingボタンが見つかりません');
        }

        // セッション一覧キャンセル
        document.getElementById('cancelSessionList').addEventListener('click', () => {
            this.hideSessionList();
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

        // スキップ中断
        document.getElementById('stopSkipping').addEventListener('click', () => {
            this.stopSkipping();
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

        // 類似フレーム自動スキップ設定
        document.getElementById('autoSkipSimilar').addEventListener('change', (e) => {
            this.autoSkipEnabled = e.target.checked;
            if (this.autoSkipEnabled) {
                showNotification('類似フレーム自動スキップを有効にしました', 'info');
            } else {
                showNotification('類似フレーム自動スキップを無効にしました', 'info');
            }
        });

        // 類似度閾値変更
        document.getElementById('similarityThreshold').addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            this.similarityThreshold = value / 100;
            document.getElementById('similarityValue').textContent = value;
            document.getElementById('currentThreshold').textContent = value;
        });
    }

    async showUploadedFiles() {
        try {
            const response = await fetch('/api/list_uploaded_videos');
            if (response.ok) {
                const files = await response.json();
                const filesList = document.getElementById('filesListContent');
                filesList.innerHTML = '';

                if (files.length === 0) {
                    filesList.innerHTML = '<div class="text-muted">アップロード済みファイルがありません</div>';
                } else {
                    files.forEach(file => {
                        const item = document.createElement('a');
                        item.href = '#';
                        item.className = 'list-group-item list-group-item-action';
                        item.innerHTML = `
                            <div class="d-flex w-100 justify-content-between">
                                <h6 class="mb-1">${file.filename}</h6>
                                <small>${file.size}</small>
                            </div>
                            <p class="mb-1 small">${file.path}</p>
                            <small>更新日時: ${file.modified}</small>
                        `;
                        item.addEventListener('click', (e) => {
                            e.preventDefault();
                            document.getElementById('videoPath').value = file.path;
                            this.videoPath = file.path;
                            document.getElementById('uploadedFilesList').style.display = 'none';
                            showNotification(`ファイルを選択しました: ${file.filename}`, 'info');
                        });
                        filesList.appendChild(item);
                    });
                }

                document.getElementById('uploadedFilesList').style.display = 'block';
            } else {
                showNotification('ファイル一覧の取得に失敗しました', 'error');
            }
        } catch (error) {
            console.error('ファイル一覧取得エラー:', error);
            showNotification('ファイル一覧取得エラー', 'error');
        }
    }

    async startSession() {
        console.log('startSession called, videoPath:', this.videoPath);

        if (!this.videoPath) {
            showNotification('動画を選択してください', 'warning');
            console.error('videoPathが設定されていません');
            return;
        }

        try {
            console.log('セッション作成リクエスト送信中...', {
                video_path: this.videoPath
            });

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

    async loadFrame(frameNumber, skipDisplay = false) {
        if (!this.sessionId) return;

        try {
            const response = await fetch(
                `/api/scene_labeling/sessions/${this.sessionId}/frame/${frameNumber}`
            );

            if (response.ok) {
                const result = await response.json();
                this.currentFrame = frameNumber;

                // スキップ中の処理
                if (skipDisplay && this.isSkipping) {
                    // 簡易的な画面更新（高速化のため最小限）
                    document.getElementById('currentFrame').textContent = frameNumber;
                    document.getElementById('frameSlider').value = frameNumber;

                    // プログレスバーも更新
                    const progress = (frameNumber / (this.totalFrames - 1)) * 100;
                    const progressBar = document.getElementById('progressBar');
                    progressBar.style.width = progress + '%';
                    progressBar.textContent = progress.toFixed(1) + '%';

                    // 高速プレビュー（縮小版の画像を表示）
                    if (result.image) {
                        const img = new Image();
                        img.onload = async () => {
                            // フルサイズで描画（類似度計算の精度のため）
                            const scale = Math.min(
                                this.canvas.width / img.width,
                                this.canvas.height / img.height
                            );
                            const width = img.width * scale;
                            const height = img.height * scale;
                            const x = (this.canvas.width - width) / 2;
                            const y = (this.canvas.height - height) / 2;

                            this.ctx.fillStyle = '#f0f0f0';
                            this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
                            this.ctx.drawImage(img, x, y, width, height);

                            // スキップ中の表示
                            this.ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
                            this.ctx.fillRect(0, 0, this.canvas.width, 40);
                            this.ctx.fillStyle = '#ff6b00';
                            this.ctx.font = 'bold 20px sans-serif';
                            this.ctx.textAlign = 'center';
                            this.ctx.fillText('スキップ中...', this.canvas.width / 2, 28);

                            // 画像描画完了後に類似度チェックを実行
                            await this.checkSimilarityAndSkip(result, frameNumber);
                        };
                        img.src = result.image;
                    } else {
                        // 画像がない場合も類似度チェックを実行
                        await this.checkSimilarityAndSkip(result, frameNumber);
                    }
                    return;
                }

                // フレームを表示
                const img = new Image();

                // 画像読み込みエラーハンドラ
                img.onerror = (error) => {
                    console.error(`🔴 画像読み込みエラー (フレーム ${frameNumber}):`, error);
                    console.error('エラー詳細:', {
                        frameNumber: frameNumber,
                        sessionId: this.sessionId,
                        imageUrl: result.image ? result.image.substring(0, 100) + '...' : 'URL無し',
                        errorEvent: error,
                        timestamp: new Date().toISOString()
                    });

                    // エラー処理
                    this.handleFrameLoadError(frameNumber, `Image load error: ${error.type || 'unknown'}`, skipDisplay);
                };

                img.onload = async () => {
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

                    // 現在のフレームデータを取得
                    const currentFrameData = this.getCanvasImageData();

                    // 前フレームとの類似度計算（常に実行）
                    let prevFrameSimilarity = 0;
                    if (this.previousFrameData) {
                        prevFrameSimilarity = this.calculateFrameSimilarity(
                            this.previousFrameData,
                            currentFrameData
                        );

                        // 画面上に類似度を表示
                        this.updateSimilarityDisplay(prevFrameSimilarity);
                        console.log(`フレーム ${frameNumber} の前フレームとの類似度: ${prevFrameSimilarity.toFixed(3)}`);
                    }

                    // 前フレームとの類似度チェック（未ラベルフレームのみ）
                    // スキップが終了したばかりの場合はスキップしない
                    if (this.autoSkipEnabled && !result.label && this.previousFrameData && !this.skipJustEnded) {
                        console.log(`フレーム ${frameNumber} の前フレームとの類似度: ${prevFrameSimilarity.toFixed(3)}`);
                        console.log(`前フレーム類似度: ${prevFrameSimilarity.toFixed(3)}, 閾値: ${this.similarityThreshold.toFixed(3)}`);
                        console.log(`🔍 詳細比較:`);
                        console.log(`   prevFrameSimilarity = ${prevFrameSimilarity} (型: ${typeof prevFrameSimilarity})`);
                        console.log(`   similarityThreshold = ${this.similarityThreshold} (型: ${typeof this.similarityThreshold})`);
                        console.log(`   prevFrameSimilarity > similarityThreshold = ${prevFrameSimilarity > this.similarityThreshold}`);
                        console.log(`   数値比較: ${Number(prevFrameSimilarity)} > ${Number(this.similarityThreshold)} = ${Number(prevFrameSimilarity) > Number(this.similarityThreshold)}`);

                        // 前フレームとの類似度が閾値を超えている場合
                        if (prevFrameSimilarity > this.similarityThreshold && this.lastLabeledFrame) {
                            // 類似フレームをスキップ
                            this.skipCount++;
                            console.log(`🚀 スキップ実行: フレーム ${frameNumber}`);
                            console.log(`   理由: 前フレームとの類似度が閾値を超過`);
                            console.log(`   類似度: ${prevFrameSimilarity.toFixed(3)} > 閾値: ${this.similarityThreshold.toFixed(3)}`);
                            console.log(`   付与ラベル: ${this.lastLabeledFrame.isGameScene ? '対局画面' : '非対局画面'}`);

                            // スキップ中はボタンを無効化
                            this.disableLabelingButtons(true);

                            // 次のフレームへ自動的に移動
                            this.isSkipping = true;

                            // スキップUI表示・更新
                            this.showSkippingUI();

                            // 最後にラベル付けしたフレームと同じラベルを付与
                            this.autoLabelSimilarFrame(frameNumber, this.lastLabeledFrame.isGameScene)
                                .then(() => {
                                    setTimeout(() => {
                                        if (this.currentFrame + 1 < this.totalFrames) {
                                            // 次のフレームをロード（isSkippingはtrueのまま、画面更新なし）
                                            this.loadFrame(this.currentFrame + 1, true);
                                        } else {
                                            // 最後のフレームに到達した場合のみスキップ終了
                                            this.isSkipping = false;
                                            this.disableLabelingButtons(false);
                                            showNotification(
                                                `${this.skipCount}個の類似フレームをスキップし、同じラベルを付与しました`,
                                                'info'
                                            );
                                            this.skipCount = 0;
                                        }
                                    }, 10);  // 待機時間を短縮
                                });
                            return;
                        } else {
                            console.log(`⏹️ スキップしない: フレーム ${frameNumber}`);
                            console.log(`   理由: 前フレームとの類似度が閾値以下`);
                            console.log(`   類似度: ${prevFrameSimilarity.toFixed(3)} <= 閾値: ${this.similarityThreshold.toFixed(3)}`);
                        }
                    } else {
                        // スキップ条件に合わない場合の理由をログ出力
                        if (!this.autoSkipEnabled) {
                            console.log(`⏹️ スキップしない: 自動スキップが無効`);
                        } else if (result.label) {
                            console.log(`⏹️ スキップしない: フレーム ${frameNumber} は既にラベル済み`);
                        } else if (!this.previousFrameData) {
                            console.log(`⏹️ スキップしない: 比較対象の前フレームデータがありません`);
                        } else if (!this.lastLabeledFrame) {
                            console.log(`⏹️ スキップしない: 付与するラベル情報がありません`);
                        } else if (this.skipJustEnded) {
                            console.log(`⏹️ スキップしない: スキップが終了したばかりです`);
                        }
                    }

                    // skipJustEndedフラグをリセット
                    if (this.skipJustEnded) {
                        this.skipJustEnded = false;
                    }

                    // 現在のフレームデータを保存（次回比較用）
                    this.previousFrameData = currentFrameData;

                // スキップカウントが残っていれば表示
                if (this.skipCount > 0) {
                    // スキップ終了
                    this.isSkipping = false;
                    this.disableLabelingButtons(false);
                    showNotification(
                        `${this.skipCount}個の類似フレームをスキップしました`,
                        'info'
                    );
                    this.skipCount = 0;
                }
                };
                img.src = result.image;

                // UI更新
                document.getElementById('currentFrame').textContent = frameNumber;
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

                    // 未ラベル検索中で、このフレームがラベル済みの場合は次を探す
                    if (this.searchingUnlabeled) {
                        console.log(`フレーム ${frameNumber} はラベル済み、次の未ラベルを検索`);
                        setTimeout(() => {
                            this.jumpToNextUnlabeled();
                        }, 50);
                        return;
                    }
                } else {
                    this.clearLabelIndicator();
                    // 未ラベルフレームが見つかった場合は検索終了
                    if (this.searchingUnlabeled) {
                        this.searchingUnlabeled = false;
                        this.disableLabelingButtons(false);
                        console.log(`未ラベルフレーム ${frameNumber} が見つかりました`);
                    }
                }

                // スキップ処理中でなければボタンを有効化
                if (!this.isSkipping && !this.searchingUnlabeled) {
                    this.disableLabelingButtons(false);
                }

            } else {
                let errorMessage = `HTTP ${response.status}`;
                try {
                    const errorData = await response.json();
                    errorMessage = errorData.error || errorMessage;
                } catch {
                    const errorText = await response.text();
                    errorMessage = errorText || errorMessage;
                }

                console.error(`フレーム読み込み失敗 (${response.status}):`, errorMessage);
                showNotification(`フレーム読み込みに失敗しました: ${errorMessage}`, 'error');

                // エラー処理
                this.handleFrameLoadError(frameNumber, errorMessage, skipDisplay);
            }
        } catch (error) {
            console.error('Frame load error:', error);
            console.error('Error details:', {
                frameNumber: frameNumber,
                sessionId: this.sessionId,
                skipDisplay: skipDisplay,
                isSkipping: this.isSkipping,
                errorType: error.name,
                errorMessage: error.message,
                errorStack: error.stack
            });

            // ネットワークエラーの場合
            if (error.name === 'TypeError' && error.message.includes('fetch')) {
                showNotification('ネットワークエラー: サーバーに接続できません', 'error');
            } else {
                showNotification(`フレーム読み込みエラー: ${error.message}`, 'error');
            }

            // エラー処理
            this.handleFrameLoadError(frameNumber, error.toString(), skipDisplay);
        }
    }

    // フレーム読み込みエラーのハンドリング
    handleFrameLoadError(frameNumber, errorMessage, skipDisplay) {
        // 同じフレームでエラーが続く場合
        if (this.lastErrorFrame === frameNumber) {
            this.frameLoadRetryCount++;
        } else {
            this.frameLoadRetryCount = 1;
            this.lastErrorFrame = frameNumber;
        }

        console.warn(`⚠️ フレーム ${frameNumber} の読み込みエラー (試行 ${this.frameLoadRetryCount}/${this.maxRetries}):`, errorMessage);
        console.warn('エラー状況:', {
            frameNumber: frameNumber,
            isSkipping: this.isSkipping,
            searchingUnlabeled: this.searchingUnlabeled,
            skipDisplay: skipDisplay,
            totalFrames: this.totalFrames,
            sessionId: this.sessionId,
            retryCount: this.frameLoadRetryCount
        });

        // 最大リトライ回数を超えた場合
        if (this.frameLoadRetryCount >= this.maxRetries) {
            console.error(`❌ フレーム ${frameNumber} の読み込みを諦めます (${this.maxRetries}回失敗)`);

            // 状態をリセット
            this.resetErrorState();

            // エラーフレームをスキップして次のフレームへ
            this.skipErrorFrame(frameNumber);
        } else {
            // リトライする
            const retryDelay = Math.min(this.frameLoadRetryCount * 1000, 5000); // 最大5秒まで

            console.log(`🔄 ${retryDelay}ms 後にフレーム ${frameNumber} の読み込みを再試行します...`);

            // UI更新: リトライ中であることを表示
            const canvas = document.getElementById('frameCanvas');
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = '#f8f9fa';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = '#dc3545';
            ctx.font = 'bold 24px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(`フレーム読み込みエラー`, canvas.width / 2, canvas.height / 2 - 20);
            ctx.font = '16px sans-serif';
            ctx.fillText(`再試行中... (${this.frameLoadRetryCount}/${this.maxRetries})`, canvas.width / 2, canvas.height / 2 + 20);

            setTimeout(() => {
                this.retryFrameLoad(frameNumber, skipDisplay);
            }, retryDelay);
        }
    }

    // エラー状態をリセット
    resetErrorState() {
        this.isSkipping = false;
        this.searchingUnlabeled = false;
        this.disableLabelingButtons(false);
        this.hideSkippingUI();

        // エラー時も比較データをクリア
        this.previousFrameData = null;
        this.lastLabeledFrameData = null;
        this.lastLabeledFrame = null;

        if (this.skipCount > 0) {
            showNotification(
                `エラーによりスキップを中断しました (${this.skipCount}個のフレームをスキップ済み)`,
                'warning'
            );
            this.skipCount = 0;
        }
    }

    // エラーフレームをスキップ
    skipErrorFrame(frameNumber) {
        if (frameNumber < this.totalFrames - 1) {
            showNotification(
                `フレーム ${frameNumber} をスキップして次のフレームに移動します`,
                'warning'
            );

            // リトライカウントをリセット
            this.frameLoadRetryCount = 0;
            this.lastErrorFrame = -1;

            // 次のフレームへ移動
            setTimeout(() => {
                this.loadFrame(frameNumber + 1, false);
            }, 1000);
        } else {
            showNotification('最後のフレームに到達しました', 'info');
        }
    }

    // フレーム読み込みリトライ
    retryFrameLoad(frameNumber, skipDisplay) {
        // スキップ中でエラーが続く場合は次のフレームへスキップ
        if (skipDisplay && this.isSkipping && frameNumber < this.totalFrames - 1 && this.frameLoadRetryCount >= 2) {
            console.log('🚀 スキップ中のため、エラーフレームをスキップして次のフレームへ移動');
            this.frameLoadRetryCount = 0;
            this.lastErrorFrame = -1;
            this.loadFrame(frameNumber + 1, true);
        } else {
            // 通常の再試行
            this.loadFrame(frameNumber, skipDisplay);
        }
    }

    // キャンバスの画像データを取得（縮小版）
    getCanvasImageData() {
        // 比較用に縮小したデータを取得（高速化のため）
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        const scale = 0.1;  // 10%に縮小

        tempCanvas.width = this.canvas.width * scale;
        tempCanvas.height = this.canvas.height * scale;

        tempCtx.drawImage(
            this.canvas,
            0, 0, this.canvas.width, this.canvas.height,
            0, 0, tempCanvas.width, tempCanvas.height
        );

        return tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
    }

    // フレーム類似度を計算
    calculateFrameSimilarity(imgData1, imgData2) {
        if (!imgData1 || !imgData2 ||
            imgData1.width !== imgData2.width ||
            imgData1.height !== imgData2.height) {
            return 0;
        }

        const data1 = imgData1.data;
        const data2 = imgData2.data;
        let diff = 0;

        // ピクセルごとの差分を計算
        for (let i = 0; i < data1.length; i += 4) {
            // RGBの差分を計算（アルファチャンネルは無視）
            diff += Math.abs(data1[i] - data2[i]);      // R
            diff += Math.abs(data1[i+1] - data2[i+1]);  // G
            diff += Math.abs(data1[i+2] - data2[i+2]);  // B
        }

        // 正規化して類似度に変換
        const maxDiff = data1.length * 255 * 0.75;  // RGB3チャンネル分
        const similarity = 1 - (diff / maxDiff);

        return similarity;
    }

    updateSimilarityDisplay(similarity, isSkipping = false) {
        // 類似度を画面に表示
        const similarityScore = document.getElementById('similarityScore');
        const currentThreshold = document.getElementById('currentThreshold');
        const similarityTarget = document.getElementById('similarityTarget');

        if (similarityScore) {
            const percentage = (similarity * 100).toFixed(1);
            similarityScore.textContent = percentage;

            // 類似度に応じて色を変更
            const alertElement = similarityScore.closest('.alert');
            if (alertElement) {
                if (similarity >= this.similarityThreshold) {
                    alertElement.classList.remove('alert-secondary', 'alert-warning');
                    alertElement.classList.add('alert-success');
                } else if (similarity >= 0.8) {
                    alertElement.classList.remove('alert-secondary', 'alert-success');
                    alertElement.classList.add('alert-warning');
                } else {
                    alertElement.classList.remove('alert-success', 'alert-warning');
                    alertElement.classList.add('alert-secondary');
                }
            }
        }

        if (similarityTarget) {
            // 常に前フレームとの比較を表示
            similarityTarget.textContent = '（前フレームとの比較）';
        }

        if (currentThreshold) {
            currentThreshold.textContent = (this.similarityThreshold * 100).toFixed(0);
        }
    }

    async labelCurrentFrame(isGameScene) {
        if (!this.sessionId || this.isSkipping) return;

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

                // ラベル情報を保存（類似フレームへの自動ラベル付与用）
                this.lastLabeledFrame = {
                    frameNumber: this.currentFrame,
                    isGameScene: isGameScene
                };
                // 現在のフレームデータも保存
                this.lastLabeledFrameData = this.getCanvasImageData();

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
        // 未ラベル検索中フラグを設定
        this.searchingUnlabeled = true;
        this.disableLabelingButtons(true);

        try {
            const response = await fetch(
                `/api/scene_labeling/sessions/${this.sessionId}/next_unlabeled?start_from=${this.currentFrame}`
            );

            if (response.ok) {
                const result = await response.json();
                if (result.next_frame !== null) {
                    await this.loadFrame(result.next_frame);
                    // loadFrameで結果を確認して、必要なら再度検索
                } else {
                    this.searchingUnlabeled = false;
                    this.disableLabelingButtons(false);
                    showNotification('未ラベルフレームがありません', 'info');
                }
            } else {
                this.searchingUnlabeled = false;
                this.disableLabelingButtons(false);
                showNotification('エラーが発生しました', 'error');
            }
        } catch (error) {
            console.error('Jump error:', error);
            this.searchingUnlabeled = false;
            this.disableLabelingButtons(false);
            showNotification('通信エラーが発生しました', 'error');
        }
    }

    // フレーム間隔でスキップする機能を追加（類似フレームの多い動画用）
    async skipToNextInterval(interval = 30) {
        // 現在のフレームから指定間隔でジャンプして、未ラベルフレームを探す
        let nextFrame = this.currentFrame + interval;

        // 最大10回試行
        for (let i = 0; i < 10 && nextFrame < this.totalFrames; i++) {
            try {
                const response = await fetch(
                    `/api/scene_labeling/sessions/${this.sessionId}/frame/${nextFrame}`
                );

                if (response.ok) {
                    const data = await response.json();
                    // ラベルがない（未ラベル）フレームが見つかった
                    if (!data.label) {
                        await this.loadFrame(nextFrame);
                        return;
                    }
                }
            } catch (error) {
                console.error('Frame check error:', error);
            }

            nextFrame += interval;
        }

        // 未ラベルフレームが見つからなかった場合は、通常の次の未ラベルフレームへジャンプ
        await this.jumpToNextUnlabeled();
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
            // 新しいフレームに移動する際はリトライカウントをリセット
            if (newFrame !== this.lastErrorFrame) {
                this.frameLoadRetryCount = 0;
            }
            this.loadFrame(newFrame);
        }
    }

    jumpToFrame(frameNumber) {
        if (frameNumber >= 0 && frameNumber < this.totalFrames) {
            // 新しいフレームにジャンプする際はリトライカウントをリセット
            if (frameNumber !== this.lastErrorFrame) {
                this.frameLoadRetryCount = 0;
            }
            this.loadFrame(frameNumber);
        }
    }

    handleKeyPress(event) {
        // 入力フィールドにフォーカスがある場合は無視
        if (event.target.tagName === 'INPUT') return;

        // キーボードショートカットが無効化されている場合は無視
        if (this.keyboardDisabled) return;

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
            case 's':  // Sキーで30フレームスキップ
                event.preventDefault();
                this.skipToNextInterval(30);
                break;
            case 'd':  // Dキーで60フレームスキップ
                event.preventDefault();
                this.skipToNextInterval(60);
                break;
            case 'escape':  // ESCキーでスキップ中断
                event.preventDefault();
                if (this.isSkipping) {
                    this.stopSkipping();
                }
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

    // ラベリングボタンの有効/無効を切り替え
    disableLabelingButtons(disabled) {
        document.getElementById('labelGame').disabled = disabled;
        document.getElementById('labelNonGame').disabled = disabled;
        document.getElementById('batchLabelGame').disabled = disabled;
        document.getElementById('batchLabelNonGame').disabled = disabled;

        // キーボードショートカットも無効化
        if (disabled) {
            this.keyboardDisabled = true;
        } else {
            this.keyboardDisabled = false;
        }
    }

    clearLabelIndicator() {
        this.canvas.style.border = '1px solid #dee2e6';
        const overlay = document.getElementById('labelOverlay');
        if (overlay) {
            overlay.remove();
        }
    }

    // スキップ時の高速類似度チェック
    async checkSimilarityAndSkip(result, frameNumber) {
        // 既にラベルがある場合はスキップ終了
        if (result.label) {
            this.endSkipping();
            // 画面を更新して表示
            this.loadFrame(frameNumber, false);
            return;
        }

        // 類似度をチェック（スキップ中でも必要）
        if (this.previousFrameData && this.lastLabeledFrame) {
            // 現在のフレームの画像データを取得
            const currentFrameData = this.getCanvasImageData();

            // 前フレームとの類似度を計算
            const similarity = this.calculateFrameSimilarity(
                this.previousFrameData,
                currentFrameData
            );

            console.log(`スキップ中: フレーム ${frameNumber} の前フレームとの類似度: ${(similarity * 100).toFixed(1)}% (閾値: ${(this.similarityThreshold * 100).toFixed(0)}%)`);

            // 類似度を画面に表示
            this.updateSimilarityDisplay(similarity);

            // 現在のフレームデータを次回比較用に保存
            this.previousFrameData = currentFrameData;

            // 類似度が閾値未満の場合はスキップ終了
            if (similarity < this.similarityThreshold) {
                console.log(`類似度が閾値未満のためスキップを終了: ${(similarity * 100).toFixed(1)}% < ${(this.similarityThreshold * 100).toFixed(0)}%`);

                // スキップ終了前にフラグを設定（再スキップを防ぐ）
                this.skipJustEnded = true;  // スキップが終了したことをマーク

                this.endSkipping();
                // 画面を更新して表示
                this.loadFrame(frameNumber, false);
                return;
            }

            // 類似度が高い場合は自動ラベル付与して継続
            await this.autoLabelSimilarFrame(frameNumber, this.lastLabeledFrame.isGameScene);
            this.skipCount++;
            this.updateSkipCount();

            // 次のフレームへ
            if (this.currentFrame + 1 < this.totalFrames) {
                setTimeout(() => {
                    this.loadFrame(this.currentFrame + 1, true);
                }, 5);  // さらに短縮
            } else {
                this.endSkipping();
            }
        } else {
            // ラベル付けフレームがない場合はスキップ終了
            this.endSkipping();
            // 画面を更新して表示
            this.loadFrame(frameNumber, false);
        }
    }

    // スキップ処理を終了
    endSkipping() {
        this.isSkipping = false;
        this.disableLabelingButtons(false);
        this.hideSkippingUI();

        // スキップ終了時も比較データをクリアしない
        // （通常のスキップ終了では次のフレームとの比較を継続するため）

        if (this.skipCount > 0) {
            showNotification(
                `${this.skipCount}個の類似フレームをスキップし、同じラベルを付与しました`,
                'info'
            );
            this.skipCount = 0;
        }
    }

    // スキップを手動で中断
    stopSkipping() {
        console.log('スキップを手動で中断します');
        this.isSkipping = false;
        this.disableLabelingButtons(false);
        this.hideSkippingUI();

        // スキップ停止時にラベル情報をクリア（前フレームデータは保持）
        this.lastLabeledFrameData = null;
        this.lastLabeledFrame = null;

        if (this.skipCount > 0) {
            showNotification(
                `スキップを中断しました (${this.skipCount}個のフレームをスキップ済み)`,
                'warning'
            );
            this.skipCount = 0;
        }

        // 現在のフレームを再読み込み（スキップなし）
        this.loadFrame(this.currentFrame, false);
    }

    // スキップ中のUI表示
    showSkippingUI() {
        document.getElementById('stopSkipping').style.display = 'block';
        document.getElementById('skipProgress').style.display = 'block';
        document.getElementById('labelGame').style.display = 'none';
        document.getElementById('labelNonGame').style.display = 'none';
        this.updateSkipCount();
    }

    // スキップUIを非表示
    hideSkippingUI() {
        document.getElementById('stopSkipping').style.display = 'none';
        document.getElementById('skipProgress').style.display = 'none';
        document.getElementById('labelGame').style.display = 'block';
        document.getElementById('labelNonGame').style.display = 'block';
    }

    // スキップカウント更新
    updateSkipCount() {
        const skipCountElement = document.getElementById('skipCount');
        if (skipCountElement) {
            skipCountElement.textContent = this.skipCount;
        }
    }

    // 類似フレームに自動的にラベルを付与
    async autoLabelSimilarFrame(frameNumber, isGameScene) {
        try {
            const response = await fetch(`/api/scene_labeling/sessions/${this.sessionId}/label`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    frame_number: frameNumber,
                    is_game_scene: isGameScene,
                    annotator: 'auto-similar'  // 類似フレームの自動ラベルであることを示す
                })
            });

            if (response.ok) {
                const result = await response.json();
                this.updateStatistics(result.statistics);
                console.log(`類似フレーム ${frameNumber} に自動ラベル付与: ${isGameScene ? '対局画面' : '非対局画面'}`);

                // 自動ラベル付与したフレームも記録
                this.lastLabeledFrame = {
                    frameNumber: frameNumber,
                    isGameScene: isGameScene
                };
                // 注: このフレームの画像データはスキップ中なので取得できない
                // lastLabeledFrameDataは元のラベル付けフレームのものを維持
            }
        } catch (error) {
            console.error('Auto label similar frame error:', error);
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
                // WebSocket接続を確立
                if (typeof window.socket === 'undefined') {
                    window.socket = io();
                }
                const socket = window.socket;

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

    async showSessionList() {
        console.log('showSessionList() が呼び出されました');
        try {
            // セッション一覧を取得
            console.log('API呼び出し: /api/scene_labeling/sessions');
            const response = await fetch('/api/scene_labeling/sessions');
            console.log('API応答:', response.status);

            if (response.ok) {
                const result = await response.json();
                console.log('取得したセッション:', result);
                this.displaySessionList(result.sessions);
                document.getElementById('sessionListArea').style.display = 'block';
            } else {
                console.error('APIエラー:', response.status);
                showNotification('セッション一覧の取得に失敗しました', 'error');
            }
        } catch (error) {
            console.error('Session list error:', error);
            showNotification('セッション一覧取得エラー', 'error');
        }
    }

    displaySessionList(sessions) {
        const tbody = document.getElementById('sessionTableBody');
        tbody.innerHTML = '';

        if (sessions.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="7" class="text-center text-muted">
                        セッションがありません
                    </td>
                </tr>
            `;
            return;
        }

        sessions.forEach(session => {
            const row = document.createElement('tr');
            const videoName = session.video_path ? session.video_path.split('/').pop() : session.video_id || '不明';
            const progress = session.statistics ? (session.statistics.progress * 100).toFixed(1) : '0';
            const labeledFrames = session.statistics ? session.statistics.labeled_frames : 0;
            const totalFrames = session.statistics ? session.statistics.total_frames : 0;
            const updatedAt = session.updated_at ? new Date(session.updated_at).toLocaleString() : '-';
            const status = session.is_active ?
                '<span class="badge bg-success">アクティブ</span>' :
                '<span class="badge bg-secondary">保存済み</span>';

            row.innerHTML = `
                <td>${videoName}</td>
                <td>${totalFrames}</td>
                <td>${labeledFrames}</td>
                <td>
                    <div class="progress" style="width: 100px;">
                        <div class="progress-bar" style="width: ${progress}%">${progress}%</div>
                    </div>
                </td>
                <td>${updatedAt}</td>
                <td>${status}</td>
                <td>
                    <div class="btn-group" role="group">
                        <button class="btn btn-sm btn-primary resume-session"
                                data-session-id="${session.session_id}"
                                data-video-path="${session.video_path}">
                            再開
                        </button>
                        <button class="btn btn-sm btn-danger delete-session"
                                data-session-id="${session.session_id}"
                                data-video-name="${videoName}">
                            削除
                        </button>
                    </div>
                </td>
            `;

            tbody.appendChild(row);
        });

        // 再開ボタンのイベントリスナー
        document.querySelectorAll('.resume-session').forEach(button => {
            button.addEventListener('click', (e) => {
                const sessionId = e.target.getAttribute('data-session-id');
                const videoPath = e.target.getAttribute('data-video-path');
                this.resumeSession(sessionId, videoPath);
            });
        });

        // 削除ボタンのイベントリスナー
        document.querySelectorAll('.delete-session').forEach(button => {
            button.addEventListener('click', (e) => {
                const sessionId = e.target.getAttribute('data-session-id');
                const videoName = e.target.getAttribute('data-video-name');
                this.deleteSession(sessionId, videoName);
            });
        });
    }

    hideSessionList() {
        document.getElementById('sessionListArea').style.display = 'none';
    }

    async deleteSession(sessionId, videoName) {
        if (!confirm(`セッション「${videoName}」を削除しますか？\nこの操作は取り消せません。`)) {
            return;
        }

        try {
            const response = await fetch(`/api/scene_labeling/sessions/${sessionId}`, {
                method: 'DELETE'
            });

            if (response.ok) {
                const result = await response.json();
                showNotification(`セッションを削除しました: ${videoName}`, 'success');

                // 削除したセッションが現在のセッションの場合
                if (this.sessionId === sessionId) {
                    this.sessionId = null;
                    this.videoInfo = null;
                    document.getElementById('labelingArea').style.display = 'none';
                }

                // セッション一覧を再読み込み
                this.showSessionList();
            } else {
                const error = await response.json();
                showNotification(`削除に失敗しました: ${error.error || 'Unknown error'}`, 'error');
            }
        } catch (error) {
            console.error('セッション削除エラー:', error);
            showNotification('セッション削除エラー', 'error');
        }
    }

    async resumeSession(sessionId, videoPath) {
        try {
            const response = await fetch('/api/scene_labeling/sessions', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    video_path: videoPath,
                    session_id: sessionId
                })
            });

            if (response.ok) {
                const result = await response.json();
                this.sessionId = sessionId;
                this.videoInfo = result.video_info;
                this.totalFrames = result.video_info.total_frames;

                // UIを更新
                document.getElementById('totalFrames').textContent = this.totalFrames;
                document.getElementById('frameSlider').max = this.totalFrames - 1;
                document.getElementById('labelingArea').style.display = 'block';
                document.getElementById('sessionListArea').style.display = 'none';

                // 統計情報を更新
                this.updateStatistics(result.statistics);

                // 最初のフレームを読み込み
                await this.loadFrame(0);

                const message = result.is_resumed ?
                    'セッションを再開しました' :
                    'セッションを開始しました';
                showNotification(message, 'success');
            } else {
                showNotification('セッション再開に失敗しました', 'error');
            }
        } catch (error) {
            console.error('Resume session error:', error);
            showNotification('セッション再開エラー', 'error');
        }
    }
}

// アプリケーションを初期化
document.addEventListener('DOMContentLoaded', () => {
    window.sceneLabelingApp = new SceneLabelingApp();
});
