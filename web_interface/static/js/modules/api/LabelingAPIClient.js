/**
 * ラベリングAPI通信クライアント
 */
export class LabelingAPIClient {
    constructor(baseURL = '') {
        this.baseURL = baseURL;
        this.defaultHeaders = {
            'Content-Type': 'application/json'
        };
    }

    /**
     * APIリクエストの基本メソッド
     */
    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const config = {
            ...options,
            headers: {
                ...this.defaultHeaders,
                ...options.headers
            }
        };

        try {
            const response = await fetch(url, config);

            if (!response.ok) {
                const error = await response.json().catch(() => ({ error: 'Unknown error' }));
                throw new Error(error.error || `HTTP ${response.status}: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error(`API request failed: ${endpoint}`, error);
            throw error;
        }
    }

    /**
     * セッションを作成
     */
    async createSession(videoPath) {
        return this.request('/api/labeling/sessions/', {
            method: 'POST',
            body: JSON.stringify({ video_path: videoPath })
        });
    }

    /**
     * セッション詳細を取得
     */
    async getSession(sessionId) {
        return this.request(`/api/labeling/sessions/${sessionId}`);
    }

    /**
     * セッション一覧を取得
     */
    async listSessions() {
        return this.request('/api/labeling/sessions/');
    }

    /**
     * 手牌領域設定を取得
     */
    async getHandAreas(sessionId) {
        return this.request(`/api/labeling/hand-areas/${sessionId}`);
    }

    /**
     * 手牌領域を設定
     */
    async setHandAreas(sessionId, regions) {
        return this.request(`/api/labeling/hand-areas/${sessionId}`, {
            method: 'PUT',
            body: JSON.stringify({ regions })
        });
    }

    /**
     * フレームを抽出
     */
    async extractFrames(sessionId, interval = 1.0, startTime = 0, endTime = null) {
        return this.request(`/api/labeling/frames/${sessionId}/extract`, {
            method: 'POST',
            body: JSON.stringify({ interval, start_time: startTime, end_time: endTime })
        });
    }

    /**
     * フレームを取得
     */
    async getFrame(sessionId, frameNumber) {
        const response = await fetch(`/api/labeling/frames/${sessionId}/${frameNumber}`);
        if (!response.ok) {
            throw new Error(`Failed to get frame: ${response.statusText}`);
        }
        return response.blob();
    }

    /**
     * 牌を分割
     */
    async splitTiles(sessionId, frameNumber, player = 'bottom') {
        return this.request(`/api/labeling/frames/${sessionId}/${frameNumber}/tiles`, {
            method: 'POST',
            body: JSON.stringify({ player })
        });
    }

    /**
     * アノテーションを追加
     */
    async addAnnotation(sessionId, frameNumber, player, tiles) {
        return this.request(`/api/labeling/annotations/${sessionId}`, {
            method: 'POST',
            body: JSON.stringify({ frame_number: frameNumber, player, tiles })
        });
    }

    /**
     * アノテーションをエクスポート
     */
    async exportAnnotations(sessionId, format = 'coco') {
        return this.request(`/api/labeling/annotations/${sessionId}/export?format=${format}`);
    }

    /**
     * 動画を読み込み
     */
    async loadVideo(videoPath, videoId = null) {
        return this.request('/api/labeling/load_video', {
            method: 'POST',
            body: JSON.stringify({ video_path: videoPath, video_id: videoId })
        });
    }

    /**
     * データをエクスポート
     */
    async exportData(sessionId, format = 'json') {
        return this.request('/api/labeling/export', {
            method: 'POST',
            body: JSON.stringify({ session_id: sessionId, format })
        });
    }

    /**
     * 動画一覧を取得
     */
    async getVideoList() {
        return this.request('/api/videos');
    }

    /**
     * 動画情報を取得
     */
    async getVideoInfo(videoId) {
        return this.request(`/api/videos/${videoId}`);
    }

    /**
     * フレーム一覧を取得
     */
    async getFrameList(videoId) {
        return this.request(`/api/videos/${videoId}/frames`);
    }

    /**
     * フレームのアノテーションを取得
     */
    async getFrameAnnotations(frameId) {
        return this.request(`/api/frames/${frameId}/annotations`);
    }

    /**
     * フレームのアノテーションを保存
     */
    async saveFrameAnnotations(frameId, annotations) {
        return this.request(`/api/frames/${frameId}/annotations`, {
            method: 'POST',
            body: JSON.stringify({ annotations })
        });
    }

    /**
     * フレーム抽出を開始
     */
    async extractFrames(videoPath, config) {
        return this.request('/api/extract_frames', {
            method: 'POST',
            body: JSON.stringify({ video_path: videoPath, config })
        });
    }

    /**
     * アップロード進捗を監視
     */
    async *uploadWithProgress(file, endpoint, onProgress) {
        const formData = new FormData();
        formData.append('file', file);

        const xhr = new XMLHttpRequest();

        // 進捗イベント
        xhr.upload.addEventListener('progress', (e) => {
            if (e.lengthComputable && onProgress) {
                const progress = (e.loaded / e.total) * 100;
                onProgress(progress);
            }
        });

        // Promiseでラップ
        const uploadPromise = new Promise((resolve, reject) => {
            xhr.addEventListener('load', () => {
                if (xhr.status >= 200 && xhr.status < 300) {
                    try {
                        resolve(JSON.parse(xhr.responseText));
                    } catch (e) {
                        reject(new Error('Invalid JSON response'));
                    }
                } else {
                    reject(new Error(`HTTP ${xhr.status}: ${xhr.statusText}`));
                }
            });

            xhr.addEventListener('error', () => reject(new Error('Network error')));
            xhr.addEventListener('abort', () => reject(new Error('Upload aborted')));
        });

        xhr.open('POST', `${this.baseURL}${endpoint}`);
        xhr.send(formData);

        return await uploadPromise;
    }

    /**
     * エラーをフォーマット
     */
    formatError(error) {
        if (error.response) {
            // サーバーエラー
            return `サーバーエラー: ${error.response.status} - ${error.response.data?.error || error.message}`;
        } else if (error.request) {
            // ネットワークエラー
            return 'ネットワークエラー: サーバーに接続できません';
        } else {
            // その他のエラー
            return `エラー: ${error.message}`;
        }
    }

    /**
     * リトライ機能付きリクエスト
     */
    async requestWithRetry(endpoint, options = {}, maxRetries = 3, delay = 1000) {
        let lastError;

        for (let i = 0; i < maxRetries; i++) {
            try {
                return await this.request(endpoint, options);
            } catch (error) {
                lastError = error;
                if (i < maxRetries - 1) {
                    // 指数バックオフ
                    await new Promise(resolve => setTimeout(resolve, delay * Math.pow(2, i)));
                }
            }
        }

        throw lastError;
    }

    /**
     * バッチリクエスト
     */
    async batchRequest(requests) {
        const promises = requests.map(({ endpoint, options }) =>
            this.request(endpoint, options).catch(error => ({ error, endpoint }))
        );

        return await Promise.all(promises);
    }

    /**
     * キャンセル可能なリクエスト
     */
    createCancellableRequest(endpoint, options = {}) {
        const controller = new AbortController();

        const promise = this.request(endpoint, {
            ...options,
            signal: controller.signal
        });

        return {
            promise,
            cancel: () => controller.abort()
        };
    }
}

// シングルトンインスタンスをエクスポート
export const labelingAPI = new LabelingAPIClient();
