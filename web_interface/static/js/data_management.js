/**
 * 麻雀牌検出システム - データ管理機能
 */

// グローバル変数
let uploadedVideos = [];
let datasetVersions = [];
let currentUploadProgress = {};
let selectedExportFormat = null;

// 初期化
document.addEventListener('DOMContentLoaded', function() {
    initializeDataManagement();
});

/**
 * データ管理機能の初期化
 */
function initializeDataManagement() {
    // イベントリスナー設定
    setupEventListeners();
    
    // データを読み込み
    loadInitialData();
    
    // WebSocketイベント設定
    setupWebSocketEvents();
    
    console.log('データ管理機能が初期化されました');
}

/**
 * イベントリスナー設定
 */
function setupEventListeners() {
    // ファイルアップロード
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('video-file-input');
    
    uploadArea.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('dragover', onDragOver);
    uploadArea.addEventListener('dragleave', onDragLeave);
    uploadArea.addEventListener('drop', onDrop);
    
    fileInput.addEventListener('change', onFileSelect);
    
    // エクスポート形式選択
    document.querySelectorAll('.format-option').forEach(option => {
        option.addEventListener('click', onFormatSelect);
    });
    
    // モーダルイベント
    document.getElementById('createVersionModal').addEventListener('show.bs.modal', onCreateVersionModalShow);
    document.getElementById('exportModal').addEventListener('show.bs.modal', onExportModalShow);
    document.getElementById('videoDetailModal').addEventListener('show.bs.modal', onVideoDetailModalShow);
    
    // ボタンイベント
    document.getElementById('create-version-btn').addEventListener('click', createDatasetVersion);
    document.getElementById('start-export-btn').addEventListener('click', startExport);
    document.getElementById('extract-frames-btn').addEventListener('click', extractFramesFromVideo);
    document.getElementById('delete-video-btn').addEventListener('click', deleteVideo);
}

/**
 * WebSocketイベント設定
 */
function setupWebSocketEvents() {
    socket.on('frame_extraction_progress', function(data) {
        updateExtractionProgress(data);
    });
    
    socket.on('export_progress', function(data) {
        updateExportProgress(data);
    });
    
    socket.on('version_creation_progress', function(data) {
        updateVersionCreationProgress(data);
    });
}

/**
 * 初期データを読み込み
 */
async function loadInitialData() {
    try {
        showLoading('データを読み込み中...');
        
        // 動画一覧を読み込み
        await loadVideoList();
        
        // データセット統計を読み込み
        await loadDatasetStatistics();
        
        // データセットバージョンを読み込み
        await loadDatasetVersions();
        
        hideLoading();
        
    } catch (error) {
        console.error('初期データ読み込みエラー:', error);
        showNotification('データの読み込みに失敗しました', 'error');
        hideLoading();
    }
}

/**
 * 動画一覧を読み込み
 */
async function loadVideoList() {
    try {
        const response = await apiRequest('/api/videos');
        uploadedVideos = response.videos || [];
        
        displayVideoList(uploadedVideos);
        
    } catch (error) {
        console.error('動画一覧読み込みエラー:', error);
        throw error;
    }
}

/**
 * 動画一覧を表示
 */
function displayVideoList(videos) {
    const container = document.getElementById('video-list');
    
    if (videos.length === 0) {
        container.innerHTML = `
            <div class="text-center text-muted p-4">
                <i class="fas fa-video fa-3x mb-3"></i>
                <div>アップロードされた動画はありません</div>
                <div class="mt-3">
                    <small>左側のエリアから動画をアップロードしてください</small>
                </div>
            </div>
        `;
        return;
    }
    
    container.innerHTML = '';
    
    videos.forEach(video => {
        const item = createVideoListItem(video);
        container.appendChild(item);
    });
}

/**
 * 動画リストアイテムを作成
 */
function createVideoListItem(video) {
    const item = document.createElement('div');
    item.className = 'list-group-item';
    
    item.innerHTML = `
        <div class="row align-items-center">
            <div class="col-md-3">
                <img src="${video.thumbnail_path || '/static/images/video-placeholder.png'}" 
                     class="video-thumbnail" alt="動画サムネイル">
            </div>
            <div class="col-md-6">
                <h6 class="mb-1">${video.name}</h6>
                <div class="file-info">
                    <div>解像度: ${video.width}x${video.height}</div>
                    <div>時間: ${formatDuration(video.duration)}</div>
                    <div>FPS: ${video.fps.toFixed(1)}</div>
                    <div>サイズ: ${formatFileSize(video.file_size || 0)}</div>
                </div>
                <small class="text-muted">
                    アップロード: ${formatDateTime(video.upload_time)}
                </small>
            </div>
            <div class="col-md-3">
                <div class="mb-2">
                    <small class="text-muted">フレーム数</small>
                    <div class="fw-bold">${video.extracted_frames || 0}</div>
                </div>
                <div class="btn-group-vertical w-100">
                    <button class="btn btn-sm btn-outline-primary" 
                            onclick="showVideoDetail('${video.id}')">
                        <i class="fas fa-eye"></i> 詳細
                    </button>
                    <button class="btn btn-sm btn-outline-info" 
                            onclick="extractFrames('${video.id}')">
                        <i class="fas fa-images"></i> フレーム抽出
                    </button>
                </div>
            </div>
        </div>
    `;
    
    return item;
}

/**
 * データセット統計を読み込み
 */
async function loadDatasetStatistics() {
    try {
        const response = await apiRequest('/api/dataset/statistics');
        
        document.getElementById('total-videos').textContent = formatNumber(response.video_count || 0);
        document.getElementById('total-frames').textContent = formatNumber(response.frame_count || 0);
        document.getElementById('total-annotations').textContent = formatNumber(response.tile_count || 0);
        
    } catch (error) {
        console.error('データセット統計読み込みエラー:', error);
        throw error;
    }
}

/**
 * データセットバージョンを読み込み
 */
async function loadDatasetVersions() {
    try {
        const response = await apiRequest('/api/dataset/versions');
        datasetVersions = response;
        
        document.getElementById('dataset-versions-count').textContent = formatNumber(datasetVersions.length);
        
        displayDatasetVersions(datasetVersions);
        
    } catch (error) {
        console.error('データセットバージョン読み込みエラー:', error);
        throw error;
    }
}

/**
 * データセットバージョンを表示
 */
function displayDatasetVersions(versions) {
    const container = document.getElementById('dataset-versions-container');
    
    if (versions.length === 0) {
        container.innerHTML = `
            <div class="text-center text-muted py-4">
                <i class="fas fa-database fa-3x mb-3"></i>
                <div>データセットバージョンはありません</div>
                <div class="mt-3">
                    <small>「新しいバージョン作成」ボタンから作成してください</small>
                </div>
            </div>
        `;
        return;
    }
    
    container.innerHTML = '';
    
    versions.forEach(version => {
        const card = createVersionCard(version);
        container.appendChild(card);
    });
}

/**
 * バージョンカードを作成
 */
function createVersionCard(version) {
    const card = document.createElement('div');
    card.className = 'col-md-4 mb-3';
    
    card.innerHTML = `
        <div class="card dataset-version-card h-100">
            <div class="card-header d-flex justify-content-between align-items-center">
                <h6 class="card-title mb-0">${version.version}</h6>
                <small class="text-muted">${formatDateTime(version.created_at)}</small>
            </div>
            <div class="card-body">
                <div class="row mb-3">
                    <div class="col-6">
                        <div class="text-center">
                            <div class="h5 mb-0 text-primary">${formatNumber(version.frame_count)}</div>
                            <small class="text-muted">フレーム</small>
                        </div>
                    </div>
                    <div class="col-6">
                        <div class="text-center">
                            <div class="h5 mb-0 text-success">${formatNumber(version.tile_count)}</div>
                            <small class="text-muted">牌</small>
                        </div>
                    </div>
                </div>
                
                ${version.description ? `
                    <div class="mb-3">
                        <small class="text-muted">説明</small>
                        <div class="small">${version.description}</div>
                    </div>
                ` : ''}
                
                <div class="d-grid gap-2">
                    <button class="btn btn-sm btn-outline-info" 
                            onclick="exportVersion('${version.id}')">
                        <i class="fas fa-download"></i> エクスポート
                    </button>
                    <button class="btn btn-sm btn-outline-danger" 
                            onclick="deleteVersion('${version.id}')">
                        <i class="fas fa-trash"></i> 削除
                    </button>
                </div>
            </div>
        </div>
    `;
    
    return card;
}

/**
 * ドラッグオーバー
 */
function onDragOver(event) {
    event.preventDefault();
    event.currentTarget.classList.add('dragover');
}

/**
 * ドラッグリーブ
 */
function onDragLeave(event) {
    event.preventDefault();
    event.currentTarget.classList.remove('dragover');
}

/**
 * ドロップ
 */
function onDrop(event) {
    event.preventDefault();
    event.currentTarget.classList.remove('dragover');
    
    const files = Array.from(event.dataTransfer.files);
    const videoFiles = files.filter(file => file.type.startsWith('video/'));
    
    if (videoFiles.length > 0) {
        uploadFiles(videoFiles);
    } else {
        showNotification('動画ファイルを選択してください', 'warning');
    }
}

/**
 * ファイル選択
 */
function onFileSelect(event) {
    const files = Array.from(event.target.files);
    if (files.length > 0) {
        uploadFiles(files);
    }
}

/**
 * ファイルをアップロード
 */
async function uploadFiles(files) {
    for (const file of files) {
        await uploadSingleFile(file);
    }
}

/**
 * 単一ファイルをアップロード
 */
async function uploadSingleFile(file) {
    const formData = new FormData();
    formData.append('video', file);
    
    // 進捗表示を開始
    showUploadProgress(file.name);
    
    try {
        const xhr = new XMLHttpRequest();
        
        // 進捗イベント
        xhr.upload.addEventListener('progress', (event) => {
            if (event.lengthComputable) {
                const progress = (event.loaded / event.total) * 100;
                updateUploadProgress(progress, `アップロード中: ${file.name}`);
            }
        });
        
        // 完了イベント
        xhr.addEventListener('load', () => {
            if (xhr.status === 200) {
                const response = JSON.parse(xhr.responseText);
                if (response.success) {
                    updateUploadProgress(100, `完了: ${file.name}`);
                    showNotification(`${file.name} のアップロードが完了しました`, 'success');
                    
                    // 自動フレーム抽出が有効な場合
                    const autoExtract = document.getElementById('auto-extract').checked;
                    if (autoExtract) {
                        setTimeout(() => {
                            extractFramesFromUploadedVideo(response.video_info);
                        }, 1000);
                    }
                    
                    // 動画一覧を更新
                    setTimeout(() => {
                        loadVideoList();
                        hideUploadProgress();
                    }, 2000);
                } else {
                    throw new Error(response.error || 'アップロードに失敗しました');
                }
            } else {
                throw new Error(`HTTP error! status: ${xhr.status}`);
            }
        });
        
        // エラーイベント
        xhr.addEventListener('error', () => {
            throw new Error('ネットワークエラーが発生しました');
        });
        
        // リクエスト送信
        xhr.open('POST', '/api/upload_video');
        xhr.send(formData);
        
    } catch (error) {
        console.error('アップロードエラー:', error);
        showNotification(`${file.name} のアップロードに失敗しました: ${error.message}`, 'error');
        hideUploadProgress();
    }
}

/**
 * アップロード進捗を表示
 */
function showUploadProgress(filename) {
    const container = document.getElementById('upload-progress-container');
    const progressBar = document.getElementById('upload-progress-bar');
    const progressText = document.getElementById('upload-progress-text');
    const statusText = document.getElementById('upload-status');
    
    container.style.display = 'block';
    progressBar.style.width = '0%';
    progressText.textContent = '0%';
    statusText.textContent = `準備中: ${filename}`;
}

/**
 * アップロード進捗を更新
 */
function updateUploadProgress(progress, status) {
    const progressBar = document.getElementById('upload-progress-bar');
    const progressText = document.getElementById('upload-progress-text');
    const statusText = document.getElementById('upload-status');
    
    progressBar.style.width = `${progress}%`;
    progressText.textContent = `${Math.round(progress)}%`;
    statusText.textContent = status;
}

/**
 * アップロード進捗を非表示
 */
function hideUploadProgress() {
    const container = document.getElementById('upload-progress-container');
    container.style.display = 'none';
}

/**
 * アップロードされた動画からフレーム抽出
 */
async function extractFramesFromUploadedVideo(videoInfo) {
    const config = {
        interval_seconds: parseFloat(document.getElementById('extraction-interval').value),
        quality_threshold: parseFloat(document.getElementById('quality-threshold').value),
        max_frames: parseInt(document.getElementById('max-frames').value),
        resize_width: parseInt(document.getElementById('resize-width').value)
    };
    
    try {
        const response = await apiRequest('/api/extract_frames', {
            method: 'POST',
            body: JSON.stringify({
                video_path: videoInfo.filepath,
                config: config
            })
        });
        
        if (response.session_id) {
            socket.emit('join_session', { session_id: response.session_id });
            showNotification('フレーム抽出を開始しました', 'info');
        }
        
    } catch (error) {
        console.error('フレーム抽出エラー:', error);
        showNotification('フレーム抽出の開始に失敗しました', 'error');
    }
}

/**
 * 動画一覧を更新
 */
async function refreshVideoList() {
    try {
        await loadVideoList();
        showNotification('動画一覧を更新しました', 'success', 2000);
    } catch (error) {
        console.error('動画一覧更新エラー:', error);
        showNotification('動画一覧の更新に失敗しました', 'error');
    }
}

/**
 * 動画詳細を表示
 */
async function showVideoDetail(videoId) {
    try {
        const response = await apiRequest(`/api/videos/${videoId}`);
        const video = response.video;
        
        // 動画情報を設定
        document.getElementById('detail-filename').textContent = video.name;
        document.getElementById('detail-resolution').textContent = `${video.width}x${video.height}`;
        document.getElementById('detail-fps').textContent = `${video.fps.toFixed(1)} fps`;
        document.getElementById('detail-duration').textContent = formatDuration(video.duration);
        document.getElementById('detail-filesize').textContent = formatFileSize(video.file_size || 0);
        document.getElementById('detail-upload-time').textContent = formatDateTime(video.upload_time);
        
        // 動画プレビューを設定
        const videoPreview = document.getElementById('video-preview');
        videoPreview.src = video.path;
        
        // フレーム情報を読み込み
        await loadExtractedFramesInfo(videoId);
        
        // モーダルにvideoIdを保存
        const modal = document.getElementById('videoDetailModal');
        modal.dataset.videoId = videoId;
        
        // モーダルを表示
        const modalInstance = new bootstrap.Modal(modal);
        modalInstance.show();
        
    } catch (error) {
        console.error('動画詳細読み込みエラー:', error);
        showNotification('動画詳細の読み込みに失敗しました', 'error');
    }
}

/**
 * 抽出済みフレーム情報を読み込み
 */
async function loadExtractedFramesInfo(videoId) {
    try {
        const response = await apiRequest(`/api/videos/${videoId}/frames`);
        const frames = response.frames || [];
        
        const container = document.getElementById('extracted-frames-info');
        
        if (frames.length === 0) {
            container.innerHTML = '<div class="text-muted">抽出済みフレームはありません</div>';
        } else {
            const annotatedCount = frames.filter(f => f.tiles && f.tiles.length > 0).length;
            container.innerHTML = `
                <div class="row">
                    <div class="col-4">
                        <div class="text-center">
                            <div class="h6 mb-0">${frames.length}</div>
                            <small class="text-muted">総フレーム数</small>
                        </div>
                    </div>
                    <div class="col-4">
                        <div class="text-center">
                            <div class="h6 mb-0">${annotatedCount}</div>
                            <small class="text-muted">アノテーション済み</small>
                        </div>
                    </div>
                    <div class="col-4">
                        <div class="text-center">
                            <div class="h6 mb-0">${formatPercentage(annotatedCount / frames.length)}</div>
                            <small class="text-muted">完了率</small>
                        </div>
                    </div>
                </div>
            `;
        }
        
    } catch (error) {
        console.error('フレーム情報読み込みエラー:', error);
        document.getElementById('extracted-frames-info').innerHTML = 
            '<div class="text-danger">フレーム情報の読み込みに失敗しました</div>';
    }
}

/**
 * フレーム抽出
 */
function extractFrames(videoId) {
    const video = uploadedVideos.find(v => v.id === videoId);
    if (!video) return;
    
    extractFramesFromUploadedVideo({
        filepath: video.path
    });
}

/**
 * 動画詳細モーダルからフレーム抽出
 */
function extractFramesFromVideo() {
    const modal = document.getElementById('videoDetailModal');
    const videoId = modal.dataset.videoId;
    
    if (videoId) {
        extractFrames(videoId);
        
        // モーダルを閉じる
        const modalInstance = bootstrap.Modal.getInstance(modal);
        modalInstance.hide();
    }
}

/**
 * 動画を削除
 */
async function deleteVideo() {
    const modal = document.getElementById('videoDetailModal');
    const videoId = modal.dataset.videoId;
    
    if (!videoId) return;
    
    if (confirm('この動画を削除しますか？関連するフレームとアノテーションも削除されます。')) {
        try {
            await apiRequest(`/api/videos/${videoId}`, {
                method: 'DELETE'
            });
            
            showNotification('動画を削除しました', 'success');
            
            // モーダルを閉じる
            const modalInstance = bootstrap.Modal.getInstance(modal);
            modalInstance.hide();
            
            // 動画一覧を更新
            await loadVideoList();
            await loadDatasetStatistics();
            
        } catch (error) {
            console.error('動画削除エラー:', error);
            showNotification('動画の削除に失敗しました', 'error');
        }
    }
}

/**
 * 動画詳細モーダル表示時
 */
function onVideoDetailModalShow(event) {
    // 必要に応じて初期化処理
}

/**
 * データセットバージョン作成
 */
function createDatasetVersion() {
    const modal = new bootstrap.Modal(document.getElementById('createVersionModal'));
    modal.show();
}

/**
 * バージョン作成モーダル表示時
 */
function onCreateVersionModalShow(event) {
    // フォームをリセット
    document.getElementById('create-version-form').reset();
}

/**
 * バージョン作成実行
 */
async function createDatasetVersion() {
    const form = document.getElementById('create-version-form');
    const formData = getFormData(form);
    
    if (!formData['version-name']) {
        showNotification('バージョン名を入力してください', 'warning');
        return;
    }
    
    try {
        showLoading('データセットバージョンを作成中...');
        
        const response = await apiRequest('/api/dataset/create_version', {
            method: 'POST',
            body: JSON.stringify({
                version: formData['version-name'],
                description: formData['version-description'],
                include_all_data: formData['include-all-data']
            })
        });
        
        if (response.version_id) {
            showNotification('データセットバージョンを作成しました', 'success');
            
            // モーダルを閉じる
            const modal = bootstrap.Modal.getInstance(document.getElementById('createVersionModal'));
            modal.hide();
            
            // バージョン一覧を更新
            await loadDatasetVersions();
        }
        
        hideLoading();
        
    } catch (error) {
        console.error('バージョン作成エラー:', error);
        showNotification('バージョンの作成に失敗しました', 'error');
        hideLoading();
    }
}

/**
 * エクスポートモーダルを表示
 */
function showExportModal() {
    const modal = new bootstrap.Modal(document.getElementById('exportModal'));
    modal.show();
}

/**
 * エクスポートモーダル表示時
 */
function onExportModalShow(event) {
    // バージョン選択肢を更新
    const selector = document.getElementById('export-version-select');
    selector.innerHTML = '<option value="">選択してください...</option>';
    
    datasetVersions.forEach(version => {
        const option = document.createElement('option');
        option.value = version.id;
        option.textContent = `${version.version} (${version.frame_count}フレーム)`;
        selector.appendChild(option);
    });
    
    // フォーマット選択をリセット
    document.querySelectorAll('.format-option').forEach(option => {
        option.classList.remove('selected');
    });
    selectedExportFormat = null;
    document.getElementById('start-export-btn').disabled = true;
}

/**
 * エクスポート形式選択
 */
function onFormatSelect(event) {
    // 前の選択を解除
    document.querySelectorAll('.format-option').forEach(option => {
        option.classList.remove('selected');
    });
    
    // 新しい選択を設定
    event.currentTarget.classList.add('selected');
    selectedExportFormat = event.currentTarget.dataset.format;
    
    // エクスポートボタンの有効/無効を更新
    updateExportButton();
}

/**
 * エクスポートボタンを更新
 */
function updateExportButton() {
    const versionSelected = document.getElementById('export-version-select').value;
    const formatSelected = selectedExportFormat;
    
    document.getElementById('start-export-btn').disabled = !(versionSelected && formatSelected);
}

/**
 * エクスポート開始
 */
async function startExport() {
    const versionId = document.getElementById('export-version-select').value;
    const outputDir = document.getElementById('export-output-dir').value;
    
    if (!versionId || !selectedExportFormat) {
        showNotification('バージョンと形式を選択してください', 'warning');
        return;
    }
    
    try {
        const response = await apiRequest('/api/dataset/export', {
            method: 'POST',
            body: JSON.stringify({
                version_id: versionId,
                format: selectedExportFormat,
                output_dir: outputDir || null
            })
        });
        
        if (response.session_id) {
            socket.emit('join_session', { session_id: response.session_id });
            showNotification('エクスポートを開始しました', 'info');
            
            // モーダルを閉じる
            const modal = bootstrap.Modal.getInstance(document.getElementById('exportModal'));
            modal.hide();
        }
        
    } catch (error) {
        console.error('エクスポートエラー:', error);
        showNotification('エクスポートの開始に失敗しました', 'error');
    }
}

/**
 * バージョンをエクスポート
 */
function exportVersion(versionId) {
    document.getElementById('export-version-select').value = versionId;
    showExportModal();
}

/**
 * バージョンを削除
 */
async function deleteVersion(versionId) {
    const version = datasetVersions.find(v => v.id === versionId);
    if (!version) return;
    
    if (confirm(`データセットバージョン "${version.version}" を削除しますか？`)) {
        try {
            await apiRequest(`/api/dataset/versions/${versionId}`, {
                method: 'DELETE'
            });
            
            showNotification('データセットバージョンを削除しました', 'success');
            
            // バージョン一覧を更新
            await loadDatasetVersions();
            
        } catch (error) {
            console.error('バージョン削除エラー:', error);
            showNotification('バージョンの削除に失敗しました', 'error');
        }
    }
}

/**
 * フレーム抽出進捗を更新
 */
function updateExtractionProgress(data) {
    if (data.status === 'completed') {
        showNotification('フレーム抽出が完了しました', 'success');
        loadVideoList();
        loadDatasetStatistics();
    } else if (data.status === 'failed') {
        showNotification(`フレーム抽出に失敗: ${data.error}`, 'error');
    } else {
        showNotification(data.message, 'info', 3000);
    }
}

/**
 * エクスポート進捗を更新
 */
function updateExportProgress(data) {
    if (data.status === 'completed') {
        showNotification('エクスポートが完了しました', 'success');
        if (data.download_url) {
            // ダウンロードリンクを提供
            setTimeout(() => {
                if (confirm('エクスポートが完了しました。ダウンロードしますか？')) {
                    window.open(data.download_url, '_blank');
                }
            }, 1000);
        }
    } else if (data.status === 'failed') {
        showNotification(`エクスポートに失敗: ${data.error}`, 'error');
    } else {
        showNotification(data.message, 'info', 3000);
    }
}

/**
 * バージョン作成進捗を更新
 */
function updateVersionCreationProgress(data) {
    if (data.status === 'completed') {
        showNotification('データセットバージョンの作成が完了しました', 'success');
        loadDatasetVersions();
    } else if (data.status === 'failed') {
        showNotification(`バージョン作成に失敗: ${data.error}`, 'error');
    } else {
        showNotification(data.message, 'info', 3000);
    }
}

console.log('データ管理機能が読み込まれました');