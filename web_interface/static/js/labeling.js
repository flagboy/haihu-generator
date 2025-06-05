/**
 * 麻雀牌検出システム - ラベリング機能
 */

// グローバル変数
let canvas = null;
let ctx = null;
let currentImage = null;
let currentFrame = null;
let currentVideo = null;
let annotations = [];
let selectedTileType = null;
let isDrawing = false;
let startX = 0;
let startY = 0;
let currentBbox = null;
let zoomLevel = 1;
let panX = 0;
let panY = 0;
let frameList = [];
let currentFrameIndex = -1;
let sessionId = null;
let currentPlayer = 'bottom';  // 現在選択中のプレイヤー
let handAreas = {  // 手牌領域設定
    bottom: null,
    top: null,
    left: null,
    right: null
};
let isSettingHandArea = false;  // 手牌領域設定モード

// 牌種類マッピング
const TILE_NAMES = {
    '1m': '一萬', '2m': '二萬', '3m': '三萬', '4m': '四萬', '5m': '五萬',
    '6m': '六萬', '7m': '七萬', '8m': '八萬', '9m': '九萬',
    '1p': '一筒', '2p': '二筒', '3p': '三筒', '4p': '四筒', '5p': '五筒',
    '6p': '六筒', '7p': '七筒', '8p': '八筒', '9p': '九筒',
    '1s': '一索', '2s': '二索', '3s': '三索', '4s': '四索', '5s': '五索',
    '6s': '六索', '7s': '七索', '8s': '八索', '9s': '九索',
    '1z': '東', '2z': '南', '3z': '西', '4z': '北',
    '5z': '白', '6z': '發', '7z': '中'
};

// 初期化
document.addEventListener('DOMContentLoaded', function() {
    initializeLabeling();
});

/**
 * ラベリング機能の初期化
 */
function initializeLabeling() {
    // キャンバス初期化
    canvas = document.getElementById('labeling-canvas');
    ctx = canvas.getContext('2d');

    // セッションIDを生成
    sessionId = `labeling_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    // イベントリスナー設定
    setupEventListeners();

    // 動画一覧を読み込み
    loadVideoList();

    // WebSocketイベント設定
    setupWebSocketEvents();

    // 手牌領域設定UI追加
    addHandAreaControls();

    console.log('ラベリング機能が初期化されました');
}

/**
 * イベントリスナー設定
 */
function setupEventListeners() {
    // 動画選択
    document.getElementById('video-selector').addEventListener('change', onVideoSelect);

    // 牌種類選択
    document.querySelectorAll('.tile-button').forEach(button => {
        button.addEventListener('click', onTileSelect);
    });

    // キャンバスイベント
    canvas.addEventListener('mousedown', onCanvasMouseDown);
    canvas.addEventListener('mousemove', onCanvasMouseMove);
    canvas.addEventListener('mouseup', onCanvasMouseUp);
    canvas.addEventListener('wheel', onCanvasWheel);
    canvas.addEventListener('contextmenu', e => e.preventDefault());

    // ズームボタン
    document.getElementById('zoom-in-btn').addEventListener('click', () => zoomCanvas(1.2));
    document.getElementById('zoom-out-btn').addEventListener('click', () => zoomCanvas(0.8));
    document.getElementById('zoom-reset-btn').addEventListener('click', resetZoom);

    // フレーム移動ボタン
    document.getElementById('prev-frame-btn').addEventListener('click', () => navigateFrame(-1));
    document.getElementById('next-frame-btn').addEventListener('click', () => navigateFrame(1));

    // 操作ボタン
    document.getElementById('auto-label-btn').addEventListener('click', performAutoLabeling);
    document.getElementById('clear-labels-btn').addEventListener('click', clearAllLabels);
    document.getElementById('save-progress-btn').addEventListener('click', saveProgress);

    // フレーム抽出
    document.getElementById('extract-frames-btn').addEventListener('click', showExtractFramesModal);
    document.getElementById('start-extraction-btn').addEventListener('click', startFrameExtraction);

    // キーボードショートカット
    document.addEventListener('keydown', onKeyDown);
}

/**
 * WebSocketイベント設定
 */
function setupWebSocketEvents() {
    socket.on('frame_extraction_progress', function(data) {
        if (data.status === 'completed') {
            showNotification('フレーム抽出が完了しました', 'success');
            loadFrameList(currentVideo.id);
        } else if (data.status === 'failed') {
            showNotification(`フレーム抽出に失敗: ${data.error}`, 'error');
        } else {
            showNotification(data.message, 'info');
        }
    });
}

/**
 * 動画一覧を読み込み
 */
async function loadVideoList() {
    try {
        const response = await apiRequest('/api/videos');
        const videos = response.videos || [];

        const selector = document.getElementById('video-selector');
        selector.innerHTML = '<option value="">動画を選択...</option>';

        videos.forEach(video => {
            const option = document.createElement('option');
            option.value = video.id;
            option.textContent = video.name;
            selector.appendChild(option);
        });

    } catch (error) {
        console.error('動画一覧の読み込みエラー:', error);
        showNotification('動画一覧の読み込みに失敗しました', 'error');
    }
}

/**
 * 動画選択時の処理
 */
async function onVideoSelect(event) {
    const videoId = event.target.value;

    if (!videoId) {
        currentVideo = null;
        clearVideoInfo();
        clearFrameList();
        return;
    }

    try {
        showLoading('動画情報を読み込み中...');

        // 動画情報を取得
        const response = await apiRequest(`/api/videos/${videoId}`);
        currentVideo = response.video;

        // 動画情報を表示
        displayVideoInfo(currentVideo);

        // フレーム一覧を読み込み
        await loadFrameList(videoId);

        hideLoading();

    } catch (error) {
        console.error('動画選択エラー:', error);
        showNotification('動画情報の読み込みに失敗しました', 'error');
        hideLoading();
    }
}

/**
 * 動画情報を表示
 */
function displayVideoInfo(video) {
    document.getElementById('video-resolution').textContent = `${video.width}x${video.height}`;
    document.getElementById('video-fps').textContent = `${video.fps.toFixed(1)} fps`;
    document.getElementById('video-duration').textContent = formatDuration(video.duration);
    document.getElementById('video-info').style.display = 'block';
    document.getElementById('extract-frames-btn').disabled = false;
}

/**
 * 動画情報をクリア
 */
function clearVideoInfo() {
    document.getElementById('video-info').style.display = 'none';
    document.getElementById('extract-frames-btn').disabled = true;
}

/**
 * フレーム一覧を読み込み
 */
async function loadFrameList(videoId) {
    try {
        const response = await apiRequest(`/api/videos/${videoId}/frames`);
        frameList = response.frames || [];

        displayFrameList(frameList);
        updateProgress();

        if (frameList.length > 0) {
            loadFrame(0);
        }

    } catch (error) {
        console.error('フレーム一覧の読み込みエラー:', error);
        showNotification('フレーム一覧の読み込みに失敗しました', 'error');
    }
}

/**
 * フレーム一覧を表示
 */
function displayFrameList(frames) {
    const container = document.getElementById('frame-list');

    if (frames.length === 0) {
        container.innerHTML = '<div class="text-center text-muted p-3">フレームがありません</div>';
        return;
    }

    container.innerHTML = '';

    frames.forEach((frame, index) => {
        const item = document.createElement('div');
        item.className = 'list-group-item list-group-item-action d-flex align-items-center';
        item.style.cursor = 'pointer';

        const thumbnail = document.createElement('img');
        thumbnail.src = frame.thumbnail_path || frame.image_path;
        thumbnail.className = 'me-3';
        thumbnail.style.width = '60px';
        thumbnail.style.height = '40px';
        thumbnail.style.objectFit = 'cover';
        thumbnail.style.borderRadius = '4px';

        const info = document.createElement('div');
        info.className = 'flex-grow-1';
        info.innerHTML = `
            <div class="fw-bold">フレーム ${index + 1}</div>
            <small class="text-muted">
                ${formatDuration(frame.timestamp)}
                ${frame.tiles ? `(${frame.tiles.length}牌)` : ''}
            </small>
        `;

        const status = document.createElement('div');
        if (frame.tiles && frame.tiles.length > 0) {
            status.innerHTML = '<i class="fas fa-check-circle text-success"></i>';
        } else {
            status.innerHTML = '<i class="fas fa-circle text-muted"></i>';
        }

        item.appendChild(thumbnail);
        item.appendChild(info);
        item.appendChild(status);

        item.addEventListener('click', () => loadFrame(index));

        container.appendChild(item);
    });
}

/**
 * フレーム一覧をクリア
 */
function clearFrameList() {
    document.getElementById('frame-list').innerHTML =
        '<div class="text-center text-muted p-3">動画を選択してください</div>';
    frameList = [];
    currentFrameIndex = -1;
}

/**
 * フレームを読み込み
 */
async function loadFrame(index) {
    if (index < 0 || index >= frameList.length) return;

    try {
        currentFrameIndex = index;
        currentFrame = frameList[index];

        // フレーム画像を読み込み
        const img = new Image();
        img.onload = function() {
            currentImage = img;
            resizeCanvas();
            redrawCanvas();
            updateFrameInfo();
            updateNavigationButtons();
        };
        img.src = currentFrame.image_path;

        // アノテーションを読み込み
        await loadAnnotations(currentFrame.id);

        // フレーム一覧のハイライトを更新
        updateFrameListHighlight(index);

    } catch (error) {
        console.error('フレーム読み込みエラー:', error);
        showNotification('フレームの読み込みに失敗しました', 'error');
    }
}

/**
 * アノテーションを読み込み
 */
async function loadAnnotations(frameId) {
    try {
        const response = await apiRequest(`/api/frames/${frameId}/annotations`);
        annotations = response.annotations || [];
        updateAnnotationList();

    } catch (error) {
        console.error('アノテーション読み込みエラー:', error);
        annotations = [];
    }
}

/**
 * キャンバスをリサイズ
 */
function resizeCanvas() {
    if (!currentImage) return;

    const container = document.getElementById('canvas-container');
    const containerWidth = container.clientWidth;
    const containerHeight = Math.min(600, container.clientHeight);

    const imageAspect = currentImage.width / currentImage.height;
    const containerAspect = containerWidth / containerHeight;

    if (imageAspect > containerAspect) {
        canvas.width = containerWidth;
        canvas.height = containerWidth / imageAspect;
    } else {
        canvas.width = containerHeight * imageAspect;
        canvas.height = containerHeight;
    }

    resetZoom();
}

/**
 * キャンバスを再描画
 */
function redrawCanvas() {
    if (!currentImage) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 画像を描画
    ctx.save();
    ctx.scale(zoomLevel, zoomLevel);
    ctx.translate(panX, panY);
    ctx.drawImage(currentImage, 0, 0, canvas.width / zoomLevel, canvas.height / zoomLevel);
    ctx.restore();

    // アノテーションを描画
    drawAnnotations();

    // 現在描画中のバウンディングボックスを描画
    if (currentBbox) {
        drawBoundingBox(currentBbox, '#ff0000', 2);
    }
}

/**
 * アノテーションを描画
 */
function drawAnnotations() {
    annotations.forEach((annotation, index) => {
        const color = annotation.selected ? '#dc3545' : '#007bff';
        const lineWidth = annotation.selected ? 3 : 2;
        drawBoundingBox(annotation.bbox, color, lineWidth);

        // ラベルを描画
        drawLabel(annotation.bbox, annotation.tile_id, color);
    });
}

/**
 * バウンディングボックスを描画
 */
function drawBoundingBox(bbox, color, lineWidth) {
    ctx.save();
    ctx.scale(zoomLevel, zoomLevel);
    ctx.translate(panX, panY);

    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth / zoomLevel;
    ctx.strokeRect(bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1);

    ctx.restore();
}

/**
 * ラベルを描画
 */
function drawLabel(bbox, tileId, color) {
    const tileName = TILE_NAMES[tileId] || tileId;

    ctx.save();
    ctx.scale(zoomLevel, zoomLevel);
    ctx.translate(panX, panY);

    const fontSize = 14 / zoomLevel;
    ctx.font = `${fontSize}px Arial`;
    ctx.fillStyle = color;
    ctx.fillRect(bbox.x1, bbox.y1 - fontSize - 4, ctx.measureText(tileName).width + 8, fontSize + 4);

    ctx.fillStyle = '#ffffff';
    ctx.fillText(tileName, bbox.x1 + 4, bbox.y1 - 4);

    ctx.restore();
}

/**
 * 牌種類選択時の処理
 */
function onTileSelect(event) {
    // 前の選択を解除
    document.querySelectorAll('.tile-button').forEach(btn => {
        btn.classList.remove('active');
    });

    // 新しい選択を設定
    event.target.classList.add('active');
    selectedTileType = event.target.dataset.tile;

    // 選択情報を表示
    const selectedInfo = document.getElementById('selected-tile-info');
    const selectedName = document.getElementById('selected-tile-name');
    selectedName.textContent = TILE_NAMES[selectedTileType] || selectedTileType;
    selectedInfo.style.display = 'block';
}

/**
 * キャンバスマウスダウン
 */
function onCanvasMouseDown(event) {
    if (!currentImage || !selectedTileType) return;

    const rect = canvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) / zoomLevel - panX;
    const y = (event.clientY - rect.top) / zoomLevel - panY;

    if (event.button === 0) { // 左クリック
        isDrawing = true;
        startX = x;
        startY = y;
        currentBbox = { x1: x, y1: y, x2: x, y2: y };
    } else if (event.button === 2) { // 右クリック
        // アノテーション選択/削除
        selectOrDeleteAnnotation(x, y);
    }
}

/**
 * キャンバスマウス移動
 */
function onCanvasMouseMove(event) {
    const rect = canvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) / zoomLevel - panX;
    const y = (event.clientY - rect.top) / zoomLevel - panY;

    // 座標表示を更新
    document.getElementById('mouse-coordinates').textContent = `${Math.round(x)}, ${Math.round(y)}`;

    if (isDrawing && currentBbox) {
        currentBbox.x2 = x;
        currentBbox.y2 = y;
        redrawCanvas();
    }
}

/**
 * キャンバスマウスアップ
 */
function onCanvasMouseUp(event) {
    if (!isDrawing || !currentBbox) return;

    isDrawing = false;

    // バウンディングボックスのサイズをチェック
    const width = Math.abs(currentBbox.x2 - currentBbox.x1);
    const height = Math.abs(currentBbox.y2 - currentBbox.y1);

    if (width > 10 && height > 10) {
        // 正規化
        const bbox = {
            x1: Math.min(currentBbox.x1, currentBbox.x2),
            y1: Math.min(currentBbox.y1, currentBbox.y2),
            x2: Math.max(currentBbox.x1, currentBbox.x2),
            y2: Math.max(currentBbox.y1, currentBbox.y2)
        };

        // アノテーションを追加
        addAnnotation(bbox, selectedTileType);
    }

    currentBbox = null;
    redrawCanvas();
}

/**
 * キャンバスホイール（ズーム）
 */
function onCanvasWheel(event) {
    event.preventDefault();

    const zoomFactor = event.deltaY > 0 ? 0.9 : 1.1;
    zoomCanvas(zoomFactor);
}

/**
 * キャンバスズーム
 */
function zoomCanvas(factor) {
    zoomLevel = Math.max(0.1, Math.min(5, zoomLevel * factor));
    redrawCanvas();
}

/**
 * ズームリセット
 */
function resetZoom() {
    zoomLevel = 1;
    panX = 0;
    panY = 0;
    redrawCanvas();
}

/**
 * アノテーションを追加
 */
function addAnnotation(bbox, tileId) {
    const annotation = {
        id: Date.now().toString(),
        tile_id: tileId,
        bbox: bbox,
        confidence: 1.0,
        area_type: 'hand',
        is_face_up: true,
        is_occluded: false,
        occlusion_ratio: 0.0,
        annotator: 'user',
        notes: ''
    };

    annotations.push(annotation);
    updateAnnotationList();
    updateProgress();
    redrawCanvas();
}

/**
 * アノテーション選択/削除
 */
function selectOrDeleteAnnotation(x, y) {
    // クリック位置にあるアノテーションを検索
    for (let i = annotations.length - 1; i >= 0; i--) {
        const annotation = annotations[i];
        const bbox = annotation.bbox;

        if (x >= bbox.x1 && x <= bbox.x2 && y >= bbox.y1 && y <= bbox.y2) {
            if (annotation.selected) {
                // 既に選択されている場合は削除
                annotations.splice(i, 1);
                updateAnnotationList();
                updateProgress();
            } else {
                // 選択状態を切り替え
                annotations.forEach(ann => ann.selected = false);
                annotation.selected = true;
            }
            redrawCanvas();
            return;
        }
    }

    // どのアノテーションもクリックされていない場合は選択解除
    annotations.forEach(ann => ann.selected = false);
    redrawCanvas();
}

/**
 * アノテーション一覧を更新
 */
function updateAnnotationList() {
    const container = document.getElementById('annotation-list');

    if (annotations.length === 0) {
        container.innerHTML = '<div class="text-center text-muted p-3">アノテーションはありません</div>';
        return;
    }

    container.innerHTML = '';

    annotations.forEach((annotation, index) => {
        const item = document.createElement('div');
        item.className = 'list-group-item d-flex justify-content-between align-items-center';

        const info = document.createElement('div');
        info.innerHTML = `
            <div class="fw-bold">${TILE_NAMES[annotation.tile_id] || annotation.tile_id}</div>
            <small class="text-muted">
                (${Math.round(annotation.bbox.x1)}, ${Math.round(annotation.bbox.y1)}) -
                (${Math.round(annotation.bbox.x2)}, ${Math.round(annotation.bbox.y2)})
            </small>
        `;

        const actions = document.createElement('div');
        actions.innerHTML = `
            <button class="btn btn-sm btn-outline-danger" onclick="deleteAnnotation(${index})">
                <i class="fas fa-trash"></i>
            </button>
        `;

        item.appendChild(info);
        item.appendChild(actions);

        container.appendChild(item);
    });
}

/**
 * アノテーションを削除
 */
function deleteAnnotation(index) {
    if (index >= 0 && index < annotations.length) {
        annotations.splice(index, 1);
        updateAnnotationList();
        updateProgress();
        redrawCanvas();
    }
}

/**
 * すべてのラベルをクリア
 */
function clearAllLabels() {
    if (annotations.length === 0) return;

    if (confirm('現在のフレームのすべてのアノテーションを削除しますか？')) {
        annotations = [];
        updateAnnotationList();
        updateProgress();
        redrawCanvas();
    }
}

/**
 * 自動ラベリング実行
 */
async function performAutoLabeling() {
    if (!currentFrame) {
        showNotification('フレームが選択されていません', 'warning');
        return;
    }

    try {
        showLoading('自動ラベリング中...');

        const response = await apiRequest('/api/auto_label', {
            method: 'POST',
            body: JSON.stringify({
                frame_id: currentFrame.id,
                image_path: currentFrame.image_path
            })
        });

        if (response.annotations) {
            annotations = response.annotations;
            updateAnnotationList();
            updateProgress();
            redrawCanvas();
            showNotification('自動ラベリングが完了しました', 'success');
        }

        hideLoading();

    } catch (error) {
        console.error('自動ラベリングエラー:', error);
        showNotification('自動ラベリングに失敗しました', 'error');
        hideLoading();
    }
}

/**
 * フレーム移動
 */
function navigateFrame(direction) {
    const newIndex = currentFrameIndex + direction;
    if (newIndex >= 0 && newIndex < frameList.length) {
        loadFrame(newIndex);
    }
}

/**
 * フレーム情報を更新
 */
function updateFrameInfo() {
    if (currentFrame) {
        document.getElementById('current-frame-info').textContent =
            `${currentFrameIndex + 1} / ${frameList.length} (${formatDuration(currentFrame.timestamp)})`;
    }
}

/**
 * ナビゲーションボタンを更新
 */
function updateNavigationButtons() {
    document.getElementById('prev-frame-btn').disabled = currentFrameIndex <= 0;
    document.getElementById('next-frame-btn').disabled = currentFrameIndex >= frameList.length - 1;
}

/**
 * フレーム一覧のハイライトを更新
 */
function updateFrameListHighlight(index) {
    document.querySelectorAll('#frame-list .list-group-item').forEach((item, i) => {
        if (i === index) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
    });
}

/**
 * 進捗を更新
 */
function updateProgress() {
    const labeledCount = frameList.filter(frame => frame.tiles && frame.tiles.length > 0).length;
    const totalCount = frameList.length;
    const progress = totalCount > 0 ? (labeledCount / totalCount) * 100 : 0;

    document.getElementById('labeling-progress').style.width = `${progress}%`;
    document.getElementById('labeled-count').textContent = labeledCount;
    document.getElementById('total-count').textContent = totalCount;

    const totalTiles = frameList.reduce((sum, frame) => sum + (frame.tiles ? frame.tiles.length : 0), 0);
    document.getElementById('total-tiles').textContent = totalTiles;
}

/**
 * 進捗を保存
 */
async function saveProgress() {
    if (!currentFrame || annotations.length === 0) {
        showNotification('保存するアノテーションがありません', 'warning');
        return;
    }

    try {
        showLoading('保存中...');

        await apiRequest(`/api/frames/${currentFrame.id}/annotations`, {
            method: 'POST',
            body: JSON.stringify({
                annotations: annotations
            })
        });

        // フレーム情報を更新
        frameList[currentFrameIndex].tiles = annotations;
        updateProgress();

        showNotification('アノテーションを保存しました', 'success');
        hideLoading();

    } catch (error) {
        console.error('保存エラー:', error);
        showNotification('保存に失敗しました', 'error');
        hideLoading();
    }
}

/**
 * フレーム抽出モーダルを表示
 */
function showExtractFramesModal() {
    if (!currentVideo) {
        showNotification('動画が選択されていません', 'warning');
        return;
    }

    const modal = new bootstrap.Modal(document.getElementById('extractFramesModal'));
    modal.show();
}

/**
 * フレーム抽出を開始
 */
async function startFrameExtraction() {
    const config = {
        interval_seconds: parseFloat(document.getElementById('interval-seconds').value),
        quality_threshold: parseFloat(document.getElementById('quality-threshold').value),
        max_frames: parseInt(document.getElementById('max-frames').value)
    };

    try {
        const response = await apiRequest('/api/extract_frames', {
            method: 'POST',
            body: JSON.stringify({
                video_path: currentVideo.path,
                config: config
            })
        });

        if (response.session_id) {
            socket.emit('join_session', { session_id: response.session_id });
            showNotification('フレーム抽出を開始しました', 'info');

            // モーダルを閉じる
            const modal = bootstrap.Modal.getInstance(document.getElementById('extractFramesModal'));
            modal.hide();
        }

    } catch (error) {
        console.error('フレーム抽出エラー:', error);
        showNotification('フレーム抽出の開始に失敗しました', 'error');
    }
}

/**
 * キーボードショートカット
 */
function onKeyDown(event) {
    if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') {
        return;
    }

    switch (event.key) {
        case 'ArrowLeft':
            event.preventDefault();
            navigateFrame(-1);
            break;
        case 'ArrowRight':
            event.preventDefault();
            navigateFrame(1);
            break;
        case 'Delete':
        case 'Backspace':
            event.preventDefault();
            const selectedAnnotation = annotations.find(ann => ann.selected);
            if (selectedAnnotation) {
                const index = annotations.indexOf(selectedAnnotation);
                deleteAnnotation(index);
            }
            break;
        case 'Escape':
            event.preventDefault();
            annotations.forEach(ann => ann.selected = false);
            redrawCanvas();
            break;
        case 's':
            if (event.ctrlKey) {
                event.preventDefault();
                saveProgress();
            }
            break;
    }
}

/**
 * 手牌領域設定UIを追加
 */
function addHandAreaControls() {
    // カード本体の後に手牌領域設定UIを追加
    const canvasCard = document.querySelector('#canvas-container').closest('.card');
    const handAreaCard = document.createElement('div');
    handAreaCard.className = 'card mt-3';
    handAreaCard.innerHTML = `
        <div class="card-header">
            <h6 class="card-title mb-0">
                <i class="fas fa-hand-paper me-2"></i>手牌領域設定
            </h6>
        </div>
        <div class="card-body">
            <div class="btn-group w-100 mb-2" role="group">
                <button type="button" class="btn btn-outline-primary player-select-btn" data-player="bottom">
                    <i class="fas fa-user"></i> 自分
                </button>
                <button type="button" class="btn btn-outline-primary player-select-btn" data-player="top">
                    <i class="fas fa-user"></i> 対面
                </button>
                <button type="button" class="btn btn-outline-primary player-select-btn" data-player="left">
                    <i class="fas fa-user"></i> 左
                </button>
                <button type="button" class="btn btn-outline-primary player-select-btn" data-player="right">
                    <i class="fas fa-user"></i> 右
                </button>
            </div>
            <button id="set-hand-area-btn" class="btn btn-warning w-100 mb-2">
                <i class="fas fa-crop"></i> 領域を設定
            </button>
            <button id="split-tiles-btn" class="btn btn-info w-100" disabled>
                <i class="fas fa-cut"></i> 牌を分割
            </button>
        </div>
    `;

    canvasCard.parentNode.insertBefore(handAreaCard, canvasCard.nextSibling);

    // プレイヤー選択ボタンのイベント
    document.querySelectorAll('.player-select-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.player-select-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            currentPlayer = e.target.dataset.player;
            updateHandAreaDisplay();
        });
    });

    // デフォルトで「自分」を選択
    document.querySelector('.player-select-btn[data-player="bottom"]').classList.add('active');

    // 手牌領域設定ボタン
    document.getElementById('set-hand-area-btn').addEventListener('click', toggleHandAreaSetting);

    // 牌分割ボタン
    document.getElementById('split-tiles-btn').addEventListener('click', splitTilesForCurrentPlayer);
}

/**
 * 手牌領域設定モードの切り替え
 */
function toggleHandAreaSetting() {
    isSettingHandArea = !isSettingHandArea;
    const btn = document.getElementById('set-hand-area-btn');

    if (isSettingHandArea) {
        btn.classList.remove('btn-warning');
        btn.classList.add('btn-success');
        btn.innerHTML = '<i class="fas fa-check"></i> 設定完了';
        showNotification('手牌領域をドラッグで選択してください', 'info');
    } else {
        btn.classList.remove('btn-success');
        btn.classList.add('btn-warning');
        btn.innerHTML = '<i class="fas fa-crop"></i> 領域を設定';

        // 設定を保存
        saveHandAreas();
    }
}

/**
 * 手牌領域を保存
 */
async function saveHandAreas() {
    try {
        await apiRequest('/api/labeling/hand_areas', {
            method: 'POST',
            body: JSON.stringify({
                frame_size: currentImage ? [currentImage.width, currentImage.height] : null,
                regions: handAreas
            })
        });

        showNotification('手牌領域を保存しました', 'success');
        document.getElementById('split-tiles-btn').disabled = false;
    } catch (error) {
        console.error('手牌領域保存エラー:', error);
        showNotification('手牌領域の保存に失敗しました', 'error');
    }
}

/**
 * 現在のプレイヤーの牌を分割
 */
async function splitTilesForCurrentPlayer() {
    if (!currentVideo || !currentFrame || !handAreas[currentPlayer]) {
        showNotification('手牌領域が設定されていません', 'warning');
        return;
    }

    try {
        showLoading('牌を分割中...');

        const response = await apiRequest('/api/labeling/split_tiles', {
            method: 'POST',
            body: JSON.stringify({
                video_id: currentVideo.id,
                frame_number: currentFrameIndex,
                player: currentPlayer
            })
        });

        if (response.tiles) {
            // 既存のアノテーションをクリア
            annotations = [];

            // 分割結果からアノテーションを作成
            response.tiles.forEach(tile => {
                const handArea = handAreas[currentPlayer];
                const annotation = {
                    id: Date.now().toString() + '_' + tile.index,
                    tile_id: tile.label || selectedTileType || '1m',
                    bbox: {
                        x1: handArea.x * canvas.width + tile.bbox.x,
                        y1: handArea.y * canvas.height + tile.bbox.y,
                        x2: handArea.x * canvas.width + tile.bbox.x + tile.bbox.w,
                        y2: handArea.y * canvas.height + tile.bbox.y + tile.bbox.h
                    },
                    confidence: tile.confidence || 0,
                    area_type: 'hand',
                    player: currentPlayer,
                    is_face_up: true,
                    is_occluded: false,
                    occlusion_ratio: 0.0,
                    annotator: 'auto_split',
                    notes: ''
                };
                annotations.push(annotation);
            });

            updateAnnotationList();
            redrawCanvas();
            showNotification(`${response.tiles.length}個の牌を検出しました`, 'success');
        }

        hideLoading();
    } catch (error) {
        console.error('牌分割エラー:', error);
        showNotification('牌の分割に失敗しました', 'error');
        hideLoading();
    }
}

/**
 * 手牌領域の表示を更新
 */
function updateHandAreaDisplay() {
    redrawCanvas();
}

/**
 * 動画選択時の処理（拡張）
 */
const originalOnVideoSelect = onVideoSelect;
async function onVideoSelect(event) {
    await originalOnVideoSelect.call(this, event);

    const videoId = event.target.value;
    if (videoId && currentVideo) {
        // 動画をラベリング用に読み込み
        try {
            const response = await apiRequest('/api/labeling/load_video', {
                method: 'POST',
                body: JSON.stringify({
                    video_path: currentVideo.path,
                    video_id: videoId
                })
            });

            if (response.video_id) {
                currentVideo.id = response.video_id;
                console.log('動画をラベリング用に読み込みました');
            }
        } catch (error) {
            console.error('動画読み込みエラー:', error);
        }
    }
}

/**
 * キャンバス再描画（拡張）
 */
const originalRedrawCanvas = redrawCanvas;
function redrawCanvas() {
    originalRedrawCanvas();

    // 手牌領域を描画
    if (currentImage) {
        ctx.save();
        ctx.scale(zoomLevel, zoomLevel);
        ctx.translate(panX, panY);

        Object.entries(handAreas).forEach(([player, area]) => {
            if (area) {
                const color = player === currentPlayer ? '#00ff00' : '#ffff00';
                const alpha = player === currentPlayer ? 0.3 : 0.1;

                // 領域を塗りつぶし
                ctx.fillStyle = color + Math.round(alpha * 255).toString(16).padStart(2, '0');
                ctx.fillRect(
                    area.x * canvas.width / zoomLevel,
                    area.y * canvas.height / zoomLevel,
                    area.w * canvas.width / zoomLevel,
                    area.h * canvas.height / zoomLevel
                );

                // 枠線
                ctx.strokeStyle = color;
                ctx.lineWidth = 2 / zoomLevel;
                ctx.strokeRect(
                    area.x * canvas.width / zoomLevel,
                    area.y * canvas.height / zoomLevel,
                    area.w * canvas.width / zoomLevel,
                    area.h * canvas.height / zoomLevel
                );

                // ラベル
                ctx.fillStyle = color;
                ctx.font = `${16 / zoomLevel}px Arial`;
                const playerName = {
                    bottom: '自分',
                    top: '対面',
                    left: '左',
                    right: '右'
                }[player];
                ctx.fillText(
                    playerName,
                    area.x * canvas.width / zoomLevel + 5,
                    area.y * canvas.height / zoomLevel + 20 / zoomLevel
                );
            }
        });

        ctx.restore();
    }
}

/**
 * マウスイベント処理（拡張）
 */
const originalOnCanvasMouseDown = onCanvasMouseDown;
function onCanvasMouseDown(event) {
    if (isSettingHandArea) {
        const rect = canvas.getBoundingClientRect();
        const x = (event.clientX - rect.left) / canvas.width;
        const y = (event.clientY - rect.top) / canvas.height;

        isDrawing = true;
        startX = x;
        startY = y;
        currentBbox = { x1: x, y1: y, x2: x, y2: y };
    } else {
        originalOnCanvasMouseDown.call(this, event);
    }
}

const originalOnCanvasMouseMove = onCanvasMouseMove;
function onCanvasMouseMove(event) {
    if (isSettingHandArea && isDrawing) {
        const rect = canvas.getBoundingClientRect();
        const x = (event.clientX - rect.left) / canvas.width;
        const y = (event.clientY - rect.top) / canvas.height;

        currentBbox.x2 = x;
        currentBbox.y2 = y;
        redrawCanvas();

        // 一時的な領域を描画
        ctx.strokeStyle = '#ff0000';
        ctx.lineWidth = 2;
        ctx.strokeRect(
            Math.min(currentBbox.x1, currentBbox.x2) * canvas.width,
            Math.min(currentBbox.y1, currentBbox.y2) * canvas.height,
            Math.abs(currentBbox.x2 - currentBbox.x1) * canvas.width,
            Math.abs(currentBbox.y2 - currentBbox.y1) * canvas.height
        );
    } else {
        originalOnCanvasMouseMove.call(this, event);
    }
}

const originalOnCanvasMouseUp = onCanvasMouseUp;
function onCanvasMouseUp(event) {
    if (isSettingHandArea && isDrawing) {
        isDrawing = false;

        if (currentBbox) {
            // 正規化
            const area = {
                x: Math.min(currentBbox.x1, currentBbox.x2),
                y: Math.min(currentBbox.y1, currentBbox.y2),
                w: Math.abs(currentBbox.x2 - currentBbox.x1),
                h: Math.abs(currentBbox.y2 - currentBbox.y1)
            };

            if (area.w > 0.01 && area.h > 0.01) {
                handAreas[currentPlayer] = area;
                showNotification(`${currentPlayer}の手牌領域を設定しました`, 'success');
            }

            currentBbox = null;
            redrawCanvas();
        }
    } else {
        originalOnCanvasMouseUp.call(this, event);
    }
}

/**
 * 自動ラベリング（拡張）
 */
async function performAutoLabeling() {
    if (!currentFrame || !currentVideo) {
        showNotification('フレームが選択されていません', 'warning');
        return;
    }

    try {
        showLoading('自動ラベリング中...');

        const response = await apiRequest('/api/labeling/auto_label', {
            method: 'POST',
            body: JSON.stringify({
                video_id: currentVideo.id,
                frame_number: currentFrameIndex,
                player: currentPlayer
            })
        });

        if (response.tiles) {
            // 自動ラベリング結果を既存のアノテーションに反映
            response.tiles.forEach((tile, index) => {
                if (annotations[index]) {
                    annotations[index].tile_id = tile.label;
                    annotations[index].confidence = tile.confidence;
                    annotations[index].annotator = 'auto_label';
                }
            });

            updateAnnotationList();
            redrawCanvas();
            showNotification('自動ラベリングが完了しました', 'success');
        }

        hideLoading();

    } catch (error) {
        console.error('自動ラベリングエラー:', error);
        showNotification('自動ラベリングに失敗しました', 'error');
        hideLoading();
    }
}

/**
 * 進捗を保存（拡張）
 */
async function saveProgress() {
    if (!currentFrame || annotations.length === 0) {
        showNotification('保存するアノテーションがありません', 'warning');
        return;
    }

    try {
        showLoading('保存中...');

        // 新しいAPIエンドポイントを使用
        await apiRequest('/api/labeling/save_annotations', {
            method: 'POST',
            body: JSON.stringify({
                session_id: sessionId,
                annotations: [{
                    video_id: currentVideo.id,
                    frame_number: currentFrameIndex,
                    player: currentPlayer,
                    tiles: annotations.map(ann => ({
                        tile_id: ann.tile_id,
                        bbox: ann.bbox,
                        confidence: ann.confidence,
                        area_type: ann.area_type,
                        is_face_up: ann.is_face_up,
                        is_occluded: ann.is_occluded,
                        occlusion_ratio: ann.occlusion_ratio
                    }))
                }]
            })
        });

        // フレーム情報を更新
        frameList[currentFrameIndex].tiles = annotations;
        updateProgress();

        showNotification('アノテーションを保存しました', 'success');
        hideLoading();

    } catch (error) {
        console.error('保存エラー:', error);
        showNotification('保存に失敗しました', 'error');
        hideLoading();
    }
}

/**
 * キーボードショートカット（拡張）
 */
const originalOnKeyDown = onKeyDown;
function onKeyDown(event) {
    // 数字キー + Q/W/E で牌を選択
    if (event.key >= '1' && event.key <= '9') {
        event.preventDefault();
        let tileType = '';

        if (event.shiftKey || event.key === 'Q' || event.key === 'q') {
            tileType = event.key + 'm';  // 萬子
        } else if (event.ctrlKey || event.key === 'W' || event.key === 'w') {
            tileType = event.key + 'p';  // 筒子
        } else if (event.altKey || event.key === 'E' || event.key === 'e') {
            tileType = event.key + 's';  // 索子
        } else {
            // デフォルトは萬子
            tileType = event.key + 'm';
        }

        // 対応するボタンをクリック
        const btn = document.querySelector(`.tile-button[data-tile="${tileType}"]`);
        if (btn) btn.click();
    }

    // 字牌のショートカット
    const honorKeys = {
        'a': '1z', // 東
        's': '2z', // 南
        'd': '3z', // 西
        'f': '4z', // 北
        'g': '5z', // 白
        'h': '6z', // 發
        'j': '7z'  // 中
    };

    if (honorKeys[event.key.toLowerCase()]) {
        event.preventDefault();
        const btn = document.querySelector(`.tile-button[data-tile="${honorKeys[event.key.toLowerCase()]}"]`);
        if (btn) btn.click();
    }

    // その他のショートカットは元の処理を実行
    originalOnKeyDown.call(this, event);
}

console.log('ラベリング機能が読み込まれました');
