/**
 * 麻雀牌検出システム - 学習管理機能
 */

// グローバル変数
let trainingSessions = [];
let datasetVersions = [];
let trainingHistoryChart = null;
let comparisonChart = null;
let selectedModels = new Set();

// 初期化
document.addEventListener('DOMContentLoaded', function() {
    initializeTraining();
});

/**
 * 学習管理機能の初期化
 */
function initializeTraining() {
    // イベントリスナー設定
    setupEventListeners();
    
    // データを読み込み
    loadInitialData();
    
    // WebSocketイベント設定
    setupWebSocketEvents();
    
    console.log('学習管理機能が初期化されました');
}

/**
 * イベントリスナー設定
 */
function setupEventListeners() {
    // 学習設定フォーム
    document.getElementById('training-config-form').addEventListener('submit', onTrainingFormSubmit);
    
    // 転移学習チェックボックス
    document.getElementById('transfer-learning').addEventListener('change', onTransferLearningChange);
    
    // データセットバージョン選択
    document.getElementById('dataset-version').addEventListener('change', onDatasetVersionChange);
    
    // ステータスフィルター
    document.getElementById('status-filter').addEventListener('change', filterSessions);
    
    // モーダルイベント
    document.getElementById('sessionDetailModal').addEventListener('show.bs.modal', onSessionDetailModalShow);
    document.getElementById('compareModelsModal').addEventListener('show.bs.modal', onCompareModelsModalShow);
    
    // モーダル内ボタン
    document.getElementById('stop-training-btn').addEventListener('click', stopTraining);
    document.getElementById('continue-training-btn').addEventListener('click', continueTraining);
    document.getElementById('download-model-btn').addEventListener('click', downloadModel);
}

/**
 * WebSocketイベント設定
 */
function setupWebSocketEvents() {
    socket.on('training_progress', function(data) {
        console.log('学習進捗更新:', data);
        updateTrainingProgress(data);
        
        // セッション一覧を更新
        refreshSessions();
        
        // 詳細モーダルが開いている場合は更新
        const modal = document.getElementById('sessionDetailModal');
        if (modal.classList.contains('show')) {
            const sessionId = modal.dataset.sessionId;
            if (sessionId === data.training_session_id) {
                updateSessionDetail(data);
            }
        }
    });
}

/**
 * 初期データを読み込み
 */
async function loadInitialData() {
    try {
        showLoading('データを読み込み中...');
        
        // データセットバージョンを読み込み
        await loadDatasetVersions();
        
        // 学習セッションを読み込み
        await loadTrainingSessions();
        
        hideLoading();
        
    } catch (error) {
        console.error('初期データ読み込みエラー:', error);
        showNotification('データの読み込みに失敗しました', 'error');
        hideLoading();
    }
}

/**
 * データセットバージョンを読み込み
 */
async function loadDatasetVersions() {
    try {
        const response = await apiRequest('/api/dataset/versions');
        datasetVersions = response;
        
        const selector = document.getElementById('dataset-version');
        selector.innerHTML = '<option value="">選択してください...</option>';
        
        datasetVersions.forEach(version => {
            const option = document.createElement('option');
            option.value = version.id;
            option.textContent = `${version.version} (${version.frame_count}フレーム, ${version.tile_count}牌)`;
            selector.appendChild(option);
        });
        
    } catch (error) {
        console.error('データセットバージョン読み込みエラー:', error);
        throw error;
    }
}

/**
 * 学習セッションを読み込み
 */
async function loadTrainingSessions() {
    try {
        const response = await apiRequest('/api/training/sessions');
        trainingSessions = response;
        
        displayTrainingSessions(trainingSessions);
        
    } catch (error) {
        console.error('学習セッション読み込みエラー:', error);
        throw error;
    }
}

/**
 * 学習セッションを表示
 */
function displayTrainingSessions(sessions) {
    const container = document.getElementById('sessions-container');
    
    if (sessions.length === 0) {
        container.innerHTML = `
            <div class="text-center text-muted py-5">
                <i class="fas fa-brain fa-3x mb-3"></i>
                <div>学習セッションはありません</div>
                <div class="mt-3">
                    <small>左側のフォームから新しい学習を開始してください</small>
                </div>
            </div>
        `;
        return;
    }
    
    container.innerHTML = '';
    
    sessions.forEach(session => {
        const card = createSessionCard(session);
        container.appendChild(card);
    });
}

/**
 * セッションカードを作成
 */
function createSessionCard(session) {
    const card = document.createElement('div');
    card.className = 'card session-card mb-3';
    
    const statusBadge = getStatusBadge(session.status);
    const progress = session.current_progress || {};
    const metrics = session.final_metrics || {};
    
    card.innerHTML = `
        <div class="card-header d-flex justify-content-between align-items-center">
            <div>
                <h6 class="card-title mb-0">${session.config.model_name}</h6>
                <small class="text-muted">${session.config.model_type}</small>
            </div>
            <div>
                ${statusBadge}
            </div>
        </div>
        <div class="card-body">
            <div class="row">
                <div class="col-md-6">
                    <div class="mb-2">
                        <small class="text-muted">開始時刻</small>
                        <div>${formatDateTime(session.start_time)}</div>
                    </div>
                    ${session.end_time ? `
                        <div class="mb-2">
                            <small class="text-muted">終了時刻</small>
                            <div>${formatDateTime(session.end_time)}</div>
                        </div>
                    ` : ''}
                </div>
                <div class="col-md-6">
                    ${session.status === 'running' ? `
                        <div class="mb-2">
                            <small class="text-muted">進捗</small>
                            <div class="progress mb-1" style="height: 6px;">
                                <div class="progress-bar progress-bar-striped progress-bar-animated" 
                                     style="width: ${progress.progress || 0}%"></div>
                            </div>
                            <small class="text-muted">
                                エポック ${progress.epoch || 0} / ${session.config.epochs}
                            </small>
                        </div>
                    ` : ''}
                    ${Object.keys(metrics).length > 0 ? `
                        <div class="mb-2">
                            <small class="text-muted">最終メトリクス</small>
                            <div class="small">
                                ${metrics.accuracy ? `精度: ${formatPercentage(metrics.accuracy)}` : ''}
                                ${metrics.loss ? `損失: ${metrics.loss.toFixed(4)}` : ''}
                            </div>
                        </div>
                    ` : ''}
                </div>
            </div>
            
            <div class="mt-3 d-flex justify-content-between">
                <div class="btn-group btn-group-sm">
                    <button class="btn btn-outline-primary" onclick="showSessionDetail('${session.session_id}')">
                        <i class="fas fa-eye"></i> 詳細
                    </button>
                    ${session.status === 'running' ? `
                        <button class="btn btn-outline-warning" onclick="stopTrainingSession('${session.session_id}')">
                            <i class="fas fa-stop"></i> 停止
                        </button>
                    ` : ''}
                    ${session.status === 'completed' && session.best_model_path ? `
                        <button class="btn btn-outline-success" onclick="downloadSessionModel('${session.session_id}')">
                            <i class="fas fa-download"></i> ダウンロード
                        </button>
                    ` : ''}
                </div>
                
                <div class="form-check">
                    <input class="form-check-input" type="checkbox" 
                           id="select-${session.session_id}" 
                           onchange="toggleModelSelection('${session.session_id}')">
                    <label class="form-check-label" for="select-${session.session_id}">
                        比較対象
                    </label>
                </div>
            </div>
        </div>
    `;
    
    return card;
}

/**
 * ステータスバッジを取得
 */
function getStatusBadge(status) {
    const badges = {
        'running': '<span class="badge bg-primary status-badge">実行中</span>',
        'completed': '<span class="badge bg-success status-badge">完了</span>',
        'failed': '<span class="badge bg-danger status-badge">失敗</span>',
        'stopped': '<span class="badge bg-warning status-badge">停止</span>'
    };
    
    return badges[status] || '<span class="badge bg-secondary status-badge">不明</span>';
}

/**
 * 学習フォーム送信
 */
async function onTrainingFormSubmit(event) {
    event.preventDefault();
    
    const formData = getFormData(event.target);
    
    // バリデーション
    if (!formData['dataset-version']) {
        showNotification('データセットバージョンを選択してください', 'warning');
        return;
    }
    
    try {
        showLoading('学習を開始中...');
        
        const config = {
            model_name: formData['model-name'],
            model_type: formData['model-type'],
            dataset_version_id: formData['dataset-version'],
            epochs: formData['epochs'],
            batch_size: formData['batch-size'],
            learning_rate: formData['learning-rate'],
            validation_split: formData['validation-split'],
            test_split: formData['test-split'],
            use_data_augmentation: formData['use-data-augmentation'],
            transfer_learning: formData['transfer-learning'],
            pretrained_model_path: formData['pretrained-model'] || null
        };
        
        const response = await apiRequest('/api/training/start', {
            method: 'POST',
            body: JSON.stringify(config)
        });
        
        if (response.session_id) {
            socket.emit('join_session', { session_id: response.session_id });
            showNotification('学習を開始しました', 'success');
            
            // フォームをリセット
            event.target.reset();
            document.getElementById('pretrained-model-group').style.display = 'none';
            
            // セッション一覧を更新
            setTimeout(() => refreshSessions(), 1000);
        }
        
        hideLoading();
        
    } catch (error) {
        console.error('学習開始エラー:', error);
        showNotification(`学習の開始に失敗しました: ${error.message}`, 'error');
        hideLoading();
    }
}

/**
 * 転移学習チェックボックス変更
 */
function onTransferLearningChange(event) {
    const pretrainedGroup = document.getElementById('pretrained-model-group');
    
    if (event.target.checked) {
        pretrainedGroup.style.display = 'block';
        loadPretrainedModels();
    } else {
        pretrainedGroup.style.display = 'none';
    }
}

/**
 * 事前学習済みモデルを読み込み
 */
async function loadPretrainedModels() {
    try {
        const completedSessions = trainingSessions.filter(s => 
            s.status === 'completed' && s.best_model_path
        );
        
        const selector = document.getElementById('pretrained-model');
        selector.innerHTML = '<option value="">選択してください...</option>';
        
        completedSessions.forEach(session => {
            const option = document.createElement('option');
            option.value = session.best_model_path;
            option.textContent = `${session.config.model_name} (${formatDateTime(session.end_time)})`;
            selector.appendChild(option);
        });
        
    } catch (error) {
        console.error('事前学習済みモデル読み込みエラー:', error);
    }
}

/**
 * データセットバージョン変更
 */
function onDatasetVersionChange(event) {
    const versionId = event.target.value;
    
    if (!versionId) {
        document.getElementById('dataset-info').innerHTML = 'データセットを選択してください';
        return;
    }
    
    const version = datasetVersions.find(v => v.id === versionId);
    if (version) {
        document.getElementById('dataset-info').innerHTML = `
            <div class="row">
                <div class="col-6">
                    <small class="text-muted">バージョン</small>
                    <div>${version.version}</div>
                </div>
                <div class="col-6">
                    <small class="text-muted">作成日</small>
                    <div>${formatDateTime(version.created_at)}</div>
                </div>
            </div>
            <div class="row mt-2">
                <div class="col-6">
                    <small class="text-muted">フレーム数</small>
                    <div>${formatNumber(version.frame_count)}</div>
                </div>
                <div class="col-6">
                    <small class="text-muted">牌数</small>
                    <div>${formatNumber(version.tile_count)}</div>
                </div>
            </div>
            ${version.description ? `
                <div class="mt-2">
                    <small class="text-muted">説明</small>
                    <div class="small">${version.description}</div>
                </div>
            ` : ''}
        `;
    }
}

/**
 * セッションをフィルター
 */
function filterSessions() {
    const statusFilter = document.getElementById('status-filter').value;
    
    let filteredSessions = trainingSessions;
    if (statusFilter) {
        filteredSessions = trainingSessions.filter(s => s.status === statusFilter);
    }
    
    displayTrainingSessions(filteredSessions);
}

/**
 * セッション一覧を更新
 */
async function refreshSessions() {
    try {
        await loadTrainingSessions();
        filterSessions(); // 現在のフィルターを適用
    } catch (error) {
        console.error('セッション更新エラー:', error);
    }
}

/**
 * セッション詳細を表示
 */
function showSessionDetail(sessionId) {
    const session = trainingSessions.find(s => s.session_id === sessionId);
    if (!session) return;
    
    const modal = document.getElementById('sessionDetailModal');
    modal.dataset.sessionId = sessionId;
    
    // 基本情報を設定
    document.getElementById('detail-session-id').textContent = session.session_id;
    document.getElementById('detail-model-name').textContent = session.config.model_name;
    document.getElementById('detail-model-type').textContent = session.config.model_type;
    document.getElementById('detail-status').innerHTML = getStatusBadge(session.status);
    document.getElementById('detail-start-time').textContent = formatDateTime(session.start_time);
    document.getElementById('detail-end-time').textContent = session.end_time ? formatDateTime(session.end_time) : '-';
    
    // 設定情報を設定
    document.getElementById('detail-epochs').textContent = session.config.epochs;
    document.getElementById('detail-batch-size').textContent = session.config.batch_size;
    document.getElementById('detail-learning-rate').textContent = session.config.learning_rate;
    document.getElementById('detail-data-augmentation').textContent = session.config.use_data_augmentation ? 'あり' : 'なし';
    document.getElementById('detail-transfer-learning').textContent = session.config.transfer_learning ? 'あり' : 'なし';
    
    // メトリクスを表示
    displayMetrics(session.final_metrics || {});
    
    // 学習履歴チャートを表示
    displayTrainingHistory(session.training_history || []);
    
    // ボタンの表示/非表示
    document.getElementById('stop-training-btn').style.display = session.status === 'running' ? 'inline-block' : 'none';
    document.getElementById('continue-training-btn').style.display = session.status === 'completed' ? 'inline-block' : 'none';
    document.getElementById('download-model-btn').style.display = 
        (session.status === 'completed' && session.best_model_path) ? 'inline-block' : 'none';
    
    // モーダルを表示
    const modalInstance = new bootstrap.Modal(modal);
    modalInstance.show();
}

/**
 * セッション詳細モーダル表示時
 */
function onSessionDetailModalShow(event) {
    // チャートを初期化
    const ctx = document.getElementById('training-history-chart').getContext('2d');
    
    if (trainingHistoryChart) {
        trainingHistoryChart.destroy();
    }
    
    trainingHistoryChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: '訓練損失',
                data: [],
                borderColor: 'rgb(255, 99, 132)',
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                tension: 0.1
            }, {
                label: '検証損失',
                data: [],
                borderColor: 'rgb(54, 162, 235)',
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                tension: 0.1
            }, {
                label: '訓練精度',
                data: [],
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                tension: 0.1,
                yAxisID: 'y1'
            }, {
                label: '検証精度',
                data: [],
                borderColor: 'rgb(153, 102, 255)',
                backgroundColor: 'rgba(153, 102, 255, 0.2)',
                tension: 0.1,
                yAxisID: 'y1'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: '損失'
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: '精度'
                    },
                    grid: {
                        drawOnChartArea: false,
                    },
                }
            }
        }
    });
}

/**
 * メトリクスを表示
 */
function displayMetrics(metrics) {
    const container = document.getElementById('metrics-container');
    container.innerHTML = '';
    
    Object.entries(metrics).forEach(([key, value]) => {
        const metricCard = document.createElement('div');
        metricCard.className = 'metric-card';
        
        let displayValue = value;
        if (typeof value === 'number') {
            if (key.includes('accuracy') || key.includes('precision') || key.includes('recall')) {
                displayValue = formatPercentage(value);
            } else {
                displayValue = value.toFixed(4);
            }
        }
        
        metricCard.innerHTML = `
            <div class="metric-value">${displayValue}</div>
            <div class="metric-label">${key}</div>
        `;
        
        container.appendChild(metricCard);
    });
}

/**
 * 学習履歴を表示
 */
function displayTrainingHistory(history) {
    if (!trainingHistoryChart || history.length === 0) return;
    
    const epochs = history.map((_, index) => index + 1);
    const trainLoss = history.map(h => h.train_loss || 0);
    const valLoss = history.map(h => h.val_loss || 0);
    const trainAcc = history.map(h => h.train_accuracy || 0);
    const valAcc = history.map(h => h.val_accuracy || 0);
    
    trainingHistoryChart.data.labels = epochs;
    trainingHistoryChart.data.datasets[0].data = trainLoss;
    trainingHistoryChart.data.datasets[1].data = valLoss;
    trainingHistoryChart.data.datasets[2].data = trainAcc;
    trainingHistoryChart.data.datasets[3].data = valAcc;
    
    trainingHistoryChart.update();
}

/**
 * 学習進捗を更新
 */
function updateTrainingProgress(data) {
    // リアルタイムで進捗を更新
    showNotification(`学習進捗: ${data.message}`, 'info', 3000);
}

/**
 * セッション詳細を更新
 */
function updateSessionDetail(data) {
    // 進捗情報を更新
    if (data.progress) {
        // 進捗バーなどを更新
        console.log('セッション詳細更新:', data);
    }
}

/**
 * 学習を停止
 */
async function stopTraining() {
    const modal = document.getElementById('sessionDetailModal');
    const sessionId = modal.dataset.sessionId;
    
    if (!sessionId) return;
    
    try {
        await apiRequest(`/api/training/stop/${sessionId}`, {
            method: 'POST'
        });
        
        showNotification('学習停止要求を送信しました', 'info');
        
        // モーダルを閉じる
        const modalInstance = bootstrap.Modal.getInstance(modal);
        modalInstance.hide();
        
        // セッション一覧を更新
        setTimeout(() => refreshSessions(), 1000);
        
    } catch (error) {
        console.error('学習停止エラー:', error);
        showNotification('学習の停止に失敗しました', 'error');
    }
}

/**
 * 継続学習
 */
function continueTraining() {
    const modal = document.getElementById('sessionDetailModal');
    const sessionId = modal.dataset.sessionId;
    
    if (!sessionId) return;
    
    // 継続学習の設定画面を表示
    showNotification('継続学習機能は開発中です', 'info');
}

/**
 * モデルをダウンロード
 */
function downloadModel() {
    const modal = document.getElementById('sessionDetailModal');
    const sessionId = modal.dataset.sessionId;
    
    if (!sessionId) return;
    
    const session = trainingSessions.find(s => s.session_id === sessionId);
    if (session && session.best_model_path) {
        downloadFile(`/api/models/download/${sessionId}`, `${session.config.model_name}.pth`);
    }
}

/**
 * 学習セッションを停止
 */
async function stopTrainingSession(sessionId) {
    if (confirm('学習を停止しますか？')) {
        try {
            await apiRequest(`/api/training/stop/${sessionId}`, {
                method: 'POST'
            });
            
            showNotification('学習停止要求を送信しました', 'info');
            setTimeout(() => refreshSessions(), 1000);
            
        } catch (error) {
            console.error('学習停止エラー:', error);
            showNotification('学習の停止に失敗しました', 'error');
        }
    }
}

/**
 * セッションモデルをダウンロード
 */
function downloadSessionModel(sessionId) {
    const session = trainingSessions.find(s => s.session_id === sessionId);
    if (session && session.best_model_path) {
        downloadFile(`/api/models/download/${sessionId}`, `${session.config.model_name}.pth`);
    }
}

/**
 * モデル選択を切り替え
 */
function toggleModelSelection(sessionId) {
    const checkbox = document.getElementById(`select-${sessionId}`);
    
    if (checkbox.checked) {
        selectedModels.add(sessionId);
    } else {
        selectedModels.delete(sessionId);
    }
    
    // 比較ボタンの有効/無効を切り替え
    updateCompareButton();
}

/**
 * 比較ボタンを更新
 */
function updateCompareButton() {
    const compareBtn = document.querySelector('[onclick="showCompareModelsModal()"]');
    if (compareBtn) {
        compareBtn.disabled = selectedModels.size < 2;
    }
}

/**
 * モデル比較モーダルを表示
 */
function showCompareModelsModal() {
    if (selectedModels.size < 2) {
        showNotification('比較するには2つ以上のモデルを選択してください', 'warning');
        return;
    }
    
    const modal = new bootstrap.Modal(document.getElementById('compareModelsModal'));
    modal.show();
}

/**
 * モデル比較モーダル表示時
 */
function onCompareModelsModalShow(event) {
    const container = document.getElementById('model-selection-list');
    container.innerHTML = '';
    
    Array.from(selectedModels).forEach(sessionId => {
        const session = trainingSessions.find(s => s.session_id === sessionId);
        if (session) {
            const item = document.createElement('div');
            item.className = 'form-check';
            item.innerHTML = `
                <input class="form-check-input" type="checkbox" 
                       id="compare-${sessionId}" checked>
                <label class="form-check-label" for="compare-${sessionId}">
                    ${session.config.model_name} (${session.status})
                </label>
            `;
            container.appendChild(item);
        }
    });
}

/**
 * 選択されたモデルを比較
 */
async function compareSelectedModels() {
    const selectedForComparison = Array.from(selectedModels).filter(sessionId => {
        const checkbox = document.getElementById(`compare-${sessionId}`);
        return checkbox && checkbox.checked;
    });
    
    if (selectedForComparison.length < 2) {
        showNotification('比較するには2つ以上のモデルを選択してください', 'warning');
        return;
    }
    
    try {
        showLoading('モデル比較中...');
        
        const response = await apiRequest('/api/training/compare', {
            method: 'POST',
            body: JSON.stringify({
                session_ids: selectedForComparison
            })
        });
        
        displayComparisonResults(response);
        hideLoading();
        
    } catch (error) {
        console.error('モデル比較エラー:', error);
        showNotification('モデル比較に失敗しました', 'error');
        hideLoading();
    }
}

/**
 * 比較結果を表示
 */
function displayComparisonResults(results) {
    const resultsContainer = document.getElementById('comparison-results');
    resultsContainer.style.display = 'block';
    
    // 比較チャートを作成
    const ctx = document.getElementById('comparison-chart').getContext('2d');
    
    if (comparisonChart) {
        comparisonChart.destroy();
    }
    
    const labels = results.sessions.map(s => s.config.model_name);
    const accuracies = results.sessions.map(s => s.metrics.accuracy || 0);
    const losses = results.sessions.map(s => s.metrics.loss || 0);
    
    comparisonChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: '精度',
                data: accuracies,
                backgroundColor: 'rgba(54, 162, 235, 0.8)',
                yAxisID: 'y'
            }, {
                label: '損失',
                data: losses,
                backgroundColor: 'rgba(255, 99, 132, 0.8)',
                yAxisID: 'y1'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: '精度'
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: '損失'
                    },
                    grid: {
                        drawOnChartArea: false,
                    },
                }
            }
        }
    });
    
    // 比較テーブルを作成
    const tableBody = document.getElementById('comparison-table-body');
    tableBody.innerHTML = '';
    
    results.sessions.forEach(session => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${session.config.model_name}</td>
            <td>${session.metrics.accuracy ? formatPercentage(session.metrics.accuracy) : '-'}</td>
            <td>${session.metrics.loss ? session.metrics.loss.toFixed(4) : '-'}</td>
            <td>${session.training_time ? formatDuration(session.training_time) : '-'}</td>
            <td>-</td>
        `;
        tableBody.appendChild(row);
    });
}

console.log('学習管理機能が読み込まれました');