/**
 * 麻雀牌検出システム - 共通JavaScript
 */

// グローバル変数
let socket = null;
let loadingOverlay = null;

// 初期化
document.addEventListener('DOMContentLoaded', function() {
    initializeCommon();
});

/**
 * 共通初期化
 */
function initializeCommon() {
    // ローディングオーバーレイを作成
    createLoadingOverlay();
    
    // ツールチップを初期化
    initializeTooltips();
    
    // 共通イベントリスナーを設定
    setupCommonEventListeners();
}

/**
 * ローディングオーバーレイを作成
 */
function createLoadingOverlay() {
    loadingOverlay = document.createElement('div');
    loadingOverlay.className = 'loading-overlay';
    loadingOverlay.style.display = 'none';
    loadingOverlay.innerHTML = `
        <div class="text-center text-white">
            <div class="loading-spinner mb-3"></div>
            <div id="loading-message">処理中...</div>
        </div>
    `;
    document.body.appendChild(loadingOverlay);
}

/**
 * ローディング表示
 */
function showLoading(message = '処理中...') {
    if (loadingOverlay) {
        document.getElementById('loading-message').textContent = message;
        loadingOverlay.style.display = 'flex';
    }
}

/**
 * ローディング非表示
 */
function hideLoading() {
    if (loadingOverlay) {
        loadingOverlay.style.display = 'none';
    }
}

/**
 * 通知表示
 */
function showNotification(message, type = 'info', duration = 5000) {
    const notificationArea = document.getElementById('notification-area');
    if (!notificationArea) return;
    
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} notification fade-in`;
    notification.setAttribute('role', 'alert');
    
    // アイコンを設定
    let icon = 'fas fa-info-circle';
    switch (type) {
        case 'success':
            icon = 'fas fa-check-circle';
            break;
        case 'warning':
            icon = 'fas fa-exclamation-triangle';
            break;
        case 'error':
        case 'danger':
            icon = 'fas fa-times-circle';
            break;
    }
    
    notification.innerHTML = `
        <div class="d-flex align-items-center">
            <i class="${icon} me-2"></i>
            <div class="flex-grow-1">${message}</div>
            <button type="button" class="btn-close btn-close-white ms-2" onclick="closeNotification(this)"></button>
        </div>
    `;
    
    notificationArea.appendChild(notification);
    
    // 自動で削除
    if (duration > 0) {
        setTimeout(() => {
            closeNotification(notification.querySelector('.btn-close'));
        }, duration);
    }
}

/**
 * 通知を閉じる
 */
function closeNotification(button) {
    const notification = button.closest('.notification');
    if (notification) {
        notification.classList.add('fade-out');
        setTimeout(() => {
            notification.remove();
        }, 300);
    }
}

/**
 * ツールチップ初期化
 */
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

/**
 * 共通イベントリスナー設定
 */
function setupCommonEventListeners() {
    // ESCキーでモーダルを閉じる
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            const modals = document.querySelectorAll('.modal.show');
            modals.forEach(modal => {
                const modalInstance = bootstrap.Modal.getInstance(modal);
                if (modalInstance) {
                    modalInstance.hide();
                }
            });
        }
    });
    
    // フォームの送信時にローディングを表示
    document.addEventListener('submit', function(e) {
        const form = e.target;
        if (form.tagName === 'FORM' && !form.hasAttribute('data-no-loading')) {
            showLoading('送信中...');
        }
    });
}

/**
 * APIリクエスト共通関数
 */
async function apiRequest(url, options = {}) {
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
        },
    };
    
    const mergedOptions = { ...defaultOptions, ...options };
    
    try {
        const response = await fetch(url, mergedOptions);
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || `HTTP error! status: ${response.status}`);
        }
        
        return data;
    } catch (error) {
        console.error('API request error:', error);
        throw error;
    }
}

/**
 * ファイルサイズをフォーマット
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * 時間をフォーマット
 */
function formatDuration(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
        return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    } else {
        return `${minutes}:${secs.toString().padStart(2, '0')}`;
    }
}

/**
 * 日時をフォーマット
 */
function formatDateTime(dateString) {
    const date = new Date(dateString);
    return date.toLocaleString('ja-JP', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
}

/**
 * 相対時間を取得
 */
function getRelativeTime(dateString) {
    const now = new Date();
    const date = new Date(dateString);
    const diffMs = now - date;
    const diffSecs = Math.floor(diffMs / 1000);
    const diffMins = Math.floor(diffSecs / 60);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);
    
    if (diffDays > 0) {
        return `${diffDays}日前`;
    } else if (diffHours > 0) {
        return `${diffHours}時間前`;
    } else if (diffMins > 0) {
        return `${diffMins}分前`;
    } else {
        return '今';
    }
}

/**
 * 数値をフォーマット
 */
function formatNumber(num, decimals = 0) {
    return num.toLocaleString('ja-JP', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    });
}

/**
 * パーセンテージをフォーマット
 */
function formatPercentage(value, decimals = 1) {
    return (value * 100).toFixed(decimals) + '%';
}

/**
 * 確認ダイアログ
 */
function confirmAction(message, callback) {
    if (confirm(message)) {
        callback();
    }
}

/**
 * ファイルダウンロード
 */
function downloadFile(url, filename) {
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

/**
 * クリップボードにコピー
 */
async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        showNotification('クリップボードにコピーしました', 'success', 2000);
    } catch (err) {
        console.error('クリップボードへのコピーに失敗:', err);
        showNotification('クリップボードへのコピーに失敗しました', 'error');
    }
}

/**
 * 要素の表示/非表示を切り替え
 */
function toggleElement(element, show = null) {
    if (typeof element === 'string') {
        element = document.getElementById(element);
    }
    
    if (!element) return;
    
    if (show === null) {
        show = element.style.display === 'none';
    }
    
    element.style.display = show ? 'block' : 'none';
}

/**
 * 要素を無効/有効にする
 */
function toggleElementDisabled(element, disabled = null) {
    if (typeof element === 'string') {
        element = document.getElementById(element);
    }
    
    if (!element) return;
    
    if (disabled === null) {
        disabled = !element.disabled;
    }
    
    element.disabled = disabled;
    
    if (disabled) {
        element.classList.add('disabled');
    } else {
        element.classList.remove('disabled');
    }
}

/**
 * フォームデータを取得
 */
function getFormData(formElement) {
    const formData = new FormData(formElement);
    const data = {};
    
    for (let [key, value] of formData.entries()) {
        // チェックボックスの処理
        if (formElement.querySelector(`[name="${key}"][type="checkbox"]`)) {
            data[key] = formElement.querySelector(`[name="${key}"]`).checked;
        }
        // 数値の処理
        else if (formElement.querySelector(`[name="${key}"][type="number"]`)) {
            data[key] = parseFloat(value) || 0;
        }
        // その他
        else {
            data[key] = value;
        }
    }
    
    return data;
}

/**
 * フォームにデータを設定
 */
function setFormData(formElement, data) {
    for (let [key, value] of Object.entries(data)) {
        const element = formElement.querySelector(`[name="${key}"]`);
        if (!element) continue;
        
        if (element.type === 'checkbox') {
            element.checked = Boolean(value);
        } else if (element.type === 'radio') {
            const radioElement = formElement.querySelector(`[name="${key}"][value="${value}"]`);
            if (radioElement) {
                radioElement.checked = true;
            }
        } else {
            element.value = value;
        }
    }
}

/**
 * テーブルをソート
 */
function sortTable(table, columnIndex, ascending = true) {
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    
    rows.sort((a, b) => {
        const aText = a.cells[columnIndex].textContent.trim();
        const bText = b.cells[columnIndex].textContent.trim();
        
        // 数値かどうかチェック
        const aNum = parseFloat(aText);
        const bNum = parseFloat(bText);
        
        if (!isNaN(aNum) && !isNaN(bNum)) {
            return ascending ? aNum - bNum : bNum - aNum;
        } else {
            return ascending ? aText.localeCompare(bText) : bText.localeCompare(aText);
        }
    });
    
    // ソート後の行を再配置
    rows.forEach(row => tbody.appendChild(row));
}

/**
 * 検索フィルター
 */
function filterTable(table, searchText, columnIndexes = null) {
    const tbody = table.querySelector('tbody');
    const rows = tbody.querySelectorAll('tr');
    
    rows.forEach(row => {
        let shouldShow = false;
        
        if (columnIndexes) {
            // 指定された列のみを検索
            columnIndexes.forEach(index => {
                const cell = row.cells[index];
                if (cell && cell.textContent.toLowerCase().includes(searchText.toLowerCase())) {
                    shouldShow = true;
                }
            });
        } else {
            // すべての列を検索
            const rowText = row.textContent.toLowerCase();
            shouldShow = rowText.includes(searchText.toLowerCase());
        }
        
        row.style.display = shouldShow ? '' : 'none';
    });
}

/**
 * デバウンス関数
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * スロットル関数
 */
function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

/**
 * ローカルストレージヘルパー
 */
const storage = {
    set: function(key, value) {
        try {
            localStorage.setItem(key, JSON.stringify(value));
        } catch (e) {
            console.error('ローカルストレージへの保存に失敗:', e);
        }
    },
    
    get: function(key, defaultValue = null) {
        try {
            const item = localStorage.getItem(key);
            return item ? JSON.parse(item) : defaultValue;
        } catch (e) {
            console.error('ローカルストレージからの読み込みに失敗:', e);
            return defaultValue;
        }
    },
    
    remove: function(key) {
        try {
            localStorage.removeItem(key);
        } catch (e) {
            console.error('ローカルストレージからの削除に失敗:', e);
        }
    },
    
    clear: function() {
        try {
            localStorage.clear();
        } catch (e) {
            console.error('ローカルストレージのクリアに失敗:', e);
        }
    }
};

/**
 * URLパラメータを取得
 */
function getUrlParameter(name) {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get(name);
}

/**
 * URLパラメータを設定
 */
function setUrlParameter(name, value) {
    const url = new URL(window.location);
    url.searchParams.set(name, value);
    window.history.pushState({}, '', url);
}

// エクスポート（モジュール使用時）
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        showLoading,
        hideLoading,
        showNotification,
        apiRequest,
        formatFileSize,
        formatDuration,
        formatDateTime,
        getRelativeTime,
        formatNumber,
        formatPercentage,
        storage
    };
}