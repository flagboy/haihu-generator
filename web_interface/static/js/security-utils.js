/**
 * セキュリティユーティリティ
 * XSS対策やCSRF対策のための関数群
 */

/**
 * HTMLエスケープ
 * @param {string} text - エスケープするテキスト
 * @returns {string} エスケープ済みテキスト
 */
function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#x27;',
        '/': '&#x2F;'
    };
    return String(text).replace(/[&<>"'\/]/g, s => map[s]);
}

/**
 * 属性値のエスケープ
 * @param {string} text - エスケープするテキスト
 * @returns {string} エスケープ済みテキスト
 */
function escapeAttribute(text) {
    return String(text).replace(/['"<>&]/g, function(match) {
        switch (match) {
            case '"': return '&quot;';
            case "'": return '&#x27;';
            case '<': return '&lt;';
            case '>': return '&gt;';
            case '&': return '&amp;';
            default: return match;
        }
    });
}

/**
 * URLのサニタイズ
 * @param {string} url - サニタイズするURL
 * @returns {string} サニタイズ済みURL
 */
function sanitizeUrl(url) {
    try {
        const parsed = new URL(url, window.location.origin);
        // 許可するプロトコルのみ
        if (!['http:', 'https:', 'data:'].includes(parsed.protocol)) {
            return '#';
        }
        return parsed.href;
    } catch (e) {
        return '#';
    }
}

/**
 * CSRFトークンの取得
 * @returns {string} CSRFトークン
 */
function getCsrfToken() {
    // セッションからCSRFトークンを取得
    return sessionStorage.getItem('csrf_token') || '';
}

/**
 * CSRFトークンの設定
 * @param {string} token - CSRFトークン
 */
function setCsrfToken(token) {
    sessionStorage.setItem('csrf_token', token);
}

/**
 * セキュアなAJAXリクエスト
 * @param {string} url - リクエストURL
 * @param {object} options - リクエストオプション
 * @returns {Promise} レスポンス
 */
async function secureAjax(url, options = {}) {
    const defaultOptions = {
        credentials: 'same-origin',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRF-Token': getCsrfToken()
        }
    };

    const mergedOptions = {
        ...defaultOptions,
        ...options,
        headers: {
            ...defaultOptions.headers,
            ...options.headers
        }
    };

    try {
        const response = await fetch(url, mergedOptions);

        // CSRFトークンの更新（レスポンスヘッダーから）
        const newToken = response.headers.get('X-New-CSRF-Token');
        if (newToken) {
            setCsrfToken(newToken);
        }

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return response;
    } catch (error) {
        console.error('セキュアAJAXエラー:', error);
        throw error;
    }
}

/**
 * ファイルタイプの検証
 * @param {File} file - 検証するファイル
 * @param {Array<string>} allowedTypes - 許可するMIMEタイプ
 * @returns {boolean} 検証結果
 */
function validateFileType(file, allowedTypes) {
    return allowedTypes.includes(file.type);
}

/**
 * ファイルサイズの検証
 * @param {File} file - 検証するファイル
 * @param {number} maxSize - 最大サイズ（バイト）
 * @returns {boolean} 検証結果
 */
function validateFileSize(file, maxSize) {
    return file.size <= maxSize;
}

/**
 * 安全なDOMツリーの構築
 * @param {string} tag - HTMLタグ名
 * @param {object} attributes - 属性
 * @param {Array|string} children - 子要素
 * @returns {HTMLElement} DOM要素
 */
function createElement(tag, attributes = {}, children = []) {
    const element = document.createElement(tag);

    // 属性の設定
    for (const [key, value] of Object.entries(attributes)) {
        if (key === 'className') {
            element.className = value;
        } else if (key === 'style' && typeof value === 'object') {
            Object.assign(element.style, value);
        } else if (key.startsWith('data-')) {
            element.setAttribute(key, value);
        } else if (key === 'onclick' || key === 'onchange') {
            // イベントハンドラは直接設定しない
            console.warn('イベントハンドラは addEventListener を使用してください');
        } else {
            element.setAttribute(key, escapeAttribute(value));
        }
    }

    // 子要素の追加
    if (typeof children === 'string') {
        element.textContent = children;
    } else if (Array.isArray(children)) {
        children.forEach(child => {
            if (typeof child === 'string') {
                element.appendChild(document.createTextNode(child));
            } else if (child instanceof HTMLElement) {
                element.appendChild(child);
            }
        });
    }

    return element;
}

/**
 * 入力値のサニタイズ
 * @param {string} input - サニタイズする入力値
 * @param {string} type - 入力タイプ（text, number, email等）
 * @returns {string} サニタイズ済み入力値
 */
function sanitizeInput(input, type = 'text') {
    switch (type) {
        case 'number':
            return String(parseInt(input, 10) || 0);
        case 'email':
            return String(input).toLowerCase().trim();
        case 'url':
            return sanitizeUrl(input);
        case 'text':
        default:
            return escapeHtml(input);
    }
}

// グローバルに公開
window.SecurityUtils = {
    escapeHtml,
    escapeAttribute,
    sanitizeUrl,
    getCsrfToken,
    setCsrfToken,
    secureAjax,
    validateFileType,
    validateFileSize,
    createElement,
    sanitizeInput
};
