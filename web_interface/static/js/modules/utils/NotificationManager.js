/**
 * 通知管理クラス
 */
export class NotificationManager {
    constructor(containerId = 'notification-container') {
        this.containerId = containerId;
        this.container = null;
        this.notifications = new Map();
        this.defaultDuration = 3000;

        // 通知タイプの設定
        this.types = {
            success: {
                icon: 'fas fa-check-circle',
                className: 'alert-success'
            },
            error: {
                icon: 'fas fa-exclamation-circle',
                className: 'alert-danger'
            },
            warning: {
                icon: 'fas fa-exclamation-triangle',
                className: 'alert-warning'
            },
            info: {
                icon: 'fas fa-info-circle',
                className: 'alert-info'
            }
        };

        this.init();
    }

    /**
     * 初期化
     */
    init() {
        // コンテナが存在しない場合は作成
        this.container = document.getElementById(this.containerId);
        if (!this.container) {
            this.createContainer();
        }
    }

    /**
     * 通知コンテナを作成
     */
    createContainer() {
        this.container = document.createElement('div');
        this.container.id = this.containerId;
        this.container.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 9999;
            max-width: 350px;
        `;
        document.body.appendChild(this.container);
    }

    /**
     * 通知を表示
     */
    show(message, type = 'info', options = {}) {
        const {
            duration = this.defaultDuration,
            closable = true,
            progress = true,
            actions = []
        } = options;

        const id = this.generateId();
        const notification = this.createNotification(id, message, type, {
            closable,
            progress,
            actions
        });

        // DOMに追加
        this.container.appendChild(notification.element);
        this.notifications.set(id, notification);

        // アニメーション
        requestAnimationFrame(() => {
            notification.element.style.opacity = '1';
            notification.element.style.transform = 'translateX(0)';
        });

        // 自動削除
        if (duration > 0) {
            notification.timeout = setTimeout(() => {
                this.remove(id);
            }, duration);

            // プログレスバーアニメーション
            if (progress) {
                this.animateProgress(notification.progressBar, duration);
            }
        }

        return id;
    }

    /**
     * 通知要素を作成
     */
    createNotification(id, message, type, options) {
        const typeConfig = this.types[type] || this.types.info;

        const element = document.createElement('div');
        element.className = `alert ${typeConfig.className} alert-dismissible fade show mb-2`;
        element.style.cssText = `
            opacity: 0;
            transform: translateX(100%);
            transition: all 0.3s ease-out;
            position: relative;
            overflow: hidden;
        `;

        // アイコンとメッセージ
        const content = document.createElement('div');
        content.className = 'd-flex align-items-start';
        content.innerHTML = `
            <i class="${typeConfig.icon} me-2 mt-1"></i>
            <div class="flex-grow-1">${this.escapeHtml(message)}</div>
        `;
        element.appendChild(content);

        // 閉じるボタン
        if (options.closable) {
            const closeBtn = document.createElement('button');
            closeBtn.type = 'button';
            closeBtn.className = 'btn-close';
            closeBtn.style.cssText = 'position: absolute; top: 5px; right: 5px;';
            closeBtn.onclick = () => this.remove(id);
            element.appendChild(closeBtn);
        }

        // アクションボタン
        if (options.actions.length > 0) {
            const actionsDiv = document.createElement('div');
            actionsDiv.className = 'mt-2';
            options.actions.forEach(action => {
                const btn = document.createElement('button');
                btn.className = `btn btn-sm ${action.className || 'btn-primary'} me-2`;
                btn.textContent = action.label;
                btn.onclick = () => {
                    action.callback();
                    if (action.closeOnClick !== false) {
                        this.remove(id);
                    }
                };
                actionsDiv.appendChild(btn);
            });
            content.appendChild(actionsDiv);
        }

        // プログレスバー
        let progressBar = null;
        if (options.progress) {
            progressBar = document.createElement('div');
            progressBar.style.cssText = `
                position: absolute;
                bottom: 0;
                left: 0;
                height: 3px;
                background-color: currentColor;
                opacity: 0.3;
                width: 100%;
                transform-origin: left;
            `;
            element.appendChild(progressBar);
        }

        return {
            id,
            element,
            progressBar,
            timeout: null
        };
    }

    /**
     * 通知を削除
     */
    remove(id) {
        const notification = this.notifications.get(id);
        if (!notification) return;

        // タイムアウトをクリア
        if (notification.timeout) {
            clearTimeout(notification.timeout);
        }

        // フェードアウトアニメーション
        notification.element.style.opacity = '0';
        notification.element.style.transform = 'translateX(100%)';

        // DOMから削除
        setTimeout(() => {
            if (notification.element.parentNode) {
                notification.element.parentNode.removeChild(notification.element);
            }
            this.notifications.delete(id);
        }, 300);
    }

    /**
     * すべての通知をクリア
     */
    clear() {
        this.notifications.forEach((notification, id) => {
            this.remove(id);
        });
    }

    /**
     * 成功通知
     */
    success(message, options) {
        return this.show(message, 'success', options);
    }

    /**
     * エラー通知
     */
    error(message, options) {
        return this.show(message, 'error', {
            duration: 5000,
            ...options
        });
    }

    /**
     * 警告通知
     */
    warning(message, options) {
        return this.show(message, 'warning', options);
    }

    /**
     * 情報通知
     */
    info(message, options) {
        return this.show(message, 'info', options);
    }

    /**
     * 確認ダイアログ
     */
    confirm(message, onConfirm, onCancel) {
        return this.show(message, 'warning', {
            duration: 0,
            closable: false,
            progress: false,
            actions: [
                {
                    label: 'はい',
                    className: 'btn-primary',
                    callback: onConfirm
                },
                {
                    label: 'いいえ',
                    className: 'btn-secondary',
                    callback: onCancel || (() => {})
                }
            ]
        });
    }

    /**
     * ローディング通知
     */
    loading(message = '処理中...') {
        const spinner = '<span class="spinner-border spinner-border-sm me-2"></span>';
        return this.show(spinner + message, 'info', {
            duration: 0,
            closable: false,
            progress: false
        });
    }

    /**
     * プログレスバーをアニメーション
     */
    animateProgress(progressBar, duration) {
        if (!progressBar) return;

        progressBar.style.transition = `transform ${duration}ms linear`;
        progressBar.style.transform = 'scaleX(0)';
    }

    /**
     * IDを生成
     */
    generateId() {
        return `notification_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    /**
     * HTMLエスケープ
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * 通知の更新
     */
    update(id, message, type) {
        const notification = this.notifications.get(id);
        if (!notification) return;

        const content = notification.element.querySelector('.flex-grow-1');
        if (content) {
            content.innerHTML = this.escapeHtml(message);
        }

        if (type && this.types[type]) {
            const icon = notification.element.querySelector('i');
            if (icon) {
                icon.className = this.types[type].icon + ' me-2 mt-1';
            }

            // クラスを更新
            Object.values(this.types).forEach(t => {
                notification.element.classList.remove(t.className);
            });
            notification.element.classList.add(this.types[type].className);
        }
    }

    /**
     * トースト通知（簡易版）
     */
    toast(message, duration = 2000) {
        return this.show(message, 'info', {
            duration,
            closable: false,
            progress: false
        });
    }

    /**
     * スナックバー（画面下部に表示）
     */
    snackbar(message, action = null) {
        // 一時的にコンテナ位置を変更
        const originalStyle = this.container.style.cssText;
        this.container.style.cssText = `
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 9999;
        `;

        const id = this.show(message, 'info', {
            duration: 4000,
            closable: false,
            progress: false,
            actions: action ? [action] : []
        });

        // 元の位置に戻す
        setTimeout(() => {
            this.container.style.cssText = originalStyle;
        }, 4100);

        return id;
    }

    /**
     * クリーンアップ
     */
    destroy() {
        this.clear();
        if (this.container && this.container.parentNode) {
            this.container.parentNode.removeChild(this.container);
        }
    }
}

// シングルトンインスタンスをエクスポート
export const notificationManager = new NotificationManager();
