/**
 * 拡張キーボードショートカット機能
 *
 * 麻雀牌ラベリングの効率化のための包括的なショートカットシステム
 */

class EnhancedShortcutManager {
    constructor(labelingInterface) {
        this.interface = labelingInterface;
        this.shortcuts = this.defineShortcuts();
        this.enabled = true;
        this.helpVisible = false;
        this.setupEventListeners();
        this.createHelpOverlay();
    }

    defineShortcuts() {
        return {
            // ナビゲーション
            'ArrowRight': {
                action: () => this.interface.nextFrame(),
                description: '次のフレーム'
            },
            'ArrowLeft': {
                action: () => this.interface.previousFrame(),
                description: '前のフレーム'
            },
            'Space': {
                action: () => this.interface.togglePlayPause(),
                description: '再生/一時停止'
            },
            'Home': {
                action: () => this.interface.goToFirstFrame(),
                description: '最初のフレーム'
            },
            'End': {
                action: () => this.interface.goToLastFrame(),
                description: '最後のフレーム'
            },

            // ラベリング操作
            'Enter': {
                action: () => this.interface.confirmCurrentBox(),
                description: '現在のボックスを確定'
            },
            'Escape': {
                action: () => this.interface.cancelCurrentBox(),
                description: '現在のボックスをキャンセル'
            },
            'Delete': {
                action: () => this.interface.deleteSelectedBox(),
                description: '選択したボックスを削除'
            },
            'Backspace': {
                action: () => this.interface.deleteSelectedBox(),
                description: '選択したボックスを削除'
            },

            // コピー＆ペースト
            'Ctrl+C': {
                action: () => this.interface.copySelectedBox(),
                description: '選択したボックスをコピー'
            },
            'Ctrl+V': {
                action: () => this.interface.pasteBox(),
                description: 'ボックスを貼り付け'
            },
            'Alt+C': {
                action: () => this.interface.copyPreviousFrame(),
                description: '前フレームの全ボックスをコピー'
            },

            // 牌の種類選択（数字キー）
            '1': {
                action: () => this.interface.selectTileType('manzu'),
                description: '萬子を選択'
            },
            '2': {
                action: () => this.interface.selectTileType('pinzu'),
                description: '筒子を選択'
            },
            '3': {
                action: () => this.interface.selectTileType('souzu'),
                description: '索子を選択'
            },
            '4': {
                action: () => this.interface.selectTileType('jihai'),
                description: '字牌を選択'
            },

            // 牌の番号選択（テンキー）
            'Numpad1': {
                action: () => this.interface.selectTileNumber(1),
                description: '1を選択'
            },
            'Numpad2': {
                action: () => this.interface.selectTileNumber(2),
                description: '2を選択'
            },
            'Numpad3': {
                action: () => this.interface.selectTileNumber(3),
                description: '3を選択'
            },
            'Numpad4': {
                action: () => this.interface.selectTileNumber(4),
                description: '4を選択'
            },
            'Numpad5': {
                action: () => this.interface.selectTileNumber(5),
                description: '5を選択'
            },
            'Numpad6': {
                action: () => this.interface.selectTileNumber(6),
                description: '6を選択'
            },
            'Numpad7': {
                action: () => this.interface.selectTileNumber(7),
                description: '7を選択'
            },
            'Numpad8': {
                action: () => this.interface.selectTileNumber(8),
                description: '8を選択'
            },
            'Numpad9': {
                action: () => this.interface.selectTileNumber(9),
                description: '9を選択'
            },

            // 特殊牌
            'R': {
                action: () => this.interface.toggleRedDora(),
                description: '赤ドラ切り替え'
            },
            'B': {
                action: () => this.interface.selectBackTile(),
                description: '裏面牌を選択'
            },

            // 表示制御
            'G': {
                action: () => this.interface.toggleGrid(),
                description: 'グリッド表示切り替え'
            },
            'L': {
                action: () => this.interface.toggleLabels(),
                description: 'ラベル表示切り替え'
            },
            'H': {
                action: () => this.toggleHelp(),
                description: 'ヘルプ表示切り替え'
            },
            'O': {
                action: () => this.interface.toggleOverlay(),
                description: 'オーバーレイ切り替え'
            },

            // バッチ操作
            'Ctrl+A': {
                action: () => this.interface.selectAllBoxes(),
                description: 'すべてのボックスを選択'
            },
            'Ctrl+D': {
                action: () => this.interface.deselectAllBoxes(),
                description: 'すべての選択を解除'
            },
            'Ctrl+Shift+D': {
                action: () => this.interface.deleteAllBoxes(),
                description: 'すべてのボックスを削除'
            },

            // クイックアクション
            'Q': {
                action: () => this.interface.quickLabelMode(),
                description: 'クイックラベリングモード'
            },
            'W': {
                action: () => this.interface.switchToNextUnlabeled(),
                description: '次の未ラベルへ移動'
            },
            'E': {
                action: () => this.interface.switchToPreviousUnlabeled(),
                description: '前の未ラベルへ移動'
            },
            'S': {
                action: () => this.interface.saveProgress(),
                description: '進捗を保存'
            },

            // ズーム機能
            '+': {
                action: () => this.interface.zoomIn(),
                description: 'ズームイン'
            },
            '-': {
                action: () => this.interface.zoomOut(),
                description: 'ズームアウト'
            },
            '0': {
                action: () => this.interface.resetZoom(),
                description: 'ズームリセット'
            },

            // プレイヤー選択
            'F1': {
                action: () => this.interface.selectPlayer(1),
                description: 'プレイヤー1を選択'
            },
            'F2': {
                action: () => this.interface.selectPlayer(2),
                description: 'プレイヤー2を選択'
            },
            'F3': {
                action: () => this.interface.selectPlayer(3),
                description: 'プレイヤー3を選択'
            },
            'F4': {
                action: () => this.interface.selectPlayer(4),
                description: 'プレイヤー4を選択'
            },
        };
    }

    setupEventListeners() {
        // キーボードイベント
        document.addEventListener('keydown', (e) => this.handleKeyDown(e));

        // マウスイベント（Shift+クリックでの複数選択）
        document.addEventListener('mousedown', (e) => {
            if (e.shiftKey && e.target.classList.contains('annotation-box')) {
                e.preventDefault();
                this.interface.addToSelection(e.target);
            }
        });
    }

    handleKeyDown(event) {
        // 入力フィールドにフォーカスがある場合は無効
        if (this.interface.isInputFocused()) {
            return;
        }

        // ショートカットが無効な場合
        if (!this.enabled) {
            return;
        }

        const key = this.getKeyCombo(event);
        const shortcut = this.shortcuts[key];

        if (shortcut && shortcut.action) {
            event.preventDefault();
            shortcut.action();
            this.showShortcutFeedback(key);
        }
    }

    getKeyCombo(event) {
        let combo = '';
        if (event.ctrlKey || event.metaKey) combo += 'Ctrl+';
        if (event.altKey) combo += 'Alt+';
        if (event.shiftKey) combo += 'Shift+';

        // 特殊キーの処理
        let key = event.key;
        if (key === ' ') key = 'Space';
        if (key === '+' || key === '=') key = '+';
        if (key === '_' || key === '-') key = '-';

        combo += key;
        return combo;
    }

    showShortcutFeedback(key) {
        // 既存のフィードバックを削除
        const existing = document.querySelector('.shortcut-feedback');
        if (existing) {
            existing.remove();
        }

        // ビジュアルフィードバックの表示
        const feedback = document.createElement('div');
        feedback.className = 'shortcut-feedback';
        feedback.textContent = key;
        feedback.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            font-family: monospace;
            font-size: 16px;
            z-index: 10000;
            animation: fadeOut 0.5s ease-in-out;
        `;

        document.body.appendChild(feedback);

        setTimeout(() => {
            feedback.remove();
        }, 500);
    }

    createHelpOverlay() {
        // ヘルプオーバーレイの作成
        const overlay = document.createElement('div');
        overlay.id = 'shortcut-help-overlay';
        overlay.className = 'shortcut-help-overlay';
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 40px;
            overflow-y: auto;
            z-index: 10001;
            display: none;
            font-family: sans-serif;
        `;

        // ヘルプコンテンツの生成
        const content = document.createElement('div');
        content.style.cssText = `
            max-width: 800px;
            margin: 0 auto;
        `;

        content.innerHTML = `
            <h2 style="text-align: center; margin-bottom: 30px;">キーボードショートカット一覧</h2>
            <div style="columns: 2; column-gap: 40px;">
                ${this.generateHelpContent()}
            </div>
            <p style="text-align: center; margin-top: 30px; opacity: 0.7;">
                Hキーまたはクリックで閉じる
            </p>
        `;

        overlay.appendChild(content);
        document.body.appendChild(overlay);

        // クリックで閉じる
        overlay.addEventListener('click', () => {
            this.toggleHelp();
        });
    }

    generateHelpContent() {
        const categories = {
            'ナビゲーション': ['ArrowRight', 'ArrowLeft', 'Space', 'Home', 'End', 'W', 'E'],
            'ラベリング操作': ['Enter', 'Escape', 'Delete', 'Q'],
            'コピー＆ペースト': ['Ctrl+C', 'Ctrl+V', 'Alt+C'],
            '牌の選択': ['1', '2', '3', '4', 'Numpad1-9', 'R', 'B'],
            '表示制御': ['G', 'L', 'H', 'O', '+', '-', '0'],
            'バッチ操作': ['Ctrl+A', 'Ctrl+D', 'Ctrl+Shift+D'],
            'その他': ['S', 'F1-F4']
        };

        let html = '';
        for (const [category, keys] of Object.entries(categories)) {
            html += `<div style="break-inside: avoid; margin-bottom: 20px;">`;
            html += `<h3 style="color: #4CAF50; margin-bottom: 10px;">${category}</h3>`;
            html += '<table style="width: 100%;">';

            for (const key of keys) {
                if (key.includes('-')) {
                    // 範囲表記の処理
                    const shortcut = this.shortcuts[key.split('-')[0].replace(/\d/, '1')];
                    if (shortcut) {
                        html += `<tr>
                            <td style="padding: 4px; font-family: monospace;">${key}</td>
                            <td style="padding: 4px; opacity: 0.8;">${shortcut.description}</td>
                        </tr>`;
                    }
                } else {
                    const shortcut = this.shortcuts[key];
                    if (shortcut) {
                        html += `<tr>
                            <td style="padding: 4px; font-family: monospace;">${key}</td>
                            <td style="padding: 4px; opacity: 0.8;">${shortcut.description}</td>
                        </tr>`;
                    }
                }
            }

            html += '</table></div>';
        }

        return html;
    }

    toggleHelp() {
        const overlay = document.getElementById('shortcut-help-overlay');
        if (overlay) {
            this.helpVisible = !this.helpVisible;
            overlay.style.display = this.helpVisible ? 'block' : 'none';
        }
    }

    enable() {
        this.enabled = true;
    }

    disable() {
        this.enabled = false;
    }
}

// クイックラベリングモード
class QuickLabelingMode {
    constructor(interface) {
        this.interface = interface;
        this.enabled = false;
        this.lastTileType = null;
        this.lastTileNumber = null;
        this.defaultBoxSize = { width: 50, height: 70 };
    }

    enable() {
        if (this.enabled) return;

        this.enabled = true;
        this.interface.showMessage('クイックラベリングモード: ON');
        this.setupQuickMode();

        // カーソルを変更
        document.body.style.cursor = 'crosshair';
    }

    disable() {
        if (!this.enabled) return;

        this.enabled = false;
        this.interface.showMessage('クイックラベリングモード: OFF');
        this.teardownQuickMode();

        // カーソルを元に戻す
        document.body.style.cursor = 'default';
    }

    toggle() {
        if (this.enabled) {
            this.disable();
        } else {
            this.enable();
        }
    }

    setupQuickMode() {
        this.clickHandler = (e) => this.handleClick(e);
        this.interface.canvas.addEventListener('click', this.clickHandler);
    }

    teardownQuickMode() {
        if (this.clickHandler) {
            this.interface.canvas.removeEventListener('click', this.clickHandler);
            this.clickHandler = null;
        }
    }

    handleClick(event) {
        if (!this.enabled) return;

        const rect = this.interface.canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        // キャンバスの座標系に変換
        const canvasX = x * (this.interface.canvas.width / rect.width);
        const canvasY = y * (this.interface.canvas.height / rect.height);

        // 最後に使用した牌種で自動的にボックスを作成
        if (this.lastTileType && this.lastTileNumber !== null) {
            const bbox = {
                x: canvasX - this.defaultBoxSize.width / 2,
                y: canvasY - this.defaultBoxSize.height / 2,
                width: this.defaultBoxSize.width,
                height: this.defaultBoxSize.height
            };

            this.interface.createBox(bbox, this.lastTileType, this.lastTileNumber);

            // フィードバック
            this.showQuickFeedback(canvasX, canvasY);
        } else {
            this.interface.showMessage('先に牌の種類を選択してください');
        }
    }

    setLastTile(type, number) {
        this.lastTileType = type;
        this.lastTileNumber = number;
    }

    showQuickFeedback(x, y) {
        // クイック作成のビジュアルフィードバック
        const feedback = document.createElement('div');
        feedback.style.cssText = `
            position: absolute;
            left: ${x}px;
            top: ${y}px;
            width: 20px;
            height: 20px;
            border: 2px solid #4CAF50;
            border-radius: 50%;
            pointer-events: none;
            animation: quickPulse 0.3s ease-out;
            transform: translate(-50%, -50%);
        `;

        this.interface.canvas.parentElement.appendChild(feedback);

        setTimeout(() => {
            feedback.remove();
        }, 300);
    }
}

// CSS アニメーションの追加
const style = document.createElement('style');
style.textContent = `
    @keyframes fadeOut {
        from { opacity: 1; }
        to { opacity: 0; }
    }

    @keyframes quickPulse {
        from {
            transform: translate(-50%, -50%) scale(0);
            opacity: 1;
        }
        to {
            transform: translate(-50%, -50%) scale(2);
            opacity: 0;
        }
    }

    .shortcut-feedback {
        pointer-events: none;
    }

    .annotation-box {
        transition: all 0.2s ease;
    }

    .annotation-box.selected {
        border-color: #FFD700 !important;
        box-shadow: 0 0 10px rgba(255, 215, 0, 0.5);
    }
`;
document.head.appendChild(style);

// エクスポート
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { EnhancedShortcutManager, QuickLabelingMode };
}
