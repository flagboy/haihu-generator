/**
 * キーボードショートカット管理クラス
 */
export class KeyboardShortcutManager {
    constructor() {
        this.shortcuts = new Map();
        this.enabled = true;
        this.activeModifiers = new Set();

        // イベントリスナーをバインド
        this.handleKeyDown = this.handleKeyDown.bind(this);
        this.handleKeyUp = this.handleKeyUp.bind(this);

        // イベントリスナーを登録
        document.addEventListener('keydown', this.handleKeyDown);
        document.addEventListener('keyup', this.handleKeyUp);
    }

    /**
     * ショートカットを登録
     */
    register(shortcut, callback, options = {}) {
        const {
            description = '',
            preventDefault = true,
            allowInInput = false,
            ctrl = false,
            shift = false,
            alt = false,
            meta = false
        } = options;

        const key = this.normalizeKey(shortcut);
        const id = this.generateShortcutId(key, { ctrl, shift, alt, meta });

        this.shortcuts.set(id, {
            key,
            callback,
            description,
            preventDefault,
            allowInInput,
            modifiers: { ctrl, shift, alt, meta }
        });

        return id;
    }

    /**
     * ショートカットを解除
     */
    unregister(shortcutId) {
        return this.shortcuts.delete(shortcutId);
    }

    /**
     * 複数のショートカットを一括登録
     */
    registerBatch(shortcuts) {
        const ids = [];
        for (const [shortcut, callback, options] of shortcuts) {
            ids.push(this.register(shortcut, callback, options));
        }
        return ids;
    }

    /**
     * すべてのショートカットをクリア
     */
    clear() {
        this.shortcuts.clear();
    }

    /**
     * ショートカットを有効/無効化
     */
    setEnabled(enabled) {
        this.enabled = enabled;
    }

    /**
     * キーダウンイベントハンドラー
     */
    handleKeyDown(event) {
        if (!this.enabled) return;

        // モディファイアキーを記録
        if (event.ctrlKey) this.activeModifiers.add('ctrl');
        if (event.shiftKey) this.activeModifiers.add('shift');
        if (event.altKey) this.activeModifiers.add('alt');
        if (event.metaKey) this.activeModifiers.add('meta');

        // 入力フィールドでのショートカットをチェック
        const isInInput = this.isInputElement(event.target);

        // ショートカットを検索
        const key = this.normalizeKey(event.key);
        const shortcutId = this.generateShortcutId(key, {
            ctrl: event.ctrlKey,
            shift: event.shiftKey,
            alt: event.altKey,
            meta: event.metaKey
        });

        const shortcut = this.shortcuts.get(shortcutId);

        if (shortcut && (!isInInput || shortcut.allowInInput)) {
            if (shortcut.preventDefault) {
                event.preventDefault();
            }

            try {
                shortcut.callback(event);
            } catch (error) {
                console.error('Error in keyboard shortcut handler:', error);
            }
        }
    }

    /**
     * キーアップイベントハンドラー
     */
    handleKeyUp(event) {
        // モディファイアキーをクリア
        if (!event.ctrlKey) this.activeModifiers.delete('ctrl');
        if (!event.shiftKey) this.activeModifiers.delete('shift');
        if (!event.altKey) this.activeModifiers.delete('alt');
        if (!event.metaKey) this.activeModifiers.delete('meta');
    }

    /**
     * 入力要素かどうかチェック
     */
    isInputElement(element) {
        const tagName = element.tagName.toLowerCase();
        return tagName === 'input' ||
               tagName === 'textarea' ||
               tagName === 'select' ||
               element.contentEditable === 'true';
    }

    /**
     * キーを正規化
     */
    normalizeKey(key) {
        // 特殊キーのマッピング
        const keyMap = {
            'ArrowUp': 'up',
            'ArrowDown': 'down',
            'ArrowLeft': 'left',
            'ArrowRight': 'right',
            'Enter': 'enter',
            'Escape': 'escape',
            'Delete': 'delete',
            'Backspace': 'backspace',
            'Tab': 'tab',
            ' ': 'space',
            'PageUp': 'pageup',
            'PageDown': 'pagedown',
            'Home': 'home',
            'End': 'end'
        };

        return keyMap[key] || key.toLowerCase();
    }

    /**
     * ショートカットIDを生成
     */
    generateShortcutId(key, modifiers) {
        const parts = [];
        if (modifiers.ctrl) parts.push('ctrl');
        if (modifiers.shift) parts.push('shift');
        if (modifiers.alt) parts.push('alt');
        if (modifiers.meta) parts.push('meta');
        parts.push(key);
        return parts.join('+');
    }

    /**
     * 登録されているショートカット一覧を取得
     */
    getShortcuts() {
        const shortcuts = [];
        this.shortcuts.forEach((shortcut, id) => {
            shortcuts.push({
                id,
                key: shortcut.key,
                modifiers: shortcut.modifiers,
                description: shortcut.description,
                displayKey: this.formatShortcutDisplay(shortcut.key, shortcut.modifiers)
            });
        });
        return shortcuts;
    }

    /**
     * ショートカットの表示形式を生成
     */
    formatShortcutDisplay(key, modifiers) {
        const parts = [];
        if (modifiers.ctrl) parts.push('Ctrl');
        if (modifiers.shift) parts.push('Shift');
        if (modifiers.alt) parts.push('Alt');
        if (modifiers.meta) parts.push(this.isMac() ? 'Cmd' : 'Win');

        // キーの表示名
        const displayKey = key.charAt(0).toUpperCase() + key.slice(1);
        parts.push(displayKey);

        return parts.join('+');
    }

    /**
     * Macかどうか判定
     */
    isMac() {
        return navigator.platform.toUpperCase().indexOf('MAC') >= 0;
    }

    /**
     * 現在押されているモディファイアキーを取得
     */
    getActiveModifiers() {
        return new Set(this.activeModifiers);
    }

    /**
     * ショートカットヘルプを生成
     */
    generateHelp() {
        const shortcuts = this.getShortcuts();
        const grouped = {};

        // カテゴリ別にグループ化
        shortcuts.forEach(shortcut => {
            const category = this.categorizeShortcut(shortcut.key);
            if (!grouped[category]) {
                grouped[category] = [];
            }
            grouped[category].push(shortcut);
        });

        return grouped;
    }

    /**
     * ショートカットをカテゴリ分け
     */
    categorizeShortcut(key) {
        if (['up', 'down', 'left', 'right'].includes(key)) {
            return 'ナビゲーション';
        } else if (['1', '2', '3', '4', '5', '6', '7', '8', '9'].includes(key)) {
            return '牌選択';
        } else if (['a', 's', 'd', 'f', 'g', 'h', 'j'].includes(key)) {
            return '字牌選択';
        } else if (['enter', 'escape', 'delete', 'backspace'].includes(key)) {
            return '編集';
        } else if (key === 's' || key === 'z' || key === 'y') {
            return 'ファイル操作';
        } else {
            return 'その他';
        }
    }

    /**
     * ショートカットをテスト（デバッグ用）
     */
    test(shortcutString) {
        const parts = shortcutString.toLowerCase().split('+');
        const key = parts[parts.length - 1];
        const modifiers = {
            ctrl: parts.includes('ctrl'),
            shift: parts.includes('shift'),
            alt: parts.includes('alt'),
            meta: parts.includes('meta') || parts.includes('cmd')
        };

        const event = new KeyboardEvent('keydown', {
            key: key,
            ctrlKey: modifiers.ctrl,
            shiftKey: modifiers.shift,
            altKey: modifiers.alt,
            metaKey: modifiers.meta
        });

        this.handleKeyDown(event);
    }

    /**
     * クリーンアップ
     */
    destroy() {
        document.removeEventListener('keydown', this.handleKeyDown);
        document.removeEventListener('keyup', this.handleKeyUp);
        this.clear();
    }
}

// デフォルトのショートカット設定
export const defaultShortcuts = [
    // ナビゲーション
    ['left', (e) => console.log('Previous frame'), { description: '前のフレーム' }],
    ['right', (e) => console.log('Next frame'), { description: '次のフレーム' }],

    // ファイル操作
    ['s', (e) => console.log('Save'), { ctrl: true, description: '保存' }],
    ['z', (e) => console.log('Undo'), { ctrl: true, description: '元に戻す' }],
    ['y', (e) => console.log('Redo'), { ctrl: true, description: 'やり直し' }],

    // 編集
    ['delete', (e) => console.log('Delete'), { description: '削除' }],
    ['escape', (e) => console.log('Cancel'), { description: 'キャンセル' }],
    ['enter', (e) => console.log('Confirm'), { description: '確定' }],

    // ズーム
    ['+', (e) => console.log('Zoom in'), { description: 'ズームイン' }],
    ['-', (e) => console.log('Zoom out'), { description: 'ズームアウト' }],
    ['0', (e) => console.log('Reset zoom'), { description: 'ズームリセット' }]
];
