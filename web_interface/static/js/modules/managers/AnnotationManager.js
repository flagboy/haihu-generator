/**
 * アノテーション管理クラス
 */
export class AnnotationManager {
    constructor(appState, canvasManager) {
        this.appState = appState;
        this.canvasManager = canvasManager;

        // アノテーションのデフォルト設定
        this.defaultAnnotation = {
            confidence: 1.0,
            area_type: 'hand',
            is_face_up: true,
            is_occluded: false,
            occlusion_ratio: 0.0,
            annotator: 'user',
            notes: ''
        };

        // 表示設定
        this.colors = {
            default: '#007bff',
            selected: '#dc3545',
            hover: '#28a745',
            low_confidence: '#ffc107'
        };
    }

    /**
     * アノテーションを追加
     */
    addAnnotation(bbox, tileId, additionalData = {}) {
        const annotation = {
            id: this.generateId(),
            tile_id: tileId,
            bbox: this.normalizeBbox(bbox),
            ...this.defaultAnnotation,
            ...additionalData,
            created_at: new Date().toISOString()
        };

        this.appState.addAnnotation(annotation);
        return annotation;
    }

    /**
     * アノテーションを更新
     */
    updateAnnotation(id, updates) {
        const index = this.findAnnotationIndex(id);
        if (index !== -1) {
            this.appState.updateAnnotation(index, {
                ...updates,
                updated_at: new Date().toISOString()
            });
            return true;
        }
        return false;
    }

    /**
     * アノテーションを削除
     */
    removeAnnotation(id) {
        const index = this.findAnnotationIndex(id);
        if (index !== -1) {
            this.appState.removeAnnotation(index);
            return true;
        }
        return false;
    }

    /**
     * インデックスでアノテーションを削除
     */
    removeAnnotationByIndex(index) {
        if (index >= 0 && index < this.appState.annotations.length) {
            this.appState.removeAnnotation(index);
            return true;
        }
        return false;
    }

    /**
     * すべてのアノテーションをクリア
     */
    clearAll() {
        this.appState.clearAnnotations();
    }

    /**
     * アノテーションを取得
     */
    getAnnotation(id) {
        return this.appState.annotations.find(ann => ann.id === id);
    }

    /**
     * すべてのアノテーションを取得
     */
    getAllAnnotations() {
        return [...this.appState.annotations];
    }

    /**
     * アノテーションのインデックスを検索
     */
    findAnnotationIndex(id) {
        return this.appState.annotations.findIndex(ann => ann.id === id);
    }

    /**
     * アノテーションを選択
     */
    selectAnnotation(id) {
        this.appState.annotations.forEach((ann, index) => {
            this.appState.updateAnnotation(index, {
                selected: ann.id === id
            });
        });
    }

    /**
     * 選択を解除
     */
    deselectAll() {
        this.appState.annotations.forEach((ann, index) => {
            if (ann.selected) {
                this.appState.updateAnnotation(index, { selected: false });
            }
        });
    }

    /**
     * 選択されているアノテーションを取得
     */
    getSelectedAnnotation() {
        return this.appState.annotations.find(ann => ann.selected);
    }

    /**
     * 点がアノテーション内にあるか判定
     */
    getAnnotationAtPoint(x, y) {
        // 逆順で検索（上に描画されているものを優先）
        for (let i = this.appState.annotations.length - 1; i >= 0; i--) {
            const ann = this.appState.annotations[i];
            if (this.isPointInBbox(x, y, ann.bbox)) {
                return ann;
            }
        }
        return null;
    }

    /**
     * 点がバウンディングボックス内にあるか判定
     */
    isPointInBbox(x, y, bbox) {
        return x >= bbox.x1 && x <= bbox.x2 && y >= bbox.y1 && y <= bbox.y2;
    }

    /**
     * アノテーションを描画
     */
    drawAnnotations() {
        this.appState.annotations.forEach(annotation => {
            const color = this.getAnnotationColor(annotation);
            const lineWidth = annotation.selected ? 3 : 2;

            // バウンディングボックスを描画
            this.canvasManager.drawBoundingBox(annotation.bbox, color, lineWidth);

            // ラベルを描画
            const label = this.getAnnotationLabel(annotation);
            this.canvasManager.drawLabel(annotation.bbox, label, color);
        });
    }

    /**
     * アノテーションの色を取得
     */
    getAnnotationColor(annotation) {
        if (annotation.selected) {
            return this.colors.selected;
        } else if (annotation.confidence < 0.5) {
            return this.colors.low_confidence;
        } else {
            return this.colors.default;
        }
    }

    /**
     * アノテーションのラベルを取得
     */
    getAnnotationLabel(annotation) {
        const tileNames = {
            '1m': '一萬', '2m': '二萬', '3m': '三萬', '4m': '四萬', '5m': '五萬',
            '6m': '六萬', '7m': '七萬', '8m': '八萬', '9m': '九萬',
            '1p': '一筒', '2p': '二筒', '3p': '三筒', '4p': '四筒', '5p': '五筒',
            '6p': '六筒', '7p': '七筒', '8p': '八筒', '9p': '九筒',
            '1s': '一索', '2s': '二索', '3s': '三索', '4s': '四索', '5s': '五索',
            '6s': '六索', '7s': '七索', '8s': '八索', '9s': '九索',
            '1z': '東', '2z': '南', '3z': '西', '4z': '北',
            '5z': '白', '6z': '發', '7z': '中'
        };

        return tileNames[annotation.tile_id] || annotation.tile_id;
    }

    /**
     * IDを生成
     */
    generateId() {
        return `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    /**
     * バウンディングボックスを正規化
     */
    normalizeBbox(bbox) {
        return {
            x1: Math.min(bbox.x1, bbox.x2),
            y1: Math.min(bbox.y1, bbox.y2),
            x2: Math.max(bbox.x1, bbox.x2),
            y2: Math.max(bbox.y1, bbox.y2)
        };
    }

    /**
     * バウンディングボックスの面積を計算
     */
    getBboxArea(bbox) {
        return Math.abs(bbox.x2 - bbox.x1) * Math.abs(bbox.y2 - bbox.y1);
    }

    /**
     * バウンディングボックスの有効性をチェック
     */
    isValidBbox(bbox) {
        const area = this.getBboxArea(bbox);
        return area > 100; // 最小100ピクセル
    }

    /**
     * アノテーションをフィルタリング
     */
    filterAnnotations(predicate) {
        return this.appState.annotations.filter(predicate);
    }

    /**
     * プレイヤー別にアノテーションを取得
     */
    getAnnotationsByPlayer(player) {
        return this.filterAnnotations(ann => ann.player === player);
    }

    /**
     * 信頼度でアノテーションをフィルタリング
     */
    getAnnotationsByConfidence(minConfidence) {
        return this.filterAnnotations(ann => ann.confidence >= minConfidence);
    }

    /**
     * アノテーションの統計情報を取得
     */
    getStatistics() {
        const annotations = this.appState.annotations;
        const stats = {
            total: annotations.length,
            byPlayer: {},
            byTileType: {},
            byConfidence: {
                high: 0,    // >= 0.8
                medium: 0,  // 0.5 - 0.8
                low: 0      // < 0.5
            },
            averageConfidence: 0
        };

        // プレイヤー別集計の初期化
        ['bottom', 'top', 'left', 'right'].forEach(player => {
            stats.byPlayer[player] = 0;
        });

        // 集計
        let totalConfidence = 0;
        annotations.forEach(ann => {
            // プレイヤー別
            if (ann.player && stats.byPlayer[ann.player] !== undefined) {
                stats.byPlayer[ann.player]++;
            }

            // 牌種別
            if (ann.tile_id) {
                stats.byTileType[ann.tile_id] = (stats.byTileType[ann.tile_id] || 0) + 1;
            }

            // 信頼度別
            if (ann.confidence >= 0.8) {
                stats.byConfidence.high++;
            } else if (ann.confidence >= 0.5) {
                stats.byConfidence.medium++;
            } else {
                stats.byConfidence.low++;
            }

            totalConfidence += ann.confidence || 0;
        });

        // 平均信頼度
        stats.averageConfidence = annotations.length > 0
            ? totalConfidence / annotations.length
            : 0;

        return stats;
    }

    /**
     * アノテーションをインポート
     */
    importAnnotations(annotations) {
        this.clearAll();
        annotations.forEach(ann => {
            this.appState.addAnnotation({
                ...ann,
                id: ann.id || this.generateId()
            });
        });
    }

    /**
     * アノテーションをエクスポート
     */
    exportAnnotations(format = 'json') {
        const annotations = this.getAllAnnotations();

        switch (format) {
            case 'json':
                return this.exportAsJSON(annotations);
            case 'coco':
                return this.exportAsCOCO(annotations);
            case 'yolo':
                return this.exportAsYOLO(annotations);
            default:
                throw new Error(`Unsupported export format: ${format}`);
        }
    }

    /**
     * JSON形式でエクスポート
     */
    exportAsJSON(annotations) {
        return {
            version: '1.0',
            annotations: annotations,
            metadata: {
                exported_at: new Date().toISOString(),
                total_count: annotations.length,
                statistics: this.getStatistics()
            }
        };
    }

    /**
     * COCO形式でエクスポート（スタブ）
     */
    exportAsCOCO(annotations) {
        // TODO: COCO形式への変換実装
        console.warn('COCO export not yet implemented');
        return null;
    }

    /**
     * YOLO形式でエクスポート（スタブ）
     */
    exportAsYOLO(annotations) {
        // TODO: YOLO形式への変換実装
        console.warn('YOLO export not yet implemented');
        return null;
    }

    /**
     * アノテーションの重複をチェック
     */
    checkOverlaps() {
        const overlaps = [];
        const annotations = this.appState.annotations;

        for (let i = 0; i < annotations.length - 1; i++) {
            for (let j = i + 1; j < annotations.length; j++) {
                if (this.bboxesOverlap(annotations[i].bbox, annotations[j].bbox)) {
                    overlaps.push({
                        annotation1: annotations[i],
                        annotation2: annotations[j],
                        overlapRatio: this.calculateOverlapRatio(
                            annotations[i].bbox,
                            annotations[j].bbox
                        )
                    });
                }
            }
        }

        return overlaps;
    }

    /**
     * バウンディングボックスが重なっているか判定
     */
    bboxesOverlap(bbox1, bbox2) {
        return !(bbox1.x2 < bbox2.x1 ||
                 bbox2.x2 < bbox1.x1 ||
                 bbox1.y2 < bbox2.y1 ||
                 bbox2.y2 < bbox1.y1);
    }

    /**
     * 重なり率を計算
     */
    calculateOverlapRatio(bbox1, bbox2) {
        const x1 = Math.max(bbox1.x1, bbox2.x1);
        const y1 = Math.max(bbox1.y1, bbox2.y1);
        const x2 = Math.min(bbox1.x2, bbox2.x2);
        const y2 = Math.min(bbox1.y2, bbox2.y2);

        if (x2 < x1 || y2 < y1) return 0;

        const overlapArea = (x2 - x1) * (y2 - y1);
        const area1 = this.getBboxArea(bbox1);
        const area2 = this.getBboxArea(bbox2);

        return overlapArea / Math.min(area1, area2);
    }
}
