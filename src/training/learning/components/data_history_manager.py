"""
データ履歴管理コンポーネント

学習に使用したデータの履歴を管理し、
継続学習時のデータ選択を支援する
"""

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ....utils.logger import LoggerMixin


@dataclass
class DataSample:
    """データサンプル情報"""

    sample_id: str
    video_id: str
    frame_id: str
    tile_count: int
    quality_score: float
    timestamp: datetime
    labels: list[str] = field(default_factory=list)
    features: dict[str, float] = field(default_factory=dict)
    usage_count: int = 0
    last_used: datetime | None = None


@dataclass
class DataHistoryEntry:
    """データ履歴エントリ"""

    session_id: str
    dataset_version: str
    sample_ids: list[str]
    timestamp: datetime
    performance_metrics: dict[str, float]
    strategy: str
    notes: str = ""


class DataHistoryManager(LoggerMixin):
    """データ履歴管理クラス"""

    def __init__(self, db_path: str = "data/training/data_history.db"):
        """
        初期化

        Args:
            db_path: データベースパス
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # データベース初期化
        self._init_database()

        # キャッシュ
        self._sample_cache = {}
        self._importance_scores = {}

    def _init_database(self):
        """データベースを初期化"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # サンプルテーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS samples (
                    sample_id TEXT PRIMARY KEY,
                    video_id TEXT NOT NULL,
                    frame_id TEXT NOT NULL,
                    tile_count INTEGER,
                    quality_score REAL,
                    timestamp TEXT,
                    labels TEXT,
                    features TEXT,
                    usage_count INTEGER DEFAULT 0,
                    last_used TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 履歴テーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    dataset_version TEXT,
                    sample_ids TEXT,
                    timestamp TEXT,
                    performance_metrics TEXT,
                    strategy TEXT,
                    notes TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # サンプル使用履歴テーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sample_usage (
                    usage_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sample_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    timestamp TEXT,
                    performance_impact REAL,
                    FOREIGN KEY (sample_id) REFERENCES samples(sample_id)
                )
            """)

            # インデックス作成
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_samples_video ON samples(video_id)")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_samples_quality ON samples(quality_score)"
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_session ON history(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_usage_sample ON sample_usage(sample_id)")

            conn.commit()

    def add_samples(self, samples: list[DataSample]):
        """
        サンプルを追加

        Args:
            samples: データサンプルのリスト
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            for sample in samples:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO samples
                    (sample_id, video_id, frame_id, tile_count, quality_score,
                     timestamp, labels, features, usage_count, last_used)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        sample.sample_id,
                        sample.video_id,
                        sample.frame_id,
                        sample.tile_count,
                        sample.quality_score,
                        sample.timestamp.isoformat() if sample.timestamp else None,
                        json.dumps(sample.labels),
                        json.dumps(sample.features),
                        sample.usage_count,
                        sample.last_used.isoformat() if sample.last_used else None,
                    ),
                )

            conn.commit()

        self.logger.info(f"{len(samples)}件のサンプルを追加しました")

    def record_usage(
        self,
        session_id: str,
        dataset_version: str,
        sample_ids: list[str],
        performance_metrics: dict[str, float],
        strategy: str,
        notes: str = "",
    ):
        """
        データ使用履歴を記録

        Args:
            session_id: セッションID
            dataset_version: データセットバージョン
            sample_ids: 使用したサンプルID
            performance_metrics: 性能メトリクス
            strategy: 学習戦略
            notes: メモ
        """
        timestamp = datetime.now()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # 履歴を記録
            cursor.execute(
                """
                INSERT INTO history
                (session_id, dataset_version, sample_ids, timestamp,
                 performance_metrics, strategy, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    session_id,
                    dataset_version,
                    json.dumps(sample_ids),
                    timestamp.isoformat(),
                    json.dumps(performance_metrics),
                    strategy,
                    notes,
                ),
            )

            # サンプル使用回数を更新
            for sample_id in sample_ids:
                cursor.execute(
                    """
                    UPDATE samples
                    SET usage_count = usage_count + 1,
                        last_used = ?
                    WHERE sample_id = ?
                """,
                    (timestamp.isoformat(), sample_id),
                )

                # 使用履歴を記録
                cursor.execute(
                    """
                    INSERT INTO sample_usage
                    (sample_id, session_id, timestamp, performance_impact)
                    VALUES (?, ?, ?, ?)
                """,
                    (
                        sample_id,
                        session_id,
                        timestamp.isoformat(),
                        performance_metrics.get("accuracy", 0.0),
                    ),
                )

            conn.commit()

    def get_sample_importance_scores(
        self, sample_ids: list[str], method: str = "frequency"
    ) -> dict[str, float]:
        """
        サンプルの重要度スコアを計算

        Args:
            sample_ids: サンプルID
            method: 計算方法（"frequency", "performance", "diversity"）

        Returns:
            サンプルIDと重要度スコアのマッピング
        """
        scores = {}

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            for sample_id in sample_ids:
                if method == "frequency":
                    # 使用頻度ベース
                    cursor.execute(
                        "SELECT usage_count FROM samples WHERE sample_id = ?", (sample_id,)
                    )
                    result = cursor.fetchone()
                    scores[sample_id] = 1.0 / (1 + (result[0] if result else 0))

                elif method == "performance":
                    # 性能影響ベース
                    cursor.execute(
                        """
                        SELECT AVG(performance_impact)
                        FROM sample_usage
                        WHERE sample_id = ?
                    """,
                        (sample_id,),
                    )
                    result = cursor.fetchone()
                    scores[sample_id] = result[0] if result and result[0] else 0.5

                elif method == "diversity":
                    # 多様性ベース（特徴量の分散）
                    cursor.execute("SELECT features FROM samples WHERE sample_id = ?", (sample_id,))
                    result = cursor.fetchone()
                    if result:
                        features = json.loads(result[0])
                        # 特徴量の分散を計算
                        if features:
                            values = list(features.values())
                            scores[sample_id] = np.std(values) if values else 0.0
                    else:
                        scores[sample_id] = 0.0

        # 正規化
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {k: v / max_score for k, v in scores.items()}

        return scores

    def select_rehearsal_samples(
        self,
        available_samples: list[str],
        num_samples: int,
        selection_strategy: str = "importance_sampling",
    ) -> list[str]:
        """
        リハーサル用サンプルを選択

        Args:
            available_samples: 利用可能なサンプル
            num_samples: 選択するサンプル数
            selection_strategy: 選択戦略

        Returns:
            選択されたサンプルID
        """
        if len(available_samples) <= num_samples:
            return available_samples

        if selection_strategy == "random":
            # ランダム選択
            import random

            return random.sample(available_samples, num_samples)

        elif selection_strategy == "importance_sampling":
            # 重要度サンプリング
            importance_scores = self.get_sample_importance_scores(
                available_samples, method="performance"
            )

            # 重要度に基づいて確率的に選択
            samples = list(importance_scores.keys())
            probabilities = np.array(list(importance_scores.values()))
            probabilities = probabilities / probabilities.sum()

            selected_indices = np.random.choice(
                len(samples), size=num_samples, replace=False, p=probabilities
            )

            return [samples[i] for i in selected_indices]

        elif selection_strategy == "diversity":
            # 多様性を重視した選択
            selected = []
            remaining = available_samples.copy()

            # 最初のサンプルをランダムに選択
            import random

            first = random.choice(remaining)
            selected.append(first)
            remaining.remove(first)

            # 残りは既選択サンプルとの距離が最大のものを選択
            while len(selected) < num_samples and remaining:
                # 簡易的な実装（実際は特徴量ベースの距離計算が必要）
                next_sample = random.choice(remaining)
                selected.append(next_sample)
                remaining.remove(next_sample)

            return selected

        else:
            raise ValueError(f"未対応の選択戦略: {selection_strategy}")

    def get_performance_trend(
        self, session_id: str | None = None, last_n_entries: int = 10
    ) -> list[dict[str, Any]]:
        """
        性能トレンドを取得

        Args:
            session_id: セッションID（Noneの場合は全体）
            last_n_entries: 取得するエントリ数

        Returns:
            性能履歴
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            if session_id:
                cursor.execute(
                    """
                    SELECT timestamp, performance_metrics, strategy
                    FROM history
                    WHERE session_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """,
                    (session_id, last_n_entries),
                )
            else:
                cursor.execute(
                    """
                    SELECT timestamp, performance_metrics, strategy, session_id
                    FROM history
                    ORDER BY timestamp DESC
                    LIMIT ?
                """,
                    (last_n_entries,),
                )

            results = cursor.fetchall()

        trend = []
        for row in results:
            entry = {"timestamp": row[0], "metrics": json.loads(row[1]), "strategy": row[2]}
            if not session_id:
                entry["session_id"] = row[3]
            trend.append(entry)

        return trend

    def cleanup_old_data(self, days_to_keep: int = 90):
        """
        古いデータをクリーンアップ

        Args:
            days_to_keep: 保持する日数
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # 古い履歴を削除
            cursor.execute(
                """
                DELETE FROM history
                WHERE julianday('now') - julianday(created_at) > ?
            """,
                (days_to_keep,),
            )

            # 使用されていない古いサンプルを削除
            cursor.execute(
                """
                DELETE FROM samples
                WHERE usage_count = 0
                AND julianday('now') - julianday(created_at) > ?
            """,
                (days_to_keep,),
            )

            deleted_history = cursor.rowcount

            conn.commit()

        self.logger.info(f"古いデータをクリーンアップ: {deleted_history}件の履歴を削除")
