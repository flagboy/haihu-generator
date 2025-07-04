"""
学習スケジューラーシステム

学習スケジュール、ハイパーパラメータ最適化、自動調整を行う
"""

import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from ...utils.config import ConfigManager
from ...utils.logger import LoggerMixin


@dataclass
class HyperParameter:
    """ハイパーパラメータ定義"""

    name: str
    param_type: str  # "float", "int", "choice"
    min_value: float | None = None
    max_value: float | None = None
    choices: list[Any] | None = None
    log_scale: bool = False
    current_value: Any = None


@dataclass
class OptimizationTrial:
    """最適化試行"""

    trial_id: str
    parameters: dict[str, Any]
    score: float | None = None
    status: str = "pending"  # "pending", "running", "completed", "failed"
    start_time: datetime | None = None
    end_time: datetime | None = None
    metrics: dict[str, float] | None = None
    notes: str = ""


@dataclass
class ScheduleTask:
    """スケジュールタスク"""

    task_id: str
    task_type: str  # "training", "evaluation", "optimization"
    config: dict[str, Any]
    scheduled_time: datetime
    priority: int = 1
    status: str = "scheduled"  # "scheduled", "running", "completed", "failed", "cancelled"
    dependencies: list[str] = None
    retry_count: int = 0
    max_retries: int = 3


class LearningScheduler(LoggerMixin):
    """学習スケジューラークラス"""

    def __init__(self, config_manager: ConfigManager | None = None):
        """
        初期化

        Args:
            config_manager: 設定管理インスタンス
        """
        self.config_manager = config_manager or ConfigManager()
        self.config = self.config_manager.get_config()

        # スケジューラー設定
        self.scheduler_root = Path(
            self.config.get("training", {}).get("scheduler_root", "data/training/scheduler")
        )
        self.scheduler_root.mkdir(parents=True, exist_ok=True)

        # ファイルパス
        self.schedules_file = self.scheduler_root / "schedules.json"
        self.trials_file = self.scheduler_root / "optimization_trials.json"
        self.hyperparams_file = self.scheduler_root / "hyperparameters.json"

        # 状態管理
        self.scheduled_tasks: dict[str, ScheduleTask] = {}
        self.optimization_trials: dict[str, OptimizationTrial] = {}
        self.hyperparameters: dict[str, HyperParameter] = {}

        # デフォルトハイパーパラメータを設定
        self._setup_default_hyperparameters()

        # 既存データを読み込み
        self._load_data()

        self.logger.info(f"LearningScheduler初期化完了: {self.scheduler_root}")

    def _setup_default_hyperparameters(self):
        """デフォルトハイパーパラメータを設定"""
        default_params = [
            HyperParameter("learning_rate", "float", 1e-5, 1e-1, log_scale=True),
            HyperParameter("batch_size", "choice", choices=[8, 16, 32, 64, 128]),
            HyperParameter("epochs", "int", 10, 200),
            HyperParameter("optimizer_type", "choice", choices=["adam", "sgd", "adamw"]),
            HyperParameter(
                "lr_scheduler_type", "choice", choices=["plateau", "step", "cosine", "none"]
            ),
            HyperParameter("dropout_rate", "float", 0.0, 0.8),
            HyperParameter("weight_decay", "float", 1e-6, 1e-2, log_scale=True),
            HyperParameter("early_stopping_patience", "int", 5, 50),
        ]

        for param in default_params:
            self.hyperparameters[param.name] = param

    def _load_data(self):
        """データを読み込み"""
        # スケジュールタスクを読み込み
        if self.schedules_file.exists():
            try:
                with open(self.schedules_file, encoding="utf-8") as f:
                    data = json.load(f)
                    for task_data in data.get("tasks", []):
                        task = ScheduleTask(
                            task_id=task_data["task_id"],
                            task_type=task_data["task_type"],
                            config=task_data["config"],
                            scheduled_time=datetime.fromisoformat(task_data["scheduled_time"]),
                            priority=task_data.get("priority", 1),
                            status=task_data.get("status", "scheduled"),
                            dependencies=task_data.get("dependencies", []),
                            retry_count=task_data.get("retry_count", 0),
                            max_retries=task_data.get("max_retries", 3),
                        )
                        self.scheduled_tasks[task.task_id] = task
            except Exception as e:
                self.logger.warning(f"スケジュールデータの読み込みに失敗: {e}")

        # 最適化試行を読み込み
        if self.trials_file.exists():
            try:
                with open(self.trials_file, encoding="utf-8") as f:
                    data = json.load(f)
                    for trial_data in data.get("trials", []):
                        trial = OptimizationTrial(
                            trial_id=trial_data["trial_id"],
                            parameters=trial_data["parameters"],
                            score=trial_data.get("score"),
                            status=trial_data.get("status", "pending"),
                            start_time=datetime.fromisoformat(trial_data["start_time"])
                            if trial_data.get("start_time")
                            else None,
                            end_time=datetime.fromisoformat(trial_data["end_time"])
                            if trial_data.get("end_time")
                            else None,
                            metrics=trial_data.get("metrics"),
                            notes=trial_data.get("notes", ""),
                        )
                        self.optimization_trials[trial.trial_id] = trial
            except Exception as e:
                self.logger.warning(f"最適化試行データの読み込みに失敗: {e}")

        # ハイパーパラメータを読み込み
        if self.hyperparams_file.exists():
            try:
                with open(self.hyperparams_file, encoding="utf-8") as f:
                    data = json.load(f)
                    for param_data in data.get("parameters", []):
                        param = HyperParameter(
                            name=param_data["name"],
                            param_type=param_data["param_type"],
                            min_value=param_data.get("min_value"),
                            max_value=param_data.get("max_value"),
                            choices=param_data.get("choices"),
                            log_scale=param_data.get("log_scale", False),
                            current_value=param_data.get("current_value"),
                        )
                        self.hyperparameters[param.name] = param
            except Exception as e:
                self.logger.warning(f"ハイパーパラメータデータの読み込みに失敗: {e}")

    def _save_data(self):
        """データを保存"""
        try:
            # スケジュールタスクを保存
            schedules_data = {"tasks": [], "last_updated": datetime.now().isoformat()}
            for task in self.scheduled_tasks.values():
                task_dict = asdict(task)
                task_dict["scheduled_time"] = task.scheduled_time.isoformat()
                schedules_data["tasks"].append(task_dict)

            with open(self.schedules_file, "w", encoding="utf-8") as f:
                json.dump(schedules_data, f, ensure_ascii=False, indent=2, default=str)

            # 最適化試行を保存
            trials_data = {"trials": [], "last_updated": datetime.now().isoformat()}
            for trial in self.optimization_trials.values():
                trial_dict = asdict(trial)
                if trial.start_time:
                    trial_dict["start_time"] = trial.start_time.isoformat()
                if trial.end_time:
                    trial_dict["end_time"] = trial.end_time.isoformat()
                trials_data["trials"].append(trial_dict)

            with open(self.trials_file, "w", encoding="utf-8") as f:
                json.dump(trials_data, f, ensure_ascii=False, indent=2, default=str)

            # ハイパーパラメータを保存
            hyperparams_data = {"parameters": [], "last_updated": datetime.now().isoformat()}
            for param in self.hyperparameters.values():
                hyperparams_data["parameters"].append(asdict(param))

            with open(self.hyperparams_file, "w", encoding="utf-8") as f:
                json.dump(hyperparams_data, f, ensure_ascii=False, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"データ保存に失敗: {e}")

    def schedule_training(
        self,
        config: dict[str, Any],
        scheduled_time: datetime | None = None,
        priority: int = 1,
        dependencies: list[str] | None = None,
    ) -> str:
        """
        学習をスケジュール

        Args:
            config: 学習設定
            scheduled_time: 実行予定時刻
            priority: 優先度
            dependencies: 依存タスクID

        Returns:
            タスクID
        """
        import uuid

        task_id = str(uuid.uuid4())

        if scheduled_time is None:
            scheduled_time = datetime.now()

        task = ScheduleTask(
            task_id=task_id,
            task_type="training",
            config=config,
            scheduled_time=scheduled_time,
            priority=priority,
            dependencies=dependencies or [],
        )

        self.scheduled_tasks[task_id] = task
        self._save_data()

        self.logger.info(f"学習スケジュール登録: {task_id}, 実行予定: {scheduled_time}")
        return task_id

    def schedule_optimization(
        self,
        base_config: dict[str, Any],
        optimization_config: dict[str, Any],
        scheduled_time: datetime | None = None,
    ) -> str:
        """
        ハイパーパラメータ最適化をスケジュール

        Args:
            base_config: ベース設定
            optimization_config: 最適化設定
            scheduled_time: 実行予定時刻

        Returns:
            タスクID
        """
        import uuid

        task_id = str(uuid.uuid4())

        if scheduled_time is None:
            scheduled_time = datetime.now()

        config = {"base_config": base_config, "optimization_config": optimization_config}

        task = ScheduleTask(
            task_id=task_id,
            task_type="optimization",
            config=config,
            scheduled_time=scheduled_time,
            priority=2,  # 最適化は高優先度
        )

        self.scheduled_tasks[task_id] = task
        self._save_data()

        self.logger.info(f"最適化スケジュール登録: {task_id}, 実行予定: {scheduled_time}")
        return task_id

    def get_next_task(self) -> ScheduleTask | None:
        """
        次に実行すべきタスクを取得

        Returns:
            次のタスク
        """
        now = datetime.now()
        ready_tasks = []

        for task in self.scheduled_tasks.values():
            if task.status != "scheduled":
                continue

            # 実行時刻チェック
            if task.scheduled_time > now:
                continue

            # 依存関係チェック
            dependencies_met = True
            for dep_id in task.dependencies:
                dep_task = self.scheduled_tasks.get(dep_id)
                if not dep_task or dep_task.status != "completed":
                    dependencies_met = False
                    break

            if dependencies_met:
                ready_tasks.append(task)

        if not ready_tasks:
            return None

        # 優先度でソート
        ready_tasks.sort(key=lambda x: (-x.priority, x.scheduled_time))
        return ready_tasks[0]

    def start_hyperparameter_optimization(
        self, base_config: dict[str, Any], optimization_config: dict[str, Any]
    ) -> list[str]:
        """
        ハイパーパラメータ最適化を開始

        Args:
            base_config: ベース設定
            optimization_config: 最適化設定

        Returns:
            試行IDのリスト
        """
        method = optimization_config.get("method", "random")
        n_trials = optimization_config.get("n_trials", 10)
        target_metric = optimization_config.get("target_metric", "val_accuracy")

        trial_ids = []

        if method == "random":
            trial_ids = self._random_search(base_config, n_trials, target_metric)
        elif method == "grid":
            trial_ids = self._grid_search(base_config, optimization_config, target_metric)
        elif method == "bayesian":
            trial_ids = self._bayesian_optimization(base_config, n_trials, target_metric)
        else:
            self.logger.error(f"サポートされていない最適化手法: {method}")
            return []

        self._save_data()
        self.logger.info(f"ハイパーパラメータ最適化開始: {len(trial_ids)}試行")
        return trial_ids

    def _random_search(
        self, base_config: dict[str, Any], n_trials: int, target_metric: str
    ) -> list[str]:
        """ランダムサーチ"""
        trial_ids = []

        for _i in range(n_trials):
            import uuid

            trial_id = str(uuid.uuid4())

            # ランダムパラメータを生成
            parameters = self._sample_random_parameters()

            trial = OptimizationTrial(trial_id=trial_id, parameters=parameters, status="pending")

            self.optimization_trials[trial_id] = trial
            trial_ids.append(trial_id)

        return trial_ids

    def _grid_search(
        self, base_config: dict[str, Any], optimization_config: dict[str, Any], target_metric: str
    ) -> list[str]:
        """グリッドサーチ"""
        # グリッドパラメータを取得
        grid_params = optimization_config.get("grid_parameters", {})

        # 全組み合わせを生成
        import itertools

        param_names = list(grid_params.keys())
        param_values = [grid_params[name] for name in param_names]

        trial_ids = []
        for combination in itertools.product(*param_values):
            import uuid

            trial_id = str(uuid.uuid4())

            parameters = dict(zip(param_names, combination, strict=False))

            trial = OptimizationTrial(trial_id=trial_id, parameters=parameters, status="pending")

            self.optimization_trials[trial_id] = trial
            trial_ids.append(trial_id)

        return trial_ids

    def _bayesian_optimization(
        self, base_config: dict[str, Any], n_trials: int, target_metric: str
    ) -> list[str]:
        """ベイジアン最適化（簡易実装）"""
        # 実際の実装では scikit-optimize などを使用
        # ここでは簡易的にランダムサーチで代替
        self.logger.warning("ベイジアン最適化は簡易実装です（ランダムサーチで代替）")
        return self._random_search(base_config, n_trials, target_metric)

    def _sample_random_parameters(self) -> dict[str, Any]:
        """ランダムパラメータをサンプリング"""
        parameters = {}

        for param_name, param in self.hyperparameters.items():
            if param.param_type == "float":
                if param.log_scale:
                    value = np.exp(
                        np.random.uniform(np.log(param.min_value), np.log(param.max_value))
                    )
                else:
                    value = np.random.uniform(param.min_value, param.max_value)
                parameters[param_name] = float(value)

            elif param.param_type == "int":
                value = np.random.randint(param.min_value, param.max_value + 1)
                parameters[param_name] = int(value)

            elif param.param_type == "choice":
                value = np.random.choice(param.choices)
                parameters[param_name] = value

        return parameters

    def update_trial_result(
        self, trial_id: str, score: float, metrics: dict[str, float] | None = None
    ):
        """
        試行結果を更新

        Args:
            trial_id: 試行ID
            score: スコア
            metrics: 詳細メトリクス
        """
        if trial_id not in self.optimization_trials:
            self.logger.error(f"試行が見つかりません: {trial_id}")
            return

        trial = self.optimization_trials[trial_id]
        trial.score = score
        trial.metrics = metrics or {}
        trial.status = "completed"
        trial.end_time = datetime.now()

        self._save_data()
        self.logger.info(f"試行結果更新: {trial_id}, スコア: {score}")

    def get_best_parameters(self, n_best: int = 1) -> list[dict[str, Any]]:
        """
        最良のパラメータを取得

        Args:
            n_best: 取得する上位数

        Returns:
            最良パラメータのリスト
        """
        completed_trials = [
            trial
            for trial in self.optimization_trials.values()
            if trial.status == "completed" and trial.score is not None
        ]

        if not completed_trials:
            return []

        # スコアでソート（降順）
        completed_trials.sort(key=lambda x: x.score, reverse=True)

        best_trials = completed_trials[:n_best]
        return [
            {
                "trial_id": trial.trial_id,
                "parameters": trial.parameters,
                "score": trial.score,
                "metrics": trial.metrics,
            }
            for trial in best_trials
        ]

    def suggest_next_parameters(self, base_config: dict[str, Any]) -> dict[str, Any]:
        """
        次のパラメータを提案

        Args:
            base_config: ベース設定

        Returns:
            提案パラメータ
        """
        # 過去の結果を分析して提案（簡易実装）
        best_params = self.get_best_parameters(n_best=3)

        if not best_params:
            # 初回の場合はランダム
            return self._sample_random_parameters()

        # 最良パラメータの周辺を探索
        best_trial = best_params[0]
        suggested_params = best_trial["parameters"].copy()

        # 一部パラメータをランダムに変更
        param_names = list(suggested_params.keys())
        n_change = max(1, len(param_names) // 3)
        change_params = random.sample(param_names, n_change)

        for param_name in change_params:
            if param_name in self.hyperparameters:
                param = self.hyperparameters[param_name]

                if param.param_type == "float":
                    current_value = suggested_params[param_name]
                    # 現在値の±20%の範囲で変更
                    noise_factor = 0.2
                    min_val = max(param.min_value, current_value * (1 - noise_factor))
                    max_val = min(param.max_value, current_value * (1 + noise_factor))

                    if param.log_scale:
                        suggested_params[param_name] = np.exp(
                            np.random.uniform(np.log(min_val), np.log(max_val))
                        )
                    else:
                        suggested_params[param_name] = np.random.uniform(min_val, max_val)

                elif param.param_type == "choice":
                    suggested_params[param_name] = np.random.choice(param.choices)

        return suggested_params

    def create_adaptive_schedule(
        self, base_config: dict[str, Any], performance_threshold: float = 0.8
    ) -> list[str]:
        """
        適応的スケジュールを作成

        Args:
            base_config: ベース設定
            performance_threshold: 性能閾値

        Returns:
            スケジュールされたタスクIDのリスト
        """
        task_ids = []

        # 段階的学習スケジュール
        stages = [
            {"epochs": 20, "lr": 0.01, "description": "初期学習"},
            {"epochs": 30, "lr": 0.001, "description": "中間学習"},
            {"epochs": 50, "lr": 0.0001, "description": "ファインチューニング"},
        ]

        current_time = datetime.now()

        for i, stage in enumerate(stages):
            config = base_config.copy()
            config.update(stage)

            # 前のステージの完了を依存関係に設定
            dependencies = [task_ids[-1]] if task_ids else []

            # 実行時刻を設定（前のステージから1時間後）
            scheduled_time = current_time + timedelta(hours=i)

            task_id = self.schedule_training(
                config=config, scheduled_time=scheduled_time, priority=1, dependencies=dependencies
            )

            task_ids.append(task_id)

        self.logger.info(f"適応的スケジュール作成: {len(task_ids)}ステージ")
        return task_ids

    def get_optimization_summary(self) -> dict[str, Any]:
        """最適化サマリーを取得"""
        completed_trials = [
            trial
            for trial in self.optimization_trials.values()
            if trial.status == "completed" and trial.score is not None
        ]

        if not completed_trials:
            return {"message": "完了した試行がありません"}

        scores = [trial.score for trial in completed_trials if trial.score is not None]

        summary = {
            "total_trials": len(self.optimization_trials),
            "completed_trials": len(completed_trials),
            "best_score": max(scores) if scores else None,
            "worst_score": min(scores) if scores else None,
            "mean_score": np.mean(scores) if scores else None,
            "std_score": np.std(scores) if scores else None,
            "best_parameters": self.get_best_parameters(n_best=1)[0] if completed_trials else None,
        }

        return summary

    def cleanup_old_trials(self, keep_count: int = 100) -> bool:
        """
        古い試行をクリーンアップ

        Args:
            keep_count: 保持する試行数

        Returns:
            クリーンアップ成功かどうか
        """
        try:
            if len(self.optimization_trials) <= keep_count:
                return True

            # 完了時刻でソート
            trials = list(self.optimization_trials.values())
            trials.sort(key=lambda x: x.end_time or datetime.min)

            # 古い試行を削除
            trials_to_remove = trials[:-keep_count]

            for trial in trials_to_remove:
                del self.optimization_trials[trial.trial_id]

            self._save_data()
            self.logger.info(f"古い試行をクリーンアップ: {len(trials_to_remove)}個削除")
            return True

        except Exception as e:
            self.logger.error(f"試行クリーンアップに失敗: {e}")
            return False
