"""
モニタリングシステムの設定

モニタリングシステムの動作を制御する設定
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MonitoringConfig:
    """モニタリング設定"""

    # ログ設定
    log_dir: Path = field(default_factory=lambda: Path("logs/monitoring"))
    log_level: str = "INFO"
    enable_console_logging: bool = True
    enable_file_logging: bool = True
    log_rotation: str = "1 day"
    log_retention: str = "30 days"

    # メトリクス設定
    metrics_flush_interval: int = 60  # 秒
    metrics_max_history: int = 10000
    metrics_export_interval: int = 3600  # 1時間

    # エラー追跡設定
    error_alert_threshold: int = 10
    error_alert_window: int = 300  # 5分
    error_max_history: int = 10000
    error_retention_days: int = 7

    # システムモニタリング設定
    system_monitor_interval: int = 60  # 秒
    cpu_alert_threshold: float = 90.0
    memory_alert_threshold: float = 90.0
    disk_alert_threshold: float = 90.0
    gpu_memory_alert_threshold: float = 95.0

    # ダッシュボード設定
    dashboard_update_interval: int = 5  # 秒
    dashboard_max_data_points: int = 60

    # パフォーマンス設定
    performance_tracking_enabled: bool = True
    batch_tracking_enabled: bool = True
    memory_tracking_enabled: bool = True

    @classmethod
    def from_dict(cls, config_dict: dict) -> "MonitoringConfig":
        """辞書から設定を作成"""
        config = cls()

        for key, value in config_dict.items():
            if hasattr(config, key):
                if key == "log_dir":
                    setattr(config, key, Path(value))
                else:
                    setattr(config, key, value)

        return config

    def to_dict(self) -> dict:
        """設定を辞書に変換"""
        return {
            "log_dir": str(self.log_dir),
            "log_level": self.log_level,
            "enable_console_logging": self.enable_console_logging,
            "enable_file_logging": self.enable_file_logging,
            "log_rotation": self.log_rotation,
            "log_retention": self.log_retention,
            "metrics_flush_interval": self.metrics_flush_interval,
            "metrics_max_history": self.metrics_max_history,
            "metrics_export_interval": self.metrics_export_interval,
            "error_alert_threshold": self.error_alert_threshold,
            "error_alert_window": self.error_alert_window,
            "error_max_history": self.error_max_history,
            "error_retention_days": self.error_retention_days,
            "system_monitor_interval": self.system_monitor_interval,
            "cpu_alert_threshold": self.cpu_alert_threshold,
            "memory_alert_threshold": self.memory_alert_threshold,
            "disk_alert_threshold": self.disk_alert_threshold,
            "gpu_memory_alert_threshold": self.gpu_memory_alert_threshold,
            "dashboard_update_interval": self.dashboard_update_interval,
            "dashboard_max_data_points": self.dashboard_max_data_points,
            "performance_tracking_enabled": self.performance_tracking_enabled,
            "batch_tracking_enabled": self.batch_tracking_enabled,
            "memory_tracking_enabled": self.memory_tracking_enabled,
        }


# グローバル設定インスタンス
_config: MonitoringConfig | None = None


def get_monitoring_config() -> MonitoringConfig:
    """モニタリング設定を取得"""
    global _config
    if _config is None:
        _config = MonitoringConfig()
    return _config


def set_monitoring_config(config: MonitoringConfig) -> None:
    """モニタリング設定を設定"""
    global _config
    _config = config


def load_monitoring_config(config_path: Path) -> MonitoringConfig:
    """設定ファイルから読み込み"""
    import yaml

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    monitoring_config = config_dict.get("monitoring", {})
    config = MonitoringConfig.from_dict(monitoring_config)

    set_monitoring_config(config)
    return config
