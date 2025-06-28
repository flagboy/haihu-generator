"""
モニタリングダッシュボード

Webベースのリアルタイムモニタリングダッシュボード
"""

from datetime import datetime
from typing import Any

from flask import Blueprint, jsonify, render_template_string, request

from .error_tracker import error_tracker
from .metrics import global_metrics
from .system_monitor import system_monitor

# ダッシュボードのBlueprint
dashboard_bp = Blueprint("monitoring_dashboard", __name__, url_prefix="/monitoring")


# HTMLテンプレート
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>システムモニタリングダッシュボード</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            color: #333;
            margin-bottom: 30px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .card h2 {
            margin-top: 0;
            color: #555;
            font-size: 18px;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 4px;
        }
        .metric-label {
            color: #666;
        }
        .metric-value {
            font-weight: bold;
            color: #333;
        }
        .alert {
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
            color: white;
        }
        .alert-warning {
            background-color: #f39c12;
        }
        .alert-critical {
            background-color: #e74c3c;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-healthy {
            background-color: #27ae60;
        }
        .status-unhealthy {
            background-color: #e74c3c;
        }
        .chart-container {
            position: relative;
            height: 300px;
            margin-top: 20px;
        }
        .error-list {
            max-height: 400px;
            overflow-y: auto;
        }
        .error-item {
            padding: 10px;
            margin: 5px 0;
            background: #f8f9fa;
            border-radius: 4px;
            border-left: 4px solid #e74c3c;
        }
        .timestamp {
            color: #999;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 麻雀牌譜作成システム - モニタリングダッシュボード</h1>

        <div class="grid">
            <!-- システム状態 -->
            <div class="card">
                <h2>システム状態</h2>
                <div id="system-status">
                    <div class="metric">
                        <span class="metric-label">ステータス</span>
                        <span class="metric-value">
                            <span class="status-indicator status-healthy"></span>
                            <span id="health-status">正常</span>
                        </span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">CPU使用率</span>
                        <span class="metric-value" id="cpu-percent">0%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">メモリ使用率</span>
                        <span class="metric-value" id="memory-percent">0%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">ディスク使用率</span>
                        <span class="metric-value" id="disk-percent">0%</span>
                    </div>
                    <div class="metric" id="gpu-metric" style="display: none;">
                        <span class="metric-label">GPU使用率</span>
                        <span class="metric-value" id="gpu-percent">0%</span>
                    </div>
                </div>
            </div>

            <!-- パフォーマンスメトリクス -->
            <div class="card">
                <h2>パフォーマンスメトリクス</h2>
                <div id="performance-metrics">
                    <div class="metric">
                        <span class="metric-label">処理速度 (FPS)</span>
                        <span class="metric-value" id="processing-fps">0</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">バッチ処理成功率</span>
                        <span class="metric-value" id="batch-success-rate">0%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">平均処理時間</span>
                        <span class="metric-value" id="avg-processing-time">0ms</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">スループット</span>
                        <span class="metric-value" id="throughput">0/秒</span>
                    </div>
                </div>
            </div>

            <!-- エラー統計 -->
            <div class="card">
                <h2>エラー統計（過去1時間）</h2>
                <div id="error-stats">
                    <div class="metric">
                        <span class="metric-label">総エラー数</span>
                        <span class="metric-value" id="total-errors">0</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">エラーレート</span>
                        <span class="metric-value" id="error-rate">0/分</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">クリティカルエラー</span>
                        <span class="metric-value" id="critical-errors">0</span>
                    </div>
                </div>
                <div id="recent-errors" class="error-list"></div>
            </div>

            <!-- アラート -->
            <div class="card">
                <h2>アクティブアラート</h2>
                <div id="alerts"></div>
            </div>
        </div>

        <!-- チャート -->
        <div class="grid">
            <div class="card">
                <h2>システムリソース推移</h2>
                <div class="chart-container">
                    <canvas id="resource-chart"></canvas>
                </div>
            </div>

            <div class="card">
                <h2>処理パフォーマンス推移</h2>
                <div class="chart-container">
                    <canvas id="performance-chart"></canvas>
                </div>
            </div>
        </div>
    </div>

    <script>
        // WebSocket接続
        const socket = io();

        // チャート設定
        const chartOptions = {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            },
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        };

        // リソースチャート
        const resourceChart = new Chart(document.getElementById('resource-chart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'CPU %',
                        data: [],
                        borderColor: 'rgb(255, 99, 132)',
                        tension: 0.1
                    },
                    {
                        label: 'メモリ %',
                        data: [],
                        borderColor: 'rgb(54, 162, 235)',
                        tension: 0.1
                    },
                    {
                        label: 'GPU %',
                        data: [],
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1,
                        hidden: true
                    }
                ]
            },
            options: chartOptions
        });

        // パフォーマンスチャート
        const performanceChart = new Chart(document.getElementById('performance-chart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'FPS',
                        data: [],
                        borderColor: 'rgb(255, 206, 86)',
                        tension: 0.1,
                        yAxisID: 'y'
                    },
                    {
                        label: '成功率 %',
                        data: [],
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        beginAtZero: true
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        beginAtZero: true,
                        max: 100,
                        grid: {
                            drawOnChartArea: false
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });

        // データ更新
        const maxDataPoints = 60;

        function updateCharts(data) {
            const timestamp = new Date().toLocaleTimeString();

            // リソースチャート更新
            resourceChart.data.labels.push(timestamp);
            resourceChart.data.datasets[0].data.push(data.system.cpu_percent);
            resourceChart.data.datasets[1].data.push(data.system.memory_percent);

            if (data.system.gpu_utilization !== null) {
                resourceChart.data.datasets[2].data.push(data.system.gpu_utilization);
                resourceChart.data.datasets[2].hidden = false;
            }

            // データポイント数を制限
            if (resourceChart.data.labels.length > maxDataPoints) {
                resourceChart.data.labels.shift();
                resourceChart.data.datasets.forEach(dataset => dataset.data.shift());
            }

            resourceChart.update('none');

            // パフォーマンスチャート更新
            if (data.performance.fps !== undefined) {
                performanceChart.data.labels.push(timestamp);
                performanceChart.data.datasets[0].data.push(data.performance.fps);
                performanceChart.data.datasets[1].data.push(data.performance.success_rate * 100);

                if (performanceChart.data.labels.length > maxDataPoints) {
                    performanceChart.data.labels.shift();
                    performanceChart.data.datasets.forEach(dataset => dataset.data.shift());
                }

                performanceChart.update('none');
            }
        }

        // UI更新
        function updateUI(data) {
            // システム状態
            document.getElementById('cpu-percent').textContent = data.system.cpu_percent.toFixed(1) + '%';
            document.getElementById('memory-percent').textContent = data.system.memory_percent.toFixed(1) + '%';
            document.getElementById('disk-percent').textContent = data.system.disk_usage_percent.toFixed(1) + '%';

            if (data.system.gpu_utilization !== null) {
                document.getElementById('gpu-metric').style.display = 'flex';
                document.getElementById('gpu-percent').textContent = data.system.gpu_utilization.toFixed(1) + '%';
            }

            // ヘルスステータス
            const healthStatus = data.health.status;
            const statusElement = document.getElementById('health-status');
            const indicatorElement = statusElement.previousElementSibling;

            statusElement.textContent = healthStatus === 'healthy' ? '正常' : '異常';
            indicatorElement.className = 'status-indicator status-' + healthStatus;

            // パフォーマンスメトリクス
            if (data.performance.fps !== undefined) {
                document.getElementById('processing-fps').textContent = data.performance.fps.toFixed(1);
                document.getElementById('batch-success-rate').textContent = (data.performance.success_rate * 100).toFixed(1) + '%';
                document.getElementById('avg-processing-time').textContent = data.performance.avg_time.toFixed(0) + 'ms';
                document.getElementById('throughput').textContent = data.performance.throughput.toFixed(1) + '/秒';
            }

            // エラー統計
            document.getElementById('total-errors').textContent = data.errors.total;
            document.getElementById('error-rate').textContent = data.errors.rate.toFixed(2) + '/分';
            document.getElementById('critical-errors').textContent = data.errors.critical;

            // 最近のエラー
            const errorContainer = document.getElementById('recent-errors');
            errorContainer.innerHTML = data.errors.recent.map(error => `
                <div class="error-item">
                    <div><strong>${error.type}</strong>: ${error.message}</div>
                    <div class="timestamp">${error.timestamp} - ${error.operation}</div>
                </div>
            `).join('');

            // アラート
            const alertContainer = document.getElementById('alerts');
            alertContainer.innerHTML = data.alerts.map(alert => `
                <div class="alert alert-${alert.level}">
                    <strong>${alert.type}</strong>: ${alert.message}
                </div>
            `).join('') || '<div class="metric">アクティブなアラートはありません</div>';
        }

        // WebSocketイベント
        socket.on('connect', function() {
            console.log('Connected to monitoring server');
        });

        socket.on('monitoring_update', function(data) {
            updateUI(data);
            updateCharts(data);
        });

        // 初期データ取得
        fetch('/monitoring/api/status')
            .then(response => response.json())
            .then(data => {
                updateUI(data);
                updateCharts(data);
            });
    </script>
</body>
</html>
"""


@dashboard_bp.route("/")
def dashboard():
    """ダッシュボードページ"""
    return render_template_string(DASHBOARD_HTML)


@dashboard_bp.route("/api/status")
def api_status():
    """現在のステータスをJSON形式で返す"""
    return jsonify(get_monitoring_data())


@dashboard_bp.route("/api/metrics/<metric_name>")
def api_metric(metric_name: str):
    """特定のメトリクスの詳細を返す"""
    summary = global_metrics.get_summary(metric_name)

    if not summary:
        return jsonify({"error": "Metric not found"}), 404

    return jsonify(
        {
            "metric": metric_name,
            "summary": {
                "count": summary.count,
                "mean": summary.mean,
                "min": summary.min,
                "max": summary.max,
                "std": summary.std,
                "percentiles": summary.percentiles,
            },
        }
    )


@dashboard_bp.route("/api/errors")
def api_errors():
    """エラー情報を返す"""
    hours = int(request.args.get("hours", 24))
    summaries = error_tracker.get_error_summary(hours=hours)

    return jsonify(
        {
            "hours": hours,
            "errors": [
                {
                    "type": s.error_type,
                    "count": s.count,
                    "first_seen": s.first_seen.isoformat(),
                    "last_seen": s.last_seen.isoformat(),
                    "message": s.sample_message,
                    "operations": list(s.affected_operations),
                }
                for s in summaries[:100]
            ],
        }
    )


@dashboard_bp.route("/api/health")
def api_health():
    """ヘルスチェック結果を返す"""
    return jsonify(system_monitor.get_health_check())


def get_monitoring_data() -> dict[str, Any]:
    """モニタリングデータを取得"""
    # システム状態
    system_status = system_monitor.get_current_status()

    # パフォーマンスメトリクス
    fps_summary = global_metrics.get_summary("processing_fps", window_seconds=300)
    success_rate_summary = global_metrics.get_summary("batch_success_rate", window_seconds=300)
    processing_time_summary = global_metrics.get_summary("processing_time_ms", window_seconds=300)
    throughput_summary = global_metrics.get_summary("batch_throughput", window_seconds=300)

    # エラー情報
    error_summaries = error_tracker.get_error_summary(hours=1)
    error_rates = error_tracker.get_error_rate(window_minutes=60)

    # アラート
    alerts = []
    health_check = system_monitor.get_health_check()
    if health_check["status"] == "unhealthy":
        for issue in health_check["message"].split("; "):
            alerts.append({"type": "system", "level": "warning", "message": issue})

    # クリティカルエラーのチェック
    critical_errors = sum(1 for s in error_summaries if "Critical" in s.error_type)
    if critical_errors > 0:
        alerts.append(
            {
                "type": "error",
                "level": "critical",
                "message": f"{critical_errors}件のクリティカルエラーが発生しています",
            }
        )

    return {
        "timestamp": datetime.now().isoformat(),
        "system": {
            "cpu_percent": system_status.cpu_percent,
            "memory_percent": system_status.memory_percent,
            "memory_used_mb": system_status.memory_used_mb,
            "memory_available_mb": system_status.memory_available_mb,
            "disk_usage_percent": system_status.disk_usage_percent,
            "gpu_memory_used_mb": system_status.gpu_memory_used_mb,
            "gpu_memory_total_mb": system_status.gpu_memory_total_mb,
            "gpu_utilization": system_status.gpu_utilization,
        },
        "performance": {
            "fps": fps_summary.mean if fps_summary else 0,
            "success_rate": success_rate_summary.mean if success_rate_summary else 0,
            "avg_time": processing_time_summary.mean if processing_time_summary else 0,
            "throughput": throughput_summary.mean if throughput_summary else 0,
        },
        "errors": {
            "total": sum(s.count for s in error_summaries),
            "rate": sum(error_rates.values()),
            "critical": critical_errors,
            "recent": [
                {
                    "type": s.error_type,
                    "message": s.sample_message[:100] + "..."
                    if len(s.sample_message) > 100
                    else s.sample_message,
                    "operation": list(s.affected_operations)[0]
                    if s.affected_operations
                    else "unknown",
                    "timestamp": s.last_seen.strftime("%H:%M:%S"),
                }
                for s in error_summaries[:5]
            ],
        },
        "health": health_check,
        "alerts": alerts,
    }


def emit_monitoring_update(socketio: Any) -> None:
    """WebSocket経由でモニタリング更新を送信"""
    data = get_monitoring_data()
    socketio.emit("monitoring_update", data, namespace="/")


# 定期更新タスク（アプリケーション起動時に設定）
def start_monitoring_updates(socketio: Any, interval: int = 5) -> None:
    """モニタリング更新を開始"""
    import threading

    def update_loop():
        while True:
            try:
                emit_monitoring_update(socketio)
            except Exception as e:
                print(f"Error emitting monitoring update: {e}")

            threading.Event().wait(interval)

    thread = threading.Thread(target=update_loop, daemon=True)
    thread.start()
