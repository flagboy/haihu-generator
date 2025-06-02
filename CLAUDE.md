# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 重要: 言語設定 (Language Setting)

**すべての出力は日本語で行うこと。** コメント、ドキュメント、エラーメッセージ、ユーザーへの応答はすべて日本語を使用してください。

## Project Overview

This is a mahjong game record creation system that processes mahjong game videos (MP4, etc.) to automatically generate game records in Tenhou JSON format. It uses AI to detect and classify mahjong tiles from video frames, tracks game states, and outputs standardized game records.

## Environment

This project uses **uv** as the package manager for Python dependency management. The `pyproject.toml` and `uv.lock` files define the project dependencies.

### Installing Dependencies
When installing packages, always use `uv` instead of `pip`:
```bash
# Install new dependency
uv add package_name

# Install development dependency
uv add --dev package_name

# Install all dependencies
uv sync

# Install optional dependencies
uv add package_name[extra]
```

### Running Python Commands
Use `uv run` for executing Python commands:
```bash
# Run pytest
uv run pytest

# Run python script
uv run python script.py

# Run with specific Python version
uv run --python 3.11 python script.py
```

## Development Guidelines

### 重要: 実行前の深い思考 (Deep Thinking Before Execution)

**毎回必ず実行前に深い思考（ultrathink）を行うこと:**
- コードの変更前に、その影響範囲を十分に検討する
- テストの実行前に、期待される結果を明確にする
- エラーの修正前に、根本原因を特定する
- 新機能の実装前に、既存システムとの整合性を確認する
- パフォーマンスの最適化前に、ボトルネックを正確に把握する

この深い思考プロセスにより、より正確で効率的な開発が可能になります。

### 重要: コミット前のテスト確認 (Test Verification Before Commit)

**コード変更をコミットする前に、必ず全てのテストが成功することを確認すること:**
- `pytest` コマンドですべてのテストを実行
- 失敗したテストがある場合は、修正してから再度テストを実行
- 全てのテストが成功（PASSED）することを確認してからコミット
- 特に、変更に関連するテストは必ず実行して確認する

### 重要: プルリクエスト作成ガイドライン (Pull Request Creation Guidelines)

**修正してテストが全て成功したらプルリクエストを作成すること:**
- 全てのテストが成功することを確認してからプルリクエストを作成
- すでにプルリクエストが存在する場合は、コミットとプッシュのみを実行
- プルリクエストのタイトルと説明は変更内容を明確に記載
- レビューが必要な重要な変更については、適切なレビュワーを指定

## Common Development Commands

### Environment Setup
```bash
# Install dependencies using uv (recommended)
uv sync

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install with auto script
./install.sh --dev  # includes dev dependencies
```

### Running the System
```bash
# Check system status
python main.py status

# Process a single video
python main.py process input_video.mp4

# Run batch processing
python main.py batch input_directory output_directory

# Validate a game record
python main.py validate record.json

# Run optimization
python main.py optimize
```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test
uv run pytest tests/test_integration.py -v

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run performance tests
uv run pytest tests/test_performance.py -v

# Run integration tests
uv run python run_integration_tests.py
```

### Code Quality
```bash
# Format code
black src tests

# Sort imports
isort src tests

# Lint code
flake8 src tests

# Type checking
mypy src
```

### Web Interface
```bash
# Start web interface
cd web_interface
python run.py
# Access at http://localhost:5000
```

## High-Level Architecture

The system follows a layered architecture with pipeline patterns:

### Core Processing Pipeline
1. **Video Input Layer** (`src/video/`)
   - `VideoProcessor`: Extracts and preprocesses frames from video
   - `FrameExtractor`: Extracts frames for training data

2. **AI/ML Layer** (`src/detection/`, `src/classification/`)
   - `TileDetector`: Detects mahjong tiles in frames using YOLO/Detectron2
   - `TileClassifier`: Classifies detected tiles using CNN/ViT
   - `AIPipeline`: Integrates detection and classification with batch processing

3. **Game Logic Layer** (`src/game/`, `src/tracking/`)
   - `GameState`: Manages current game state
   - `StateTracker`: Tracks state changes over time
   - `ActionDetector`: Detects player actions (draw, discard, call)
   - `GamePipeline`: Orchestrates game logic processing

4. **Output Layer** (`src/output/`)
   - `TenhouJsonFormatter`: Formats game records in Tenhou JSON format

### Supporting Systems

1. **Learning System** (`src/training/`)
   - `DatasetManager`: Manages training datasets with versioning
   - `TrainingManager`: Handles model training sessions
   - `ModelEvaluator`: Evaluates model performance
   - `SemiAutoLabeler`: Semi-automatic labeling for training data

2. **Optimization System** (`src/optimization/`)
   - `PerformanceOptimizer`: Overall system optimization
   - `GPUOptimizer`: GPU usage optimization
   - `MemoryOptimizer`: Memory usage optimization

3. **Validation System** (`src/validation/`)
   - `QualityValidator`: Validates output quality
   - `TenhouValidator`: Validates Tenhou format compliance
   - `ConfidenceCalculator`: Calculates confidence scores

4. **Integration** (`src/integration/`)
   - `SystemIntegrator`: Main entry point that orchestrates all components

## Key Configurations

Main configuration file: `config.yaml`

Important sections:
- `video`: Frame extraction settings
- `ai`: Detection/classification thresholds
- `training`: Learning system settings
- `system`: Performance settings (workers, memory, GPU)
- `directories`: Data paths
- `validation`: Quality thresholds

## Data Flow

1. Video → Frame extraction → AI detection/classification
2. Detected tiles → Game state tracking → Action detection
3. Game states → Tenhou JSON formatting → Validation → Output

## Important Notes

- The system is optimized for Tenhou JSON format output
- Default tile classes: 34 types (1-9m, 1-9p, 1-9s, 7 honor tiles)
- GPU is recommended for performance but not required
- Training data uses SQLite database for management
- Web interface provides GUI for data management and labeling
