# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a mahjong game record creation system that processes mahjong game videos (MP4, etc.) to automatically generate game records in Tenhou JSON format. It uses AI to detect and classify mahjong tiles from video frames, tracks game states, and outputs standardized game records.

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
pytest

# Run specific test
pytest tests/test_integration.py -v

# Run with coverage
pytest --cov=src --cov-report=html

# Run performance tests
pytest tests/test_performance.py -v

# Run integration tests
python run_integration_tests.py
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
