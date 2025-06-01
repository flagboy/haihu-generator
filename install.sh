#!/bin/bash
# 麻雀牌譜作成システム インストールスクリプト

set -e

# カラー出力用の定数
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ログ関数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# システム情報の検出
detect_system() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v apt-get &> /dev/null; then
            SYSTEM="ubuntu"
        elif command -v yum &> /dev/null; then
            SYSTEM="centos"
        else
            SYSTEM="linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        SYSTEM="macos"
    else
        SYSTEM="unknown"
    fi
    
    log_info "検出されたシステム: $SYSTEM"
}

# システム依存関係のインストール
install_system_dependencies() {
    log_info "システム依存関係をインストール中..."
    
    case $SYSTEM in
        "ubuntu")
            sudo apt-get update
            sudo apt-get install -y \
                python3 \
                python3-pip \
                python3-venv \
                libgl1-mesa-glx \
                libglib2.0-0 \
                libsm6 \
                libxext6 \
                libxrender-dev \
                libgomp1 \
                libgtk-3-0 \
                libavcodec-dev \
                libavformat-dev \
                libswscale-dev \
                libv4l-dev \
                libxvidcore-dev \
                libx264-dev \
                libjpeg-dev \
                libpng-dev \
                libtiff-dev \
                libatlas-base-dev \
                gfortran \
                git \
                wget \
                curl
            ;;
        "centos")
            sudo yum update -y
            sudo yum install -y \
                python3 \
                python3-pip \
                python3-devel \
                opencv-devel \
                gcc \
                gcc-c++ \
                git \
                wget \
                curl
            ;;
        "macos")
            if ! command -v brew &> /dev/null; then
                log_error "Homebrewが見つかりません。先にHomebrewをインストールしてください。"
                log_info "Homebrew インストール: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
                exit 1
            fi
            
            brew update
            brew install python@3.9 opencv git wget
            ;;
        *)
            log_warning "未対応のシステムです。手動で依存関係をインストールしてください。"
            ;;
    esac
    
    log_success "システム依存関係のインストールが完了しました"
}

# Python仮想環境の作成
create_virtual_environment() {
    log_info "Python仮想環境を作成中..."
    
    if [ -d "venv" ]; then
        log_warning "既存の仮想環境が見つかりました。削除して再作成しますか? (y/N)"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            rm -rf venv
        else
            log_info "既存の仮想環境を使用します"
            return
        fi
    fi
    
    python3 -m venv venv
    source venv/bin/activate
    
    # pipのアップグレード
    pip install --upgrade pip
    
    log_success "仮想環境の作成が完了しました"
}

# Python依存関係のインストール
install_python_dependencies() {
    log_info "Python依存関係をインストール中..."
    
    if [ ! -f "venv/bin/activate" ]; then
        log_error "仮想環境が見つかりません"
        exit 1
    fi
    
    source venv/bin/activate
    
    # 基本的な依存関係
    pip install -r requirements.txt
    
    # 開発用依存関係（オプション）
    if [ "$1" == "--dev" ]; then
        log_info "開発用依存関係をインストール中..."
        pip install \
            pytest \
            pytest-cov \
            pytest-xdist \
            flake8 \
            black \
            isort \
            mypy \
            sphinx \
            sphinx-rtd-theme
    fi
    
    log_success "Python依存関係のインストールが完了しました"
}

# ディレクトリ構造の作成
create_directories() {
    log_info "ディレクトリ構造を作成中..."
    
    mkdir -p data/input
    mkdir -p data/output
    mkdir -p data/temp
    mkdir -p logs
    mkdir -p models
    
    # .gitkeepファイルの作成（既に存在する場合はスキップ）
    touch data/input/.gitkeep
    touch data/output/.gitkeep
    touch data/temp/.gitkeep
    touch logs/.gitkeep
    touch models/.gitkeep
    
    log_success "ディレクトリ構造の作成が完了しました"
}

# 設定ファイルの初期化
initialize_config() {
    log_info "設定ファイルを初期化中..."
    
    if [ ! -f "config.yaml" ]; then
        log_warning "config.yamlが見つかりません。デフォルト設定を作成しますか? (Y/n)"
        read -r response
        if [[ ! "$response" =~ ^[Nn]$ ]]; then
            cp config.yaml.example config.yaml 2>/dev/null || {
                log_warning "config.yaml.exampleが見つかりません。基本的な設定ファイルを作成します"
                cat > config.yaml << EOF
# 麻雀牌譜作成システム設定ファイル

video:
  frame_extraction:
    fps: 1
    output_format: "jpg"
    quality: 95

ai:
  detection:
    confidence_threshold: 0.5
  classification:
    confidence_threshold: 0.8

system:
  max_workers: 4
  memory_limit: "8GB"
  gpu_enabled: false

directories:
  input: "data/input"
  output: "data/output"
  temp: "data/temp"
  models: "models"
  logs: "logs"

logging:
  level: "INFO"
  file_path: "logs/mahjong_system.log"
EOF
            }
            log_success "設定ファイルを作成しました"
        fi
    else
        log_info "既存の設定ファイルを使用します"
    fi
}

# インストール検証
verify_installation() {
    log_info "インストールを検証中..."
    
    source venv/bin/activate
    
    # Pythonモジュールのインポートテスト
    python -c "
import sys
sys.path.insert(0, '.')

try:
    from src.utils.config import ConfigManager
    from src.utils.logger import get_logger
    print('✓ 基本モジュールのインポート成功')
except ImportError as e:
    print(f'✗ インポートエラー: {e}')
    sys.exit(1)

try:
    import cv2
    import numpy as np
    import yaml
    import pandas as pd
    print('✓ 外部依存関係のインポート成功')
except ImportError as e:
    print(f'✗ 外部依存関係エラー: {e}')
    sys.exit(1)

print('✓ インストール検証完了')
"
    
    if [ $? -eq 0 ]; then
        log_success "インストール検証が完了しました"
    else
        log_error "インストール検証に失敗しました"
        exit 1
    fi
}

# 使用方法の表示
show_usage() {
    log_info "インストールが完了しました！"
    echo
    echo "使用方法:"
    echo "  1. 仮想環境をアクティベート:"
    echo "     source venv/bin/activate"
    echo
    echo "  2. システムの状態確認:"
    echo "     python main.py status"
    echo
    echo "  3. 動画を処理:"
    echo "     python main.py process input_video.mp4"
    echo
    echo "  4. ヘルプを表示:"
    echo "     python main.py --help"
    echo
    echo "設定ファイル: config.yaml"
    echo "ログファイル: logs/mahjong_system.log"
    echo
}

# GPU サポートのインストール（オプション）
install_gpu_support() {
    log_info "GPU サポートをインストールしますか? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        log_info "GPU サポートをインストール中..."
        
        source venv/bin/activate
        
        # PyTorch GPU版
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        
        # TensorFlow GPU版（オプション）
        log_info "TensorFlow GPU版もインストールしますか? (y/N)"
        read -r tf_response
        if [[ "$tf_response" =~ ^[Yy]$ ]]; then
            pip install tensorflow[and-cuda]
        fi
        
        log_success "GPU サポートのインストールが完了しました"
    fi
}

# メイン処理
main() {
    echo "========================================"
    echo "  麻雀牌譜作成システム インストーラー"
    echo "========================================"
    echo
    
    # コマンドライン引数の解析
    DEV_MODE=false
    SKIP_SYSTEM=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dev)
                DEV_MODE=true
                shift
                ;;
            --skip-system)
                SKIP_SYSTEM=true
                shift
                ;;
            --help)
                echo "使用方法: $0 [オプション]"
                echo
                echo "オプション:"
                echo "  --dev          開発用依存関係もインストール"
                echo "  --skip-system  システム依存関係のインストールをスキップ"
                echo "  --help         このヘルプを表示"
                exit 0
                ;;
            *)
                log_error "不明なオプション: $1"
                exit 1
                ;;
        esac
    done
    
    # システム検出
    detect_system
    
    # インストール手順の実行
    if [ "$SKIP_SYSTEM" = false ]; then
        install_system_dependencies
    fi
    
    create_virtual_environment
    
    if [ "$DEV_MODE" = true ]; then
        install_python_dependencies --dev
    else
        install_python_dependencies
    fi
    
    create_directories
    initialize_config
    verify_installation
    
    # GPU サポート（オプション）
    install_gpu_support
    
    show_usage
    
    log_success "インストールが正常に完了しました！"
}

# スクリプトの実行
main "$@"