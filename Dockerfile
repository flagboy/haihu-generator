# 麻雀牌譜作成システム Dockerfile
# マルチステージビルドで最適化
FROM python:3.9-slim as builder

# ビルド用パッケージのインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libgl1-mesa-dev \
    libglib2.0-dev \
    libgtk-3-dev \
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
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# uvのインストール
RUN pip install uv

# 作業ディレクトリの設定
WORKDIR /app

# 依存関係ファイルをコピー
COPY pyproject.toml uv.lock* ./

# 依存関係のインストール
RUN uv sync --frozen

# 本番用イメージ
FROM python:3.9-slim as production

# 非rootユーザーの作成
RUN groupadd -r mahjong && useradd -r -g mahjong mahjong

# ランタイム用パッケージのインストール
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgtk-3-0 \
    libavcodec58 \
    libavformat58 \
    libswscale5 \
    libv4l-0 \
    libxvidcore4 \
    libx264-160 \
    libjpeg62-turbo \
    libpng16-16 \
    libtiff5 \
    libatlas3-base \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 作業ディレクトリの設定
WORKDIR /app

# ビルダーステージから仮想環境をコピー
COPY --from=builder /app/.venv /app/.venv

# 仮想環境をPATHに追加
ENV PATH="/app/.venv/bin:$PATH"

# アプリケーションコードをコピー
COPY --chown=mahjong:mahjong . .

# 必要なディレクトリを作成
RUN mkdir -p \
    data/input \
    data/output \
    data/temp \
    data/training \
    logs \
    models \
    web_interface/uploads \
    web_interface/logs \
    && chown -R mahjong:mahjong /app

# 環境変数の設定
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LOG_LEVEL=INFO
ENV GPU_ENABLED=0

# セキュリティ設定
ENV PYTHONHASHSEED=random

# ヘルスチェック用のスクリプト
COPY --chown=mahjong:mahjong docker/healthcheck.py /app/healthcheck.py
RUN chmod +x /app/healthcheck.py

# ヘルスチェックの設定
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python /app/healthcheck.py

# ポートの公開
EXPOSE 5000 8080

# 非rootユーザーに切り替え
USER mahjong

# デフォルトコマンド
CMD ["python", "main.py", "status"]

# 開発用イメージ
FROM production as development

# rootユーザーに戻る（開発用パッケージインストールのため）
USER root

# 開発用パッケージのインストール
RUN apt-get update && apt-get install -y \
    vim \
    git \
    htop \
    && rm -rf /var/lib/apt/lists/*

# 開発用Python依存関係
RUN /app/.venv/bin/pip install \
    pytest \
    pytest-cov \
    pytest-xdist \
    black \
    flake8 \
    isort \
    mypy \
    jupyter \
    ipython

# 開発用環境変数
ENV LOG_LEVEL=DEBUG
ENV FLASK_ENV=development
ENV FLASK_DEBUG=1

# 非rootユーザーに戻る
USER mahjong

# 開発用デフォルトコマンド
CMD ["python", "main.py", "status"]

# GPU対応イメージ
FROM nvidia/cuda:11.8-runtime-ubuntu20.04 as gpu

# 非rootユーザーの作成
RUN groupadd -r mahjong && useradd -r -g mahjong mahjong

# Python 3.9のインストール
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgtk-3-0 \
    libavcodec58 \
    libavformat58 \
    libswscale5 \
    libv4l-0 \
    libxvidcore4 \
    libx264-160 \
    libjpeg8-dev \
    libpng16-16 \
    libtiff5 \
    libatlas3-base \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Python 3.9をデフォルトに設定
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# pipのアップグレード
RUN python -m pip install --upgrade pip

# 作業ディレクトリの設定
WORKDIR /app

# ビルダーステージから仮想環境をコピー
COPY --from=builder /app/.venv /app/.venv

# GPU用PyTorchの再インストール
RUN /app/.venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# アプリケーションコードをコピー
COPY --chown=mahjong:mahjong . .

# 必要なディレクトリを作成
RUN mkdir -p \
    data/input \
    data/output \
    data/temp \
    data/training \
    logs \
    models \
    web_interface/uploads \
    web_interface/logs \
    && chown -R mahjong:mahjong /app

# 環境変数の設定
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LOG_LEVEL=INFO
ENV GPU_ENABLED=1
ENV CUDA_VISIBLE_DEVICES=0

# ヘルスチェック用のスクリプト
COPY --chown=mahjong:mahjong docker/healthcheck.py /app/healthcheck.py
RUN chmod +x /app/healthcheck.py

# ヘルスチェックの設定
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python /app/healthcheck.py

# ポートの公開
EXPOSE 5000 8080

# 非rootユーザーに切り替え
USER mahjong

# GPU用デフォルトコマンド
CMD ["python", "main.py", "status"]