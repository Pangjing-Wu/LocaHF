FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime


RUN apt-get update && apt-get install -y \
    git \
    sudo \
    lsb-release \
    wget \
    curl \
    build-essential \
    libgl1-mesa-glx \ 
    && rm -rf /var/lib/apt/lists/*

# ------------------------ Python 依赖 ------------------------
# 安装常用数据科学包
RUN pip install --no-cache-dir \
    numpy==1.24.3 \
    pandas==2.0.3 \
    fastapi==0.115.12 \
    uvicorn[standard]==0.34.2 \
    pydantic==2.11.1 \
    transformers==4.51.3 \
    torch==2.7.0 \
    accelerate==1.6.0


# ------------------------ 项目配置 ------------------------
# 设置工作目录
WORKDIR /workspace

# 复制本地代码到容器
COPY . .