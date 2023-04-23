FROM nvidia/cuda:11.7.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
## Set timezone as it is required by some packages.
ENV TZ=Europe/Zagreb
## CUDA architectures, required by tiny-cuda-nn.
ENV TCNN_CUDA_ARCHITECTURES=61
## CUDA Home, required to find CUDA in some packages.
ENV CUDA_HOME="/usr/local/cuda"
ENV CMAKE_CUDA_ARCHITECTURES=61

# Update and upgrade all packages
RUN apt update -y
RUN apt upgrade -y

RUN apt install -y git python3 software-properties-common python3-pip

RUN apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    htop \
    ffmpeg \
    wget \
    gcc \
    g++ \
    git \
    cmake \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libqt5gui5 \
    qt5-qmake \
    libxcb-util-dev \
    libprotobuf-dev \
    libatlas-base-dev \
    xz-utils \
    nano \
    unzip

# Install python packets
COPY requirements.txt /tmp/requirements.txt
COPY requirements-dev.txt /tmp/requirements-dev.txt

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements-dev.txt

ENV TERM xterm-256color
RUN echo 'export PS1="\A \[\033[1;36m\]\h\[\033[1;34m\] \w \[\033[0;015m\]\\$ \[$(tput sgr0)\]\[\033[0m\]"' >> ~/.bashrc
ENV PATH="${PATH}:/blender-3.4.1-linux-x64"

CMD ["bash", "-l"]
