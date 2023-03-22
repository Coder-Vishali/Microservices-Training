# Set base image as jenkins official docker image
FROM jenkins/jenkins as base

FROM base as base-amd64

ENV NVARCH x86_64

ENV NVIDIA_REQUIRE_CUDA "cuda>=11.5 brand=tesla,driver>=418,driver<419 brand=tesla,driver>=450,driver<451 brand=tesla,driver>=470,driver<471 brand=unknown,driver>=470,driver<471 brand=nvidia,driver>=470,driver<471 brand=nvidiartx,driver>=470,driver<471 brand=geforce,driver>=470,driver<471 brand=geforcertx,driver>=470,driver<471 brand=quadro,driver>=470,driver<471 brand=quadrortx,driver>=470,driver<471 brand=titan,driver>=470,driver<471 brand=titanrtx,driver>=470,driver<471"
ENV NV_CUDA_CUDART_VERSION 11.5.50-1
ENV NV_CUDA_COMPAT_PACKAGE cuda-compat-11-5

FROM base as base-arm64

ENV NVARCH sbsa
ENV NVIDIA_REQUIRE_CUDA "cuda>=11.5"
ENV NV_CUDA_CUDART_VERSION 11.5.50-1

FROM base-amd64

LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

# Set USER
USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/${NVARCH}/3bf863cc.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/${NVARCH} /" > /etc/apt/sources.list.d/cuda.list && \
    apt-get purge --autoremove -y curl \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_VERSION 11.5.0

# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-11-5=${NV_CUDA_CUDART_VERSION} \
    ${NV_CUDA_COMPAT_PACKAGE} \
    && rm -rf /var/lib/apt/lists/*

# Required for nvidia-docker v1
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
    && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

COPY NGC-DL-CONTAINER-LICENSE /

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Define the proxies
ENV HTTP_PROXY "<proxy>"
ENV HTTPS_PROXY "<proxy>"

# Allow local checkout in jenkins
ENV JAVA_OPTS "-Dhudson.plugins.git.GitSCM.ALLOW_LOCAL_CHECKOUT=true"

RUN printf 'Acquire::http::Proxy "<proxy>";\n Acquire::https::Proxy "<proxy>";\n' >> /etc/apt/apt.conf

# Create USER
RUN useradd -ms /bin/bash <user>
# Set Password
RUN echo '<user>:<pass>' | chpasswd

# Update Ubuntu Software repository
RUN apt-get clean
RUN apt-get update -y

# Install python
RUN apt-get install -y python3
RUN apt-get install -y python3-venv
RUN apt-get install -y python3-pip

# Install Git
RUN apt-get install git

# Install other necessary packages
RUN apt-get install net-tools
RUN apt-get install -y openssh-server 
RUN apt-get install -y vim 
RUN apt-get install -y curl
RUN apt install sudo 

# Set the workspace directory
WORKDIR /home

# Copy the contents of your project repository
COPY ./requirements.txt .
COPY ./startup.sh .

# Setup the environment
RUN chmod +x startup.sh
RUN ./startup.sh
