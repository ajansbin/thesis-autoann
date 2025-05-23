FROM nvcr.io/nvidia/pytorch:21.06-py3

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV FORCE_CUDA="1"

ENV TZ=Europe/Stockholm
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Lock a specific opencv version
RUN pip install opencv-python==4.5.5.64

# Install MMCV
RUN pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.9.0/index.html

# Install MMDetection and MMSegmentation
RUN pip install mmdet==2.25 mmsegmentation==0.28

# Fixes
RUN pip install pyyaml llvmlite setuptools==59.5.0 --ignore-installed pyyaml llvmlite

# Extra dependencies
RUN pip install zod wandb

# Install MMDetection3D requirements
RUN conda clean --all
COPY . /mmdetection3d
WORKDIR /mmdetection3d
RUN pip install --no-cache-dir -e .
ENV PYTHONPATH='.'
