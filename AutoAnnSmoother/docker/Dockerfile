ARG PYTORCH="1.9.0"
ARG CUDA="11.1"
ARG CUDNN="8"
#FROM nvcr.io/nvidia/pytorch:21.06-py3
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 8.0 8.6+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV FORCE_CUDA="1"
# To fix GPG key error when running apt-get update (https://github.com/NVIDIA/nvidia-docker/issues/1631)
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    #libegl1 libgles2 libglvnd-dev libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN conda clean --all
RUN conda update -n base -c defaults conda

# install pytorch and other requirements
RUN yes | pip install numpy
RUN conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c conda-forge -y

RUN git clone https://github.com/coltonstearns/SpOT.git /SpOT
WORKDIR /SpOT

RUN pip install -r requirements.txt --ignore-installed llvmlite
# install MMCV as base-package for some CUDA kernels in third_party/
RUN yes | pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
RUN yes | pip install mmdet==2.11.0
# install pytorch-geometric and pytorch3d
RUN conda install pyg==2.0.3 -c pyg -c conda-forge -y
RUN conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
RUN conda install -c bottler nvidiacub -y
RUN conda install pytorch3d==0.6.2 -c pytorch3d -y

# compile PointNet++ CUDA kernels
WORKDIR /SpOT/third_party/pointnet2
RUN python setup.py install
WORKDIR /SpOT

# compile general CUDA kernels
RUN python setup.py develop

RUN pip install numpy==1.19.2 --no-cache-dir

