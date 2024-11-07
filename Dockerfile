# inspired by https://github.com/slabban/ros2_pytorch_cuda/blob/master/docker/Dockerfile
# DOCKER_BUILDKIT=1 docker build --build-arg USER_ID=$(id -u) --build-arg USER_NAME=$(whoami) --build-arg GROUP_ID=$(id -g) -t ros2-image-processing-rico .

# FROM nvidia/cuda:11.3.1-devel-ubuntu20.04
FROM rwthika/ros2-ml:iron-desktop-full-tf2.16.1-torch2.3.0-v24.08

# disable terminal interaction for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install cudnn8 and move necessary header files to cuda include directory
RUN apt-get update && apt install libcudnn8-dev -y && \
	cp /usr/include/cudnn_version.h /usr/local/cuda/include && \
	cp /usr/include/cudnn.h /usr/local/cuda/include/ && \
	rm -rf /var/lib/apt/lists/*

# Fundamentals
RUN apt-get update && apt-get install -y \
        bash-completion \
        build-essential \
        clang-format \
        cmake \
        curl \
        git \
        gnupg2 \
        locales \
        lsb-release \
        rsync \
        software-properties-common \
        wget \
        vim \
        unzip \
        mlocate \
	libgoogle-glog-dev \
        && rm -rf /var/lib/apt/lists/*  

# Install libtorch
RUN wget https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcu113.zip && \
        unzip libtorch-cxx11-abi-shared-with-deps-1.11.0+cu113.zip && \
        rm -rf libtorch-cxx11-abi-shared-with-deps-1.11.0+cu113.zip

# Python basics
RUN apt-get update && apt-get install -y \
        python3-flake8 \
        python3-opencv \
        python3-pip \
        python3-pytest-cov \
        python3-setuptools \
        && rm -rf /var/lib/apt/lists/*

# Python3 (PIP)
RUN python3 -m pip install -U \
        argcomplete \
        # autopep8 \
        flake8 \
        # flake8-blind-except \
        # flake8-builtins \
        # flake8-class-newline \
        # flake8-comprehensions \
        # flake8-deprecated \
        # flake8-docstrings \
        # flake8-import-order \
        # flake8-quotes \
        onnx \
        # torch \
        # torchvision \
        # scikit-learn \
        pytest 
        # pytest-repeat \
        # pytest-rerunfailures \
        # pydocstyle \
# RUN python3 -m pip install -U tensorflow
RUN python3 -m pip install -U \
        matplotlib \
        pillow
RUN pip install -U jupyter notebook 

# Create a user to match the host's UID/GID and create necessary directories
# Ros needs to access /home/${USER_NAME}/.ros/ So we need to set up the permission properly there. 
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USER_NAME
RUN echo "Building as user ${USER_NAME}"
RUN groupadd -g ${GROUP_ID} ${USER_NAME} && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash ${USER_NAME} && \
    mkdir -p /home/${USER_NAME}/.ros && \
    chown -R ${USER_NAME}:${USER_NAME} /home/${USER_NAME}

# Create and add content to the .inputrc file
RUN echo '"\e[A": history-search-backward' >> /home/${USER_NAME}/.inputrc && \
    echo '"\e[B": history-search-forward' >> /home/${USER_NAME}/.inputrc && \
    echo "source /opt/ros/foxy/setup.bash" >> ~/.bashrc && \
    echo "source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash" >> ~/.bashrc && \
    echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> ~/.bashrc

ENV USER_NAME=${USER_NAME}

# Switch to user
USER ${USER_NAME}
WORKDIR /home/${USER_NAME}

# jupyter notebook port
EXPOSE 8888

CMD ["bash"]
