# inspired by https://github.com/slabban/ros2_pytorch_cuda/blob/master/docker/Dockerfile
# DOCKER_BUILDKIT=1 docker build --build-arg USER_ID=$(id -u) --build-arg USER_NAME=$(whoami) --build-arg GROUP_ID=$(id -g) -t ros2-image-processing-rico .

FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

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
        autopep8 \
        flake8 \
        flake8-blind-except \
        flake8-builtins \
        flake8-class-newline \
        flake8-comprehensions \
        flake8-deprecated \
        flake8-docstrings \
        flake8-import-order \
        flake8-quotes \
        onnx \
        pytest-repeat \
        pytest-rerunfailures \
        pytest \
        pydocstyle \
        scikit-learn \
        torch \
        torchvision 

RUN python3 -m pip install -U tensorflow
RUN python3 -m pip install -U \
        matplotlib \
        pillow
RUN pip install -U jupyter notebook 

# # Setup ROS2 Foxy
# RUN locale-gen en_US en_US.UTF-8
# RUN update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
# ENV LANG=en_US.UTF-8

# RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
# RUN sh -c 'echo "deb [arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list'      

# RUN apt-get update && apt-get install -y \
#         python3-colcon-common-extensions \
#         python3-rosdep \
#         python3-vcstool \
#         ros-foxy-camera-calibration-parsers \
#         ros-foxy-camera-info-manager \
#         ros-foxy-desktop \
#         ros-foxy-launch-testing-ament-cmake \
#         ros-foxy-rqt* \
#         ros-foxy-v4l2-camera \
#         ros-foxy-vision-msgs \
#         ros-foxy-pcl-conversions \
#         ros-foxy-sensor-msgs-py \
#         ros-foxy-stereo-image-proc \
#         ros-foxy-pcl-ros \
#         ros-foxy-usb-cam \
#         && rm -rf /var/lib/apt/lists/*

# RUN apt-get update && apt-get install -y \
# 	&& apt install ros-foxy-rmw-cyclonedds-cpp -y \
#         && rm -rf /var/lib/apt/lists/*

# RUN rosdep init

# RUN rosdep update

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