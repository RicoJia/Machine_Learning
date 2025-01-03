# jetson-containers build --name=rico_cuda_image jupyterlab pytorch torchvision ros:iron-desktop
# On desktop, it's equivalent to rwthika/ros2-ml
# docker exec -it ros2-image-processing-rico-container bash
# Intro:
#   - tensorboard uses port 6006
show_help(){
  echo "Usage: $0 [-j] [-h]"
  echo
  echo "Options:"
  echo "  -j    Enable Jupyter"
  echo "  -h    Show help"
  exit 0
}

echo "Preparing Xauthority data..."
xauth_list=$(xauth nlist :0 | tail -n 1 | sed -e 's/^..../ffff/')
if [ ! -f $XAUTH ]; then
    if [ ! -z "$xauth_list" ]; then
        echo $xauth_list | xauth -f $XAUTH nmerge -
    else
        touch $XAUTH
    fi
    chmod a+r $XAUTH
fi

echo "Done."
echo ""
echo "Verifying file contents:"
file $XAUTH
echo "--> It should say \"X11 Xauthority data\"."
echo ""
echo "Permissions:"
ls -FAlh $XAUTH
echo ""
echo "Running docker..."

DESKTOP_IMAGE_NAME=ros2-image-processing-rico
ORIN_IMAGE_NAME=rico_cuda_image
ARCHITECTURE=$(dpkg --print-architecture)

# Check if the CUDA image exists by trying to retrieve its ID
if [ "$ARCHITECTURE" == "amd64" ] || [ "$ARCHITECTURE" == "x86_64" ]; then
    IMAGE_NAME="$DESKTOP_IMAGE_NAME"
    TAG_NAME=latest
else
    IMAGE_NAME="$ORIN_IMAGE_NAME"
    TAG_NAME=r36.3.0
fi

echo "Launching image ${IMAGE_NAME}:${TAG_NAME}"

THIS_DIRECTORY_PATH="$(dirname "${BASH_SOURCE[0]}")"
THIS_DIRECTORY_NAME="$(basename "$(dirname "${BASH_SOURCE[0]}")")"
# Add this for nvidia 
# --runtime=nvidia \

jupyter_enabled=false
while getopts "jh" opt; do
  case ${opt} in
    j)
      jupyter_enabled=true
      ;;
    h)
      show_help
      ;;
    *)
      show_help
      ;;
  esac
done

if $jupyter_enabled; then
echo "jupyter"
    docker run \
        --name ${IMAGE_NAME}-container \
        -e DISPLAY=$DISPLAY \
        -e QT_X11_NO_MITSHM=1 \
        -e XAUTHORITY=$XAUTH \
        -v /etc/passwd:/etc/passwd:ro \
        -v /etc/group:/etc/group:ro \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        -v $HOME/.Xauthority:/root/.Xauthority \
        -v ${THIS_DIRECTORY_PATH}:/home/${USER}/${THIS_DIRECTORY_NAME} \
        -v ~/.ssh:/root/.ssh \
        -v /dev/bus/usb:/dev/bus/usb \
        -v $XAUTH:$XAUTH \
        -v /dev:/dev:rw \
        -p 8888:8888 \
        -itd --rm \
        --runtime nvidia \
        --network=host \
        --privileged \
        ${IMAGE_NAME}:${TAG_NAME} \
        jupyter notebook --ip=0.0.0.0 --no-browser --allow-root
else 
    docker run \
        --name ${IMAGE_NAME}-container \
        -e DISPLAY=$DISPLAY \
        -e QT_X11_NO_MITSHM=1 \
        -e XAUTHORITY=$XAUTH \
        -v /etc/passwd:/etc/passwd:ro \
        -v /etc/group:/etc/group:ro \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        -v $HOME/.Xauthority:/root/.Xauthority \
        -v ${THIS_DIRECTORY_PATH}:/home/${USER}/${THIS_DIRECTORY_NAME} \
        -v ~/.ssh:/root/.ssh \
        -v /dev/bus/usb:/dev/bus/usb \
        -v $XAUTH:$XAUTH \
        -v /dev:/dev:rw \
        -p 8888:8888 \
        -p 6006:6006 \
        -w /home/${USER}/${THIS_DIRECTORY_NAME} \
        -itd --rm \
        --runtime nvidia \
        --network=host \
        --privileged \
        ${IMAGE_NAME}:${TAG_NAME} /bin/bash -c "\
    echo 'Creating .inputrc file' && \
    cat <<EOL > /root/.inputrc
\"\e[A\": history-search-backward
\"\e[B\": history-search-forward
EOL
    bash"
fi

 #docker run --runtime nvidia -it --rm --network=host -v /home/rico:/home/rico rico_cuda_image:r36.3.0-cu122-torchvision
