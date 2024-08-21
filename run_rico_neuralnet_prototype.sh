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

IMAGE_NAME=ros2-image-processing-rico
THIS_DIRECTORY_PATH="$(dirname "${BASH_SOURCE[0]}")"
THIS_DIRECTORY_NAME="$(basename "$(dirname "${BASH_SOURCE[0]}")")"
# Add this for nvidia 
# --runtime=nvidia \
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
    -it --rm \
    --privileged \
    ${IMAGE_NAME}
