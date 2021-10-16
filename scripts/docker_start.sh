docker run -it --runtime=nvidia --rm -P --shm-size=120g --ulimit memlock=-1 \
    --ulimit stack=67108864 --name mvt --net host -v xxx:yyy {docker-img} bash
