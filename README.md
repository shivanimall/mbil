# Model-Based Meta IL with WTL


Env set up using Dockerfile- 

assumes nvidia drivers installed on host. use setup/dockerfile.
docker build -t wtl-x:latest .
sudo xhost +local:root
docker run
--rm
-it
--gpus all
-v /tmp/.X11-unix:/tmp/.X11-unix
-e DISPLAY=$DISPLAY
-e QT_X11_NO_MITSHM=1
wtl-x
bash
alternatively, expose port and run using jupyter notebook

