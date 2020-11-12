# Model-Based Meta IL with WTL


Env set up using Dockerfile- 

0. assumes nvidia drivers installed on host. use setup/dockerfile.
1. docker build -t wtl-x:latest .
2. sudo xhost +local:root
3. docker run
--rm
-it
--gpus all
-v /tmp/.X11-unix:/tmp/.X11-unix
-e DISPLAY=$DISPLAY
-e QT_X11_NO_MITSHM=1
wtl-x
bash
4. alternatively, expose port and run using jupyter notebook

