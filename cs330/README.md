# This doc has the instructions to run WTL. 


Steps (if you are running on Adrian's VM):
-------------------------------------------
1. sudo nvidia-xconfig --busid=PCI:0:30:0 --use-display-device=none --virtual=1280x1024
2. sudo Xorg :1
3. export DISPLAY=:1
4. conda activate py37c
5. python run_run_meta_env.py --gin_config configs/collect_random_trial.gin <or anything other config file> 

Note that before running step 1 and step 2, do a "ps -ef | grep Xorg" to see if it is run already. We only need to do this once per start. 

Example: 
--------
(py37c) adrian_cmu@cs330-tf115-vm:/tmp/tmp2/cs330/watch_try_learn_toy$ python run_run_meta_env.py --gin_config configs/collect_random_trial.gin
... 
I1105 02:36:46.607171 139970320729856 run_meta_env.py:153] Demo: -43.68	Trial: -43.32	Inference: -40.92

Creating a local repo: 
-----------------------
git clone https://github.com/adriancmu/cs330.git
