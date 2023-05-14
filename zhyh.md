mbpo run_local examples.development --config=examples.config.walker2d.0 --gpus=1 --trial-gpus=1 > /dev/null &

mbpo run_local examples.development --config=examples.config.humanoid.0 --gpus=1 --trial-gpus=1 > /dev/null &

mbpo run_local examples.development --config=examples.config.ant.0 --gpus=1 --trial-gpus=1 > /dev/null &

mbpo run_local examples.development --config=examples.config.hopper.0 --gpus=1 --trial-gpus=1 > /dev/null &

mbpo run_local examples.development --config=examples.config.halfcheetah.0 --gpus=1 --trial-gpus=1 > /dev/null &

mbpo run_local examples.development --config=examples.config.inverted_pendulum.0 --gpus=1 --trial-gpus=1 > /dev/null &

export CUDA_VISIBLE_DEVICES=0

gpustat -c -u -p -F --watch

ls --time=ctime --time-style=full-iso -lt | grep seed | awk 'NR==1{print $9}'

conda activate mbpo

tensorboard --logdir=~/ray_mbpo/

ps -ef | grep ray | awk '{print $2}' | xargs -I {} kill -9 {} #每次运行完以后清空ray

ps -ef | grep ****** | awk '{print $2}' | xargs -I {} kill -9 {} #运行完以后杀死进程

sudo ldconfig /usr/local/cuda-10.0/lib64/

watch -n 1 nvidia-smi pmon -c 1