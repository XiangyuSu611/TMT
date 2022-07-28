# Denpendency & Environment
We recommend using a docker container to run our code, but we still provide a way to configure the environment without docker.

## Run TMT with docker
First install [docker](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository) following the instructions linked. Once you have this installed, download and start our container using the following command:
```shell
# pull docker image.
docker pull xiangyusu611/tmt:release
# create a new container.
docker run -it --gpus all --name TMT -p 1111:1111 xiangyusu611/tmt:release bash
# start container.
docker start TMT
# enter container.
sudo docker exec -it TMT bash
# enter python enviornment.
conda deactivate
pyenv activate TMT
```
Code has been placed in `/home/code/TMT/`.
## Run TMT without docker
### Python Environment
We use pyenv to manage python environments.
```shell
# install pyenv 
curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
pyenv update
# install dependencies 
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
xz-utils tk-dev libffi-dev liblzma-dev
# install python
pyenv install 3.7.1
# create new environments
pyenv virtualenv 3.7.1 TMT
# install python packages
pip install -r requirements.txt
```
Blender is used as an shape processing and rendering tool. We install Blender as a Python module. Here are some [guidelines](https://gist.github.com/keunhong/279c98de28877a3a33a1eb95fa7d56a5).
### Non-python dependencies
+ Blender 2.79 or higher
+ NVIDIA GPU for deep learning and OpenGL rendering
### More help
💪We appreciate the initial environment provided by [Photoshape](https://github.com/keunhong/photoshape). If you encounter problems during the environment configuration, you can try to find help here or create an issue in this page.