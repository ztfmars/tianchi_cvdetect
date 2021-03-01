# Dockerfile
# Base Images
## 从天池基础镜像构建
FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:1.4-cuda10.1-py3
## 把当前文件夹里的文件构建到镜像的根目录下
ADD . /
## 指定默认工作目录为根目录(需要把 run.sh 和生成的结果文件都放在该文件夹下,提交后才能运行)

WORKDIR /
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install opencv-python
RUN pip install matplotlib
RUN pip install scipy
RUN pip install tensorboard
RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN sed -i s@/security.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN apt-get clean && apt-get update
RUN apt update && apt install -y libgl1-mesa-glx && apt-get install -y libglib2.0-0
RUN apt-get install -y vim
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple  --ignore-installed -r requirements.txt
## 镜像启动后统一执行 sh run.sh
CMD ["sh", "run.sh"]