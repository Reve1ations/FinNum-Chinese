# 指定基础镜像，选择我之前制作的paddle镜像
FROM hub.data.wust.edu.cn:30880/nlp/paddle:1.6.3-gpu-cuda10.0-cudnn7

# 将工程下面的文件复制到容器某个目录
COPY ./ /home/Model

# 设置容器中的工作目录
WORKDIR /home/Model

# 安装依赖
RUN python3 -m pip install --upgrade pip \
&& python3 -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple some-package

# 设置容器启动时的命令，这里直接运行data/run_MAMS.sh
ENTRYPOINT ["bash", "/home/Model/data/run_MAMS.sh"]
