FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu18.04
ENV TZ=Asia/Novokuznetsk
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get install -y --no-install-recommends \
        python3-dev \
        python3-pip \
        python3-opencv \
        python3-psycopg2 \
        python3-setuptools \
        ca-certificates \
        git \
        wget \
        sudo \
        cmake \
        ninja-build \
        libgl1-mesa-dev \
        libgtk2.0-dev \
        gcc \
        g++ \
        unixodbc-dev \
        odbc-postgresql
RUN apt update && apt install -y openssh-server
RUN apt-get clean
RUN pip3 install --upgrade pip

#COPY ./requirements.txt /tmp/requirements.txt
#RUN pip3 install -r /tmp/requirements.txt
#RUN pip3 install torch==1.7.0+cu110 -f https://download.pytorch.org/whl/torch_stable.html
#RUN pip3 install torchvision==0.8.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html

# PASSWORD
RUN mkdir /var/run/sshd \
    && echo 'root:PASSWORD' | chpasswd \
    && sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd \
    && echo "export VISIBLE=now" >> /etc/profile \
    && service ssh restart

# PASSWORD
RUN useradd -ms /bin/bash user \
    && echo 'user:PASSWORD' | chpasswd

RUN service ssh start

