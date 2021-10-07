The code was taken from [here](https://towardsdatascience.com/effective-deep-learning-development-environment-with-pycharm-and-docker-34018f122d92).

1. Build container and run
```
docker build --tag center_net_env --file test/DockerCenterNet/CenterNet.dockerfile .

docker run --gpus all -it --rm -p 8022:22 -v /home/vpavlishen/CenterNet:/home/test --name center_net_env center_net_env bash
```
2. In the container do the following commands:
```
apt update && apt install -y openssh-server

mkdir /var/run/sshd

echo 'root:<USE_YOUR_OWN_STRONG_PASSWORD>' | chpasswd
# Root password was changed with <USE_YOUR_OWN_STRONG_PASSWORD>

sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

echo "export VISIBLE=now" >> /etc/profile

service ssh restart

adduser user

adduser user sudo
```
3. In another terminal window, outside the container
```
docker commit center_net_env center_net_env_ssh
```
4. Exit the container
5. Run container from the new image.
```
docker run --gpus all -it --rm -p 8022:22 -v /home/vpavlishen/CenterNet:/home/test --name center_net_env_ssh center_net_env_ssh bash

service ssh start
```
6.  ssh tunnel
```
ssh -L 8022:192.168.0.55:8022 vpavlishen@195.91.176.132
```

7. Connect to python interpreter through ssh tunnel.