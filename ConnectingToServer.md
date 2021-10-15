The code was taken from [here](https://towardsdatascience.com/effective-deep-learning-development-environment-with-pycharm-and-docker-34018f122d92).

1. Build container and run. (Before running write in dockerfile user and password)
```
docker build --tag remote_ssh_python --file path/to/dockerfile/remote_ssh_python.dockerfile .
docker run --gpus all -d -it -p 8022:22  --name remote_ssh_python remote_ssh_python
```
2. Run the following command:
```
docker exec -it remote_ssh_python service ssh start
```
3.  ssh tunnel (own machine)
```
ssh -L 8022:192.168.0.55:8022 test_user@195.91.176.132
```

4. Connect to the python interpreter through a ssh tunnel using the "user" defined in dockerfile
5. To execute files, You must sync folders. (Note: create separate directly)