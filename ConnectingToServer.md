The code was taken from [here](https://towardsdatascience.com/effective-deep-learning-development-environment-with-pycharm-and-docker-34018f122d92).

1. Build container and run. (Before running write in dockerfile user and password)
```
docker build --tag remote_ssh_python --file path/to/dockerfile/remote_ssh_python.dockerfile .
docker run --gpus all -it -p 8022:22  --name remote_ssh_python remote_ssh_python bash
```
2. Exit the container and run the following command:
```
docker start "container_id"    
```
3. Go inside the container:
```
docker exec -it "container_id" bash
```
4. Inside the container run the following command:
```
service ssh start
```
5. Now you can quit from container
6.  ssh tunnel (own machine)
```
ssh -L 8022:192.168.0.55:8022 test_user@195.91.176.132
```

7. Connect to the python interpreter through a ssh tunnel using the "user" defined in dockerfile
8. To execute files, You must sync folders. (Note: create separate directly)