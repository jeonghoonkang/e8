# e8 - test 

## How to run docker

Build docker image: 
```
nvidia-docker build --network=host -t e8 docker/
```
## Above creates docker image of name e8

Run docker container:
```
nvidia-docker run --gpus all --name e8 --mount type=bind,source={source_data},target={target_dir} --network=host --ipc=host -i -t e8
```
For example, if the source data is in your local directory `/data/e8` and you want to mount it to the default directory `/dataset`, arguments would be `--mount type=bind,source=/data/e8,target=/dataset`.


