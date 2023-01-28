**To create a volume:**

docker volume create test-data	

**To inspect the volume:**

docker volume inspect test-data	

**To run a container with the volume created:**

docker container run --name test -it  -v test-data:/data alpine sh

ls

cd data

echo ""some data"" > file1.txt

echo ""some more data"" > file2.txt

exit

**To remove a container:**

docker container rm test	

**To run a centos container with volume:**

docker container run --name test2 -it -v test-data:/data centos sh

ls

cd data

You can still find out file1.txt and file2.txt. Docker volume are able to presist the data beyond the lifetime of the container.

**To run alpine container:**

Create a volume and bind it to the container

docker container run -it --name writer -v shared-space:/app/data alpine sh

echo ""add some file"" > /app/data/file1.txt

echo ""add some more file"" > /app/data/file2.txt

ls

cd app/data

exit

**To run a container in a read only mode:**

docker container run -it --name reader -v shared-space:/app/data:ro ubuntu:17.04 sh

ls

cd app/data

echo ""try to add a new file""> file3.txt

Since the volume is in read only mode, we can't create file3.txt

**To pull mongo image:**

docker image pull mongo:3.7	

**To inspect the image:**

docker image inspect --format='{{json .ContainerConfig.Volumes}}' mongo:3.7	

**To run the container:**

docker run --name mongodb -d mongo:3.7	

**To inspect the container:**

docker inspect --format='{{json .Mounts}}' mongodb	

**To know the resource consumptions:**

docker system df	

**To get the live data stream for running containers:**

docker stats

**To create a volume of specified size:**

docker volume create writer1 -o size=500MB	

Error response from daemon: create writer1: quota size requested but no quota support. We can't do that.

**To create volumes with drivers:**

docker volume create --driver local --opt type=tmpfs --opt device=tmpfs --opt o=size=100m,uid=1000 testvol1	

docker volume create --driver local --opt type=tmpfs --opt device=tmpfs --opt o=size=2GB,uid=1000 testvol2	

**To clean all stopped container:**

docker container prune	

**To remove the running containers:**

docker container prune -f	

**To clearn all containers all at once:**

docker container rm -f $(docker container ls -aq)

**You can check the spaces:**

docker system df	

**Clear images:**

docker image prune	

**Clear Volumes:**

docker volume prune	

**To remove everything (images, volumes, containers):**

docker system prune	

**Only remove dangling images:**

docker image prune --all	

**To know what is happening in the docker:**

docker system events

**To write it inside the container:**

docker container run --rm alpine echo "writing"	
