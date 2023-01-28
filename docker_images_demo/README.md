**Pulling hello-world image:**

This image gets downloaded from docker hub.

docker run hello-world	

**To remove all the images:**

docker image prune -a	

**To display all the images which you have locally:**

docker images	

**Running simple docker container image:**

docker run alphine	

docker container run alpine

**To display all running containers:**

docker ps -a	

**Docker CLI:**

docker run alpine echo "My first statement"

**Self ping:**

docker run centos ping -c 5 127.0.0.1

**To run a container in a daemon mode:**

docker run -d --name jsonplaceholder3 alpine sh -c "while :; do wget -qO- https://jsonplaceholder.typicode.com/posts; printf  '\n'; sleep 5; done "

**To show the list of running containers:**

docker ps	

**To run inside the container in shell script mode:**

docker exec -it jsonplaceholder sh 

**Similar to docker ps -a command:**

docker container ls -a	

**To display the container ID for all:**

docker container ls -q	

**To remove all the containers:**
docker container rm -f $(docker container ls -a -q)	

**To stop a particular container:**

docker stop jsonplaceholder

docker ps -a 

To stop the container. In backend, docker will send a signal to linux so it waits for 10 secs and then kills and terminates the container.

**To inspect the containers:**
docker container inspect jsonplaceholder	

**Attach command:**

docker attach jsonplaceholder	

You can see the json responses for each 5 secs

**Port container mapping:**

docker run -d --name nginx -p 9090:80 nginx:alpine	

docker exec -it nginx sh	"curl -5 localhost:80

curl localhost:80

You can see the html of the same page.

**To remove the container:**

docker container rm nginx

**To get the logs statement:**

docker container logs nginx	

**To get the last 5 logs:**

docker container logs --tail 5 nginx

**To run alphine container in iterative mode: Interative image creation**

docker run -it --name iter_image alpine sh

**Inside the container, execute the below commands:**

ls     # show what's inside the container

apk update  # package manager

apk add iputils # adding a utilies

ping 127.0.0.1

ip address

exit

**Grep command:**

To perform text searches for a defined criteria of words or strings. grep stands for Globally search for a Regular Expression and Print it out.

docker container ls -a | grep iter_image	

**To show what got change:**

docker container diff iter_image

**To create a new image & committed locally:**

docker container commit iter_image new-alphine	

You can check for new alpine image:

docker images	


**To copy the contents from container to local PC:**

docker cp <containerId>:/file/path/within/container /host/path/target

eg: docker cp d6f53729d287:/var/jenkins_home/workspace/Job_1 .  
