<h3> Docker Swarm Setup </h3>

**To initialize a swarm. It consists of single node:**

docker swarm init	

**To leave the swarm:**

docker swarm leave --force	

**To join the swarm:**

docker swarm join --token SWMTKN-1-2yvjyjowrc7hvl6uscxsyskq0sxfgivrr66824554pzjddo811-aen2um1ubg5zisdzx7xoh2m9i 192.168.65.3:2377	

**To make the node as a manager:**

docker swarm join-token manager	

**To know the details of the swamrm like status, ID:**

docker node ls	 

.* represents that this is the node the docker has executed. Copy the ID

**To inspect the node:**

docker node inspect m1utad9hd9e9ojes4yqkv3a1o	


<h3> With the help of "Play with docker" you can create nodes in swarm: https://labs.play-with-docker.com/ </h3>

**To make a swarm as a leader:** 

Make sure you give this command in node 1

docker swarm init --advertise-addr=eth0  

**To know the details of the swarm:**

docker node ls

**To promote the node 2 and node 3:**

docker node promote node2 node3

docker node ls

<h3> Docker Swarm Stack </h3>

**To create a docker swarm stack:**

docker stack deploy -c servicedemo1.yml sample-stack

**To list of the networks:**

docker network ls

**To show the name of the stack, services and orchestrator:**

docker stack ls	

**To show the replicas, id , mode:**

docker service ls	

**To inspect a particular stack:**

docker service ps sample-stack_newapp	

**To remove the container:**

docker container rm 3b1eff864dc4  -f

Delete the containers and check. It will recreate the container again. Since we have used replicas. You can see 6/6 containers

docker service ls

**To get the service logs:**

docker service logs sample-stack_newapp	

docker service logs 55yg5eqrl7i6	

**To remove the docker swarm stack:**

docker stack rm sample-stack_newapp

<h3> Rolling updates </h3>

**To create the rolling update:**

docker stack deploy -c rollingupdates.yaml web	

**To view the details of the rolling updates:**
docker stack ps web	

**To watch what is happening behind the updates:**

For Linux:
watch docker stack ps web

For Windows in powershell:
while (1) {docker stack ps web; sleep 5} 

**To update the service:**

docker service update --image nginx:1.13-alpine web_web	

<h3>Dealing with secret file</h3>

**To write some message in the secret file:**

echo  "some message " > ./my-secrets/secret-value.txt 	

**To create a secret file:**

docker secret create my-secret ./my-secrets/secret-value.txt	

**To list of the secret file:**

docker secret ls	

**To inspect the secret file:**

docker secret inspect my-secret	

**To converge the service:**

docker service create --name web --secret my-secret --publish 9000:9000 fundamentalsofdocker/whoami	
