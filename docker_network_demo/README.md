**To know the network ID, name, driver and scope:**

docker network ls

**Inspect bridge network:**

docker inspect bridge	

**Inspect host network:**

docker inspect host

**Inspect null network:**

docker network inspect null	

**To create a driver:**

docker create --driver bridge demo-bridge-1	

**To list all the networks:**

docker network ls	

**To inspect the driver:**

docker inpsect demo-bridge-1	

**To create another driver:**

docker network create --driver --subnet "10.1.0.0./16" demo-bridge-2	

docker inpsect demo-bridge-2	

**To create a alpine container:**

docker container run --name mycont -it --rm alpine sh	

**To create bridge container:**

docker container inspect mycont	bridge container

docker run --name mycont2 -d --rm --network demo-bridge-2 alpine sh ping 127.0.0.1	

docker inspect demo-bridge-2	

docker run --name mycont5 -d --rm --network demo-bridge-2 alpine sh ping 127.0.0.1	

**You will see the containers using this network:**

docker netowrk inspect demo-bridge-2	

**To create other container in the same network:**

docker run --name mycont6 -d --rm --network demo-bridge-1 alpine sh ping 127.0.0.1	

docker exec -it mycont2 sh

**Inside the container, execute the following commands:**

ping mycont5

ping mycont6	

For cont6, you will face bad address since it is connected to the same network.
