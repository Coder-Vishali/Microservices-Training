Go to the path where Dockerfile is present:

**To build the image from Dockerfile:**

docker image build -t new-python-app .	

If you face i/o error, to resolve it set the builtkit = false in docker setting

""experimental"": false,
  
  ""features"": {
    
    ""buildkit"": false
  
  },""
  
**To view all the docker images:**

docker images	

**To create container from that image:**

docker run --name new-python-app-cont -p 4005:4005 new-python-app	http://localhost:4005/welcome

**To run the container in shell mode:**

docker run --name new-python-app-cont -it new-python-app sh

curl localhost:4005/welcome

docker exec -it new-python-app-cont sh

curl localhost:4005/welcome

**To get the container logs:**

docker container logs container_ID

**To add the arugment varaibles:**
  
ARG variable_name=values
 
This is to add environment variable like proxy variable
  
docker build —build-arg variable_name=a_value -t tag name .

**Attach command:**

docker attach container_id

**Centos-Dockerfile:**

docker image build -t my-centos .	

**To create container:**

docker run -it --name my-centos-cont my-centos bash

**Inside the centos container, execute the below commands:**

wget

wget https://jsonplaceholder.typicode.com/posts

exit	

**To login to docker hub:**

docker login	

**Provide tag for the image created:**

docker image tag new-python-app:latest vishali007/new-python-app:1.0

**To push the image into the registry:**

docker image push docker_user_name/new-python-app:1.0	

**To pull from registry:**

docker pull docker_user_name/new-python-app:1.0	

**To echo some message:**

docker run --name my_container_alpine alpine sh -c 'echo "sample text " > a.txt'

**To find the difference:**

docker container diff my_container_alpine	
