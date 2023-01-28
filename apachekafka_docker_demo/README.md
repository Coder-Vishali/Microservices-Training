Make sure your go to the path where .yml file is present and excute the below command

**To start the kafka:**

docker-compose up	

**You can notice that kafka and zookeeper are running:**

docker-compose ps	

**To create topic mytopic (Open another terminal and run the below command):**

docker-compose exec kafka kafka-topics --create --topic mytopic --bootstrap-server broker:9092	

**Listing the listerer:**

docker-compose exec kafka bash

kafka-console-consumer --topic mytopic --bootstrap-server broker:9092

**Start another process inside the docker container from another command prompt & you can start sending messages:**

docker-compose exec kafka bash

kafka-console-producer --topic mytopic --bootstrap-server broker:9092

**You can start another consumer. Both will get the messages:**

docker-compose exec kafka bash

kafka-console-consumer --topic mytopic --bootstrap-server broker:9092	
