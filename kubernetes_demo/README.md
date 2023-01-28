<h3> Kubernetes </h3>

**To know the version of kubectl:**

kubectl version --short	

kubectl version	

If you face the issue ""Unable to connect to the server: net/http: TLS handshake timeout"

The problem is that Docker ran out of memory. To fix:

Fully close your k8s emulator. (minikube, docker-desktop, etc.)
Shutdown WSL2. (wsl --shutdown)
Restart your k8s emulator.

**Basic kubectl command:**

kubectl [command] [type] [name] [flags]	

**To get the kubernetes nodes:**

kubectl get nodes	

**To get the cluster details:**

kubectl get cs

**To setup the Linux Cluster:**

kubeadm

**To run the kubernetes:** 

kubectl run kubenginx --image=nginx --port=80	

**To get pods:**

kubectl get pods	

**Deploying kubernetes:**

kubectl get deployment	

**To create service:**

kubectl expose pod kubenginx --port=80 --type=LoadBalancer

**To get the service:**

kubectl get service	

**To describe the service:**

kubectl describe service kubenginx	


**Demo 1:**

kubectl apply -f demo1.yaml

kubectl get deployment

**Demo 2:**

kubectl apply -f demo2.yaml

kubectl get deployment
