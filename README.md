# mlops-pytorch-pipeline

The goal of this assignment was to take a basic image classification model and push it all the way through a production-style workflow—from writing the training/serving scripts, containerizing them, deploying them to a local k8s cluster with GPU support, and setting up autoscaling.

## Architecture Diagram

```text
+-------------------------------------------------------+
|                    Local Workspace                    |
|   (PyTorch Code, Dataset, Dockerfiles, K8s Manifests) |
+-------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------+
|                 Docker Containerization               |
|          (Multi-stage builds for Train & Serve)       |
+-------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------+
|                 Kubernetes Cluster                    |
|  +---------------------------+  +------------------+  |
|  |     Training Job (GPU)    |  | Serving + HPA    |  |
|  +---------------------------+  +------------------+  |
+-------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------+
|                    Inference API                      |
|            (FastAPI Endpoint via Port-Forward)        |
+-------------------------------------------------------+
```

## Contents
- **`src/`**: All the core python stuff—model definition, dataset processing, training loop, and the FastAPI serving app.
- **`docker/`**: Separate Dockerfiles for training and serving so everything stays modular and clean.
- **`k8s/`**: All the kubernetes YAML files (namespace, configmap, training job, deployment, service, and HPA).
- **`requirements/`**: Pip requirements split up for training vs serving.

## How to run it

### 1. Build the Docker images
First, build the images locally:
```bash
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
```

### 2. Deploy to Kubernetes
Apply the manifests to spin everything up in the cluster:
```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/training-job.yaml
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml
```

### 3. Check status & test the model
To check if pods and jobs are running:
```bash
kubectl get all -n ml-training
```

To test out predictions, port-forward the service:
```bash
kubectl port-forward svc/model-serving 8080:80 -n ml-training
```