# Kubernetes deployment

The **same manifests** run on a local cluster (kind/minikube) and in the cloud
(GKE/EKS). Only the storage class and Service type differ for cloud (noted below).

## Local (kind)

```bash
# 1. cluster
kind create cluster --name vol

# 2. build the image and load it into the cluster (no registry needed)
docker build -t volatility-forecaster:latest .
kind load docker-image volatility-forecaster:latest --name vol

# 3. apply everything
kubectl apply -k k8s/

# 4. wait + smoke test
kubectl rollout status deploy/volatility-api
kubectl port-forward svc/volatility-api 8000:80
curl localhost:8000/health
curl -XPOST localhost:8000/predict -H 'content-type: application/json' -d '{"ticker":"RELIANCE.NS"}'

# 5. trigger the nightly batch on demand
kubectl create job --from=cronjob/volatility-nightly nightly-manual
kubectl logs job/nightly-manual -f
```

`minikube` is the same, but load the image with
`minikube image load volatility-forecaster:latest`.

## Components
| manifest | purpose |
|---|---|
| `configmap.yaml` / `secret.yaml` | env config + placeholder secret |
| `pvc.yaml` | shared feature/artifact store (seeded from the image) |
| `redis.yaml` | in-cluster Redis (Deployment + Service) |
| `deployment.yaml` | API (2 replicas, `initContainer` seeds PVC, `/health` probes) |
| `service.yaml` | ClusterIP (+ Prometheus scrape annotations) |
| `hpa.yaml` | HorizontalPodAutoscaler, CPU 70%, 2–6 replicas |
| `cronjob.yaml` | nightly post-close: ingest → build → precompute → monitoring |

## HPA note
`kubectl autoscale` needs the metrics-server. On kind/minikube enable it:
`minikube addons enable metrics-server` (minikube) or install metrics-server
(kind). Generate load to watch it scale:
`kubectl run -it load --rm --image=busybox -- /bin/sh -c "while true; do wget -q -O- http://volatility-api/predict --post-data='{\"ticker\":\"INFY.NS\"}' --header='content-type: application/json'; done"`

## Cloud (GKE/EKS) deltas
- **PVC**: switch `accessModes` to `ReadWriteMany` (GKE Filestore / EFS) so 2–6
  API replicas across nodes can share the feature store. (kind is single-node so
  `ReadWriteOnce` is fine locally.)
- **Service**: `type: LoadBalancer` (or an Ingress) for external access.
- **Image**: push to a registry (`gcr.io/...`, ECR) and set `imagePullPolicy: Always`.
