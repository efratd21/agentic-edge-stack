# Bonus 3 — Qdrant on Kubernetes

Swap the in-memory FAISS index (Parts 2–3) for a **standalone Qdrant vector
database** running on a local Kubernetes cluster. The RAG code
([`rag_qdrant.py`](rag_qdrant.py)) is a drop-in replacement for
[`src/rag.py`](../../src/rag.py): identical corpus, chunking and embeddings;
only the vector store moves out of the process and into the cluster.

## What's here

| File | Purpose |
|---|---|
| [`00-namespace.yaml`](00-namespace.yaml) | `agentic-edge` namespace |
| [`05-qdrant-pvc.yaml`](05-qdrant-pvc.yaml) | PersistentVolumeClaim so the vector data survives pod restarts |
| [`10-qdrant-deployment.yaml`](10-qdrant-deployment.yaml) | Qdrant Deployment (1 replica) mounting the PVC |
| [`20-qdrant-service.yaml`](20-qdrant-service.yaml) | ClusterIP service `qdrant` (ports 6333 REST / 6334 gRPC) |
| [`rag_qdrant.py`](rag_qdrant.py) | Qdrant-backed RAG: build the collection + query it |

A verbatim capture of the full lifecycle (apply → rollout → build → query →
pod-restart persistence check) is committed at
[`logs/qdrant_retrieval.log`](../../logs/qdrant_retrieval.log).

## Cluster

Any local Kubernetes works (k3s / kind / minikube). These steps use
**Docker Desktop's built-in Kubernetes** (Settings → Kubernetes → *Enable
Kubernetes*); on my machine its images were already cached locally, which
mattered on a restricted network (see the note below).

> **Image note:** the manifest uses `imagePullPolicy: IfNotPresent` with the
> `qdrant/qdrant:latest` tag. On your machine the image is simply pulled once on
> first use (~185 MB). The policy exists because my development network blocked
> the Docker Hub CDN, so the pod had to start from the locally cached image; in
> production you would pin an explicit version tag (e.g. `qdrant/qdrant:v1.12.4`).

## Deploy

```bash
# 1. Confirm the cluster is up
kubectl get nodes

# 2. Apply the manifests (namespace → pvc → deployment → service)
kubectl apply -f bonus/k8s/

# 3. Wait for Qdrant to become ready
kubectl -n agentic-edge rollout status deployment/qdrant --timeout=120s
kubectl -n agentic-edge get pods,svc,pvc
```

## Point the RAG app at the cluster

The RAG app runs on the host, so forward the service port to localhost:

```bash
kubectl -n agentic-edge port-forward svc/qdrant 6333:6333 6334:6334
```

`settings.qdrant_url` already defaults to `http://localhost:6333`
(override with the `QDRANT_URL` env var / `.env` if needed).

## Build the collection and query it

In a second terminal (with the port-forward running):

```bash
# Load → chunk → embed the corpus and upsert it into Qdrant
python bonus/k8s/rag_qdrant.py --build

# Retrieve top-3 chunks for a query (from Qdrant, not FAISS)
python bonus/k8s/rag_qdrant.py "how does an agent decide to use a tool?"
```

Requires Ollama running for the `nomic-embed-text` embeddings, exactly like the
FAISS path.

## See it for yourself

While the port-forward is open:

- **Qdrant dashboard:** open <http://localhost:6333/dashboard> in a browser.
  The `agent_corpus` collection holds 20 points; click into it to browse the
  stored chunks (text, source file and section live in each point's payload).
- **REST check:**

  ```bash
  curl -s localhost:6333/collections/agent_corpus | python3 -m json.tool
  # → "status": "green", "points_count": 20
  ```

- **Persistence:** delete the pod and watch the data survive the restart. The
  Deployment reschedules a fresh pod, and the collection answers with **no
  rebuild**, because the vectors live on the PersistentVolumeClaim:

  ```bash
  kubectl -n agentic-edge delete pod -l app=qdrant
  kubectl -n agentic-edge rollout status deployment/qdrant --timeout=90s
  python bonus/k8s/rag_qdrant.py "what is the ReAct pattern?"   # note: no --build
  ```

## Teardown

```bash
kubectl delete -f bonus/k8s/          # or: kubectl delete ns agentic-edge
```
