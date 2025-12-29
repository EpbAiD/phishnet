# Docker Deployment Guide

Complete guide for deploying the Phishing Detection API using Docker.

## Quick Start

### 1. Build and Run with Docker Compose (Recommended)

```bash
# Build and start the container
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

The API will be available at `http://localhost:8000`

### 2. Build and Run with Docker (Manual)

```bash
# Build the image
docker build -t phishing-detection-api .

# Run the container
docker run -d \
  --name phishing-api \
  -p 8000:8000 \
  --restart unless-stopped \
  phishing-detection-api

# View logs
docker logs -f phishing-api

# Stop the container
docker stop phishing-api
docker rm phishing-api
```

## Testing the Deployed API

### Health Check

```bash
curl http://localhost:8000/health
```

### Test Ensemble Prediction

```bash
curl -X POST http://localhost:8000/predict/ensemble \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.github.com"}'
```

### Test Dual Explanations

```bash
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{"url": "http://secure-login-apple.com-verify.tk/account", "include_shap": true}'
```

## API Endpoints

Once deployed, the following endpoints are available:

- **GET `/health`** - Health check
- **POST `/predict/url`** - URL-only prediction
- **POST `/predict/whois`** - WHOIS-only prediction
- **POST `/predict/dns`** - DNS-only prediction
- **POST `/predict/ensemble`** - Combined prediction (URL + WHOIS)
- **POST `/explain`** - Full dual explanation with SHAP analysis

## Environment Variables

You can customize the deployment with environment variables:

```yaml
# In docker-compose.yml
environment:
  - PYTHONUNBUFFERED=1
  - LOG_LEVEL=info  # Options: debug, info, warning, error
```

## Production Deployment

### AWS ECS / Fargate

```bash
# Tag and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

docker tag phishing-detection-api:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/phishing-api:latest

docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/phishing-api:latest
```

### Google Cloud Run

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/<project-id>/phishing-api

# Deploy to Cloud Run
gcloud run deploy phishing-api \
  --image gcr.io/<project-id>/phishing-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8000
```

### Azure Container Instances

```bash
# Build and push to ACR
az acr build --registry <registry-name> --image phishing-api:latest .

# Deploy to ACI
az container create \
  --resource-group <resource-group> \
  --name phishing-api \
  --image <registry-name>.azurecr.io/phishing-api:latest \
  --dns-name-label phishing-api \
  --ports 8000
```

### Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: phishing-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: phishing-api
  template:
    metadata:
      labels:
        app: phishing-api
    spec:
      containers:
      - name: phishing-api
        image: phishing-detection-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: phishing-api
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
  selector:
    app: phishing-api
```

Deploy with:
```bash
kubectl apply -f deployment.yaml
```

## Scaling & Performance

### Resource Requirements

Recommended resource allocation:
- **CPU**: 1 vCPU minimum (2 vCPU for high traffic)
- **Memory**: 1GB minimum (2GB recommended)
- **Storage**: 1GB for models and logs

### Horizontal Scaling

For high-traffic scenarios, deploy multiple replicas behind a load balancer:

```bash
# Docker Swarm
docker service create \
  --name phishing-api \
  --replicas 3 \
  --publish 8000:8000 \
  phishing-detection-api

# Kubernetes
kubectl scale deployment phishing-api --replicas=5
```

### Performance Optimization

1. **Enable CPU optimizations**:
   ```dockerfile
   ENV OMP_NUM_THREADS=4
   ENV MKL_NUM_THREADS=4
   ```

2. **Use multi-worker uvicorn** (modify CMD in Dockerfile):
   ```dockerfile
   CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
   ```

3. **Add caching layer** (Redis) for frequently requested URLs

## Monitoring & Logging

### View Logs

```bash
# Docker Compose
docker-compose logs -f phishing-api

# Docker
docker logs -f phishing-api

# Kubernetes
kubectl logs -f deployment/phishing-api
```

### Health Monitoring

The `/health` endpoint returns:
```json
{
  "status": "healthy",
  "models": {
    "url": "loaded",
    "whois": "loaded"
  }
}
```

Set up monitoring alerts based on:
- Health check failures
- High response latency (>2s)
- Error rate (>1%)

## Security Considerations

### Production Hardening

1. **Run as non-root user** (add to Dockerfile):
   ```dockerfile
   RUN useradd -m -u 1000 apiuser
   USER apiuser
   ```

2. **Enable HTTPS** (use reverse proxy):
   ```bash
   # Nginx reverse proxy
   server {
       listen 443 ssl;
       server_name api.example.com;

       location / {
           proxy_pass http://localhost:8000;
       }
   }
   ```

3. **Rate limiting** (add to FastAPI):
   ```python
   from slowapi import Limiter
   limiter = Limiter(key_func=get_remote_address)
   app.state.limiter = limiter
   ```

4. **API authentication** (add to endpoints):
   ```python
   from fastapi.security import APIKeyHeader
   api_key_header = APIKeyHeader(name="X-API-Key")
   ```

## Troubleshooting

### Container won't start

```bash
# Check logs
docker logs phishing-api

# Common issues:
# - Missing models/ directory
# - Port 8000 already in use
# - Insufficient memory
```

### High latency

- Check if WHOIS lookups are timing out
- Consider caching frequently analyzed URLs
- Scale horizontally with load balancer

### Out of memory

- Increase Docker memory limit:
  ```bash
  docker run --memory="2g" phishing-detection-api
  ```

## Updating the Deployment

```bash
# Rebuild with latest changes
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Or with Docker directly
docker stop phishing-api
docker rm phishing-api
docker build -t phishing-detection-api .
docker run -d --name phishing-api -p 8000:8000 phishing-detection-api
```

## Backup & Recovery

### Backup Models

```bash
# Create backup
docker cp phishing-api:/app/models ./models_backup

# Restore backup
docker cp ./models_backup phishing-api:/app/models
docker restart phishing-api
```

### Export Logs

```bash
docker logs phishing-api > phishing-api-logs-$(date +%Y%m%d).txt
```

## Load Testing

Test the deployed API with load testing tools:

```bash
# Using Apache Bench
ab -n 1000 -c 10 -p test_payload.json -T application/json http://localhost:8000/predict/ensemble

# Using wrk
wrk -t4 -c100 -d30s -s post.lua http://localhost:8000/predict/ensemble
```

## Support

For issues or questions:
- Check logs: `docker logs -f phishing-api`
- Verify health: `curl http://localhost:8000/health`
- Review documentation: [README_PIPELINE.md](README_PIPELINE.md)
