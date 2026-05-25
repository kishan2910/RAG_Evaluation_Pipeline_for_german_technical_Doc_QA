# Production Deployment Guide

## Overview

This guide covers deploying the IT Support RAG System to production with emphasis on reliability, observability, and performance.

## Pre-Deployment Checklist

- [ ] All tests passing locally
- [ ] Docker images built and tested
- [ ] Environment variables configured
- [ ] Database backups strategy defined
- [ ] Monitoring and alerting configured
- [ ] Documentation updated
- [ ] Security review completed

## Environment Setup

### 1. Server Requirements

**Minimum Specification:**
- CPU: 4-core processor
- RAM: 8GB (16GB recommended)
- GPU: Optional but recommended (4GB+ VRAM for Mistral)
- Storage: 50GB (for models, data, logs)
- Network: Stable internet (for model downloads)

**Recommended OS:**
- Ubuntu 20.04 LTS or later
- CentOS 8 or later
- Any OS with Docker support

### 2. Pre-Install Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install Python 3.13+
sudo apt install -y python3.13 python3.13-venv

# Verify installations
docker --version
docker-compose --version
python3 --version
```

### 3. Clone & Setup Project

```bash
# Clone repository
git clone <repository-url>
cd IT-Support-Enterprise-Helpdesk-RAG-System

# Create Python environment
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -e .

# Install uv for faster package management
pip install uv
uv sync
```

### 4. Configure Environment

Create `.env` file in project root:

```bash
# Ollama Configuration
OLLAMA_HOST=http://ollama:11434
OLLAMA_MODEL=mistral

# ChromaDB Configuration
CHROMADB_HOST=chromadb
CHROMADB_PORT=8000

# Embeddings
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Dataset
HUGGINGFACE_DATASET=rojagtap/tech-qa

# Application Settings
LOG_LEVEL=INFO
```

## Docker Deployment

### 1. Build Images

```bash
# Build using provided docker-compose
docker-compose build

# Verify images built
docker images | grep rag
```

### 2. Start Services

```bash
# Start all services in detached mode
docker-compose up -d

# Verify services are running
docker-compose ps

# View logs
docker-compose logs -f

# Check specific service
docker-compose logs ollama
```

### 3. Initialize System

```bash
# Wait for services to be ready (30-60 seconds)
sleep 60

# Load dataset
docker-compose exec app python scripts/load_dataset.py

# Run health check
curl http://localhost:6000/health
```

## Production Considerations

### 1. Security Hardening

**API Security:**
```python
# In app/main.py - Restrict CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Restrict origin
    allow_methods=["POST", "GET"],  # Restrict methods
    allow_headers=["Content-Type", "Authorization"],
)
```

**Add Authentication:**
```python
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/query")
async def query(request: QueryRequest, credentials = Depends(security)):
    # Verify API key
    if credentials.credentials != os.getenv("API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid API key")
    # ... process query
```

**Database Access:**
```bash
# Bind ChromaDB to localhost only
export CHROMADB_HOST=127.0.0.1

# Use firewall rules
sudo ufw allow 22/tcp    # SSH only
sudo ufw allow 6000/tcp  # API only from app
sudo ufw allow 8000/tcp  # ChromaDB from localhost
```

### 2. Performance Optimization

**API Server:**
```bash
# Use Gunicorn for production
pip install gunicorn

gunicorn app.main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:6000 \
    --access-logfile - \
    --error-logfile - \
    --log-level info
```

**Caching:**
```bash
# Add Redis for embedding caching
docker-compose up -d redis

# In app/services/vector_db.py
import redis
cache = redis.Redis(host='redis', port=6379)
```

**Load Balancing:**
```bash
# Use Nginx to balance multiple API instances
# See nginx.conf.example for configuration
docker-compose up -d nginx
```

### 3. Monitoring & Logging

**Application Metrics:**
```bash
# Install Prometheus client
pip install prometheus-client

# Export metrics endpoint
# GET http://localhost:6000/metrics
```

**Log Aggregation:**
```bash
# Add ELK stack for log monitoring
# elasticsearch, logstash, kibana services in docker-compose
docker-compose up -d elasticsearch logstash kibana
```

**Alert Configuration:**
```yaml
# prometheus-rules.yml
groups:
  - name: rag_alerts
    rules:
      - alert: HighLatency
        expr: avg(response_time_ms) > 5000
        for: 5m
        annotations:
          summary: "High query latency detected"
      
      - alert: HighHallucinationRate
        expr: hallucination_rate > 0.3
        for: 15m
        annotations:
          summary: "High hallucination rate detected"
```

### 4. Data Backup & Recovery

**Backup Strategy:**

```bash
#!/bin/bash
# backup.sh - Daily backup script

BACKUP_DIR="/backups/rag-system"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Backup ChromaDB data
docker exec chromadb tar czf - /data | gzip > $BACKUP_DIR/chromadb_$TIMESTAMP.tar.gz

# Backup logs
tar czf $BACKUP_DIR/logs_$TIMESTAMP.tar.gz logs/

# Upload to S3
aws s3 cp $BACKUP_DIR/ s3://backup-bucket/rag-system/ --recursive

# Keep only last 30 days
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
```

**Recovery Procedure:**

```bash
# Restore ChromaDB from backup
docker-compose down
tar xzf /backups/chromadb_20240115_100000.tar.gz -C chromadb-data/
docker-compose up -d

# Verify recovery
curl http://localhost:6000/stats
```

### 5. Scaling Strategy

**Horizontal Scaling:**
```yaml
# docker-compose.yml - Multiple API instances
services:
  api:
    image: rag-api:latest
    environment:
      - PYTHONUNBUFFERED=1
    deploy:
      replicas: 3
    depends_on:
      - chromadb
      - ollama
  
  nginx:
    image: nginx:latest
    ports:
      - "6000:6000"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - api
```

**Vertical Scaling:**
- Increase GPU memory for faster generation
- Add CPU cores for parallel requests
- Increase RAM for embedding cache

### 6. Database Maintenance

```bash
# Periodic collection cleanup
docker-compose exec app python -c "
from app.services.vector_db import VectorDBService
db = VectorDBService()

# Get collection stats
info = db.get_collection_info()
print(f'Collection: {info[\"collection_name\"]}')
print(f'Documents: {info[\"document_count\"]}')

# Optional: Clear and reload
# db.clear_collection()
"

# Optimize ChromaDB
docker-compose exec chromadb chromadb compact
```

## Deployment Workflow

### 1. Staging Deployment

```bash
# Deploy to staging environment first
git checkout staging
git pull

docker-compose -f docker-compose.staging.yml build
docker-compose -f docker-compose.staging.yml up -d

# Run integration tests
python -m pytest tests/integration/

# Performance testing
python scripts/benchmark.py

# User acceptance testing
# ... manual testing
```

### 2. Production Deployment

```bash
# Blue-Green Deployment Strategy

# Step 1: Deploy new version to "green" environment
export ENV=green
docker-compose up -d

# Step 2: Run smoke tests
curl http://localhost:6000/health

# Step 3: Switch traffic
# Update load balancer/nginx to route to green

# Step 4: Keep blue running for rollback
# If issues detected, switch back to blue
```

### 3. Rollback Procedure

```bash
# If production deployment fails:

# 1. Detect issue
curl http://localhost:6000/health  # Check status

# 2. View recent logs
docker-compose logs --tail 100

# 3. Rollback to previous version
git revert HEAD
docker-compose up -d

# 4. Verify
curl http://localhost:6000/health

# 5. Investigate issue
grep ERROR logs/app.log | tail -20
```

## Monitoring Dashboards

### Key Metrics to Track

**Availability:**
- API uptime (target: 99.9%)
- Service status (Ollama, ChromaDB)

**Performance:**
- Query latency (p50, p95, p99)
- Retrieval time vs. generation time
- Throughput (requests/min)

**Quality:**
- Average retrieval score
- Hallucination rate
- Success rate

**Resources:**
- CPU usage
- Memory usage
- GPU memory usage
- Disk space

### Sample Grafana Dashboard Query

```promql
# Query latency over time
rate(query_time_ms_sum[5m]) / rate(query_time_ms_count[5m])

# Error rate
rate(failed_requests_total[5m]) / rate(total_requests[5m])

# Hallucination trend
rate(hallucinated_responses_total[1h])
```

## Troubleshooting Common Issues

### Issue: High Latency

**Symptoms:** Query response times > 10 seconds

**Solutions:**
1. Check GPU memory: `nvidia-smi`
2. Check system load: `top`
3. Reduce n_results in query
4. Use smaller embedding model
5. Add more workers/instances

### Issue: Memory Leaks

**Symptoms:** Memory usage grows over time

**Solutions:**
1. Monitor: `docker stats`
2. Check for unclosed connections
3. Restart container periodically
4. Profile memory: `memory_profiler`

### Issue: ChromaDB Disconnections

**Symptoms:** Connection refused to 127.0.0.1:8000

**Solutions:**
1. Check container: `docker logs chromadb`
2. Restart: `docker-compose restart chromadb`
3. Check logs: `docker-compose logs chromadb`
4. Verify network: `docker network inspect rag-system_default`

## Maintenance Schedule

| Task | Frequency | Duration |
|------|-----------|----------|
| Health check | Every 5 min | < 1 sec |
| Log rotation | Daily | < 5 min |
| Database backup | Daily | 10-30 min |
| Performance review | Weekly | 30 min |
| Security update | Monthly | 30-60 min |
| Full system test | Quarterly | 1-2 hours |

---

## Support & Incident Response

### Escalation Path
1. **Automated Alert** → Check `/health` and `/metrics`
2. **Level 1** → Review logs, restart service
3. **Level 2** → Check database, verify configuration
4. **Level 3** → Review code, analyze metrics
5. **Incident** → Page on-call engineer, begin investigation

### Quick Reference Commands

```bash
# Check system status
curl http://localhost:6000/health | jq .

# View logs (last 50 lines)
docker-compose logs --tail 50 app

# Restart specific service
docker-compose restart app

# Full restart
docker-compose down
docker-compose up -d

# Check database
docker-compose exec chromadb curl http://localhost:8000/api/v1/heartbeat

# Check Ollama
docker-compose exec ollama curl http://localhost:11434/api/tags
```

---

For operational runbooks, see `RUNBOOK.md`
