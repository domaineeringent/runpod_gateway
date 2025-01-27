# RunPod Gateway

Simple gateway to start/check RunPod instances. Waits for pods to be fully running before responding.

## Quick Start

1. Install:
```bash
pip install fastapi uvicorn aiohttp python-multipart requests python-dotenv
```

2. Set up `.env`:
```env
RUNPOD_API_KEY=your_api_key_here
```

3. Run:
```bash
python gateway.py
```

## API Endpoints

### Start Pod
```bash
POST /api/start/{pod_id}
```
- Starts pod if stopped
- Waits for "RUNNING" state
- Optional health check URL
- Returns when pod is ready

Example:
```python
# Basic start
response = requests.post("http://localhost:8080/api/start/your-pod-id")

# With health check
response = requests.post(
    "http://localhost:8080/api/start/your-pod-id",
    params={"health_url": "http://your-pod-ip:8000/health"}
)
```

### Check Status
```bash
GET /api/status/{pod_id}
```
- Gets current pod status
- Checks if pod is running
- Optional health check

Example:
```python
response = requests.get("http://localhost:8080/api/status/your-pod-id")
```

## Response States

1. Pod Starting:
```json
{
    "status": "starting",
    "message": "Pod is starting but not ready yet"
}
```

2. Pod Ready:
```json
{
    "status": "ready",
    "message": "Pod started and ready"
}
```

3. Status Check:
```json
{
    "pod_status": "RUNNING",
    "ready": true,
    "pod_id": "your-pod-id"
}
```

## Important Notes

- Gateway waits up to 2 minutes for pod to be ready
- Concurrent requests handled automatically by FastAPI
- No queueing/forwarding - just starts pods and checks status
- Optional health check URL for application readiness
- Returns error if pod fails to start

## Error Handling

- 404: Pod not found
- 500: Failed to start pod or other errors
- 503: Pod not ready yet 
