# RunPod Gateway

A simple gateway service that manages RunPod instances, automatically starting them when needed and handling request queuing. Perfect for cost optimization by keeping pods stopped when not in use.

## Features

- 🚀 Automatically starts stopped RunPod instances when requests arrive
- 📊 Queues requests during pod startup
- ♻️ Retries failed requests
- 🔄 Supports multiple pods
- 💰 Cost optimization by using stopped pods

## Quick Start

1. Install dependencies:
```bash
pip install fastapi uvicorn aiohttp python-multipart requests python-dotenv
```

2. Create `.env` file:
```env
RUNPOD_API_KEY=your_api_key_here
```

3. Run the gateway:
```bash
python gateway.py
```

## Usage

1. Get your RunPod ID:
   - Go to RunPod dashboard
   - Find your pod
   - Copy the Pod ID (even if it's stopped)

2. Send requests to the gateway:
```python
import requests

# Process audio file
response = requests.post(
    "http://localhost:8080/api/process/your-pod-id",
    files={"file": open("audio.mp3", "rb")}
)

# Check status
status = requests.get(
    "http://localhost:8080/api/status/your-pod-id"
)
```

## API Endpoints

- `POST /api/process/{pod_id}`
  - Processes file through specified pod
  - Returns 202 Accepted with queue position
  - Automatically starts pod if stopped

- `GET /api/status/{pod_id}`
  - Get pod status and queue information

## Configuration

- `MAX_RETRIES`: Number of retry attempts (default: 3)
- `RETRY_DELAY`: Seconds between retries (default: 5)
- `MAX_QUEUE_SIZE`: Maximum queued requests per pod (default: 50)

## Integration Example

```python
# Example integration with your application
async def process_audio(file_path: str, pod_id: str):
    # Prepare file
    files = {
        "file": ("audio.mp3", open(file_path, "rb"), "audio/mpeg")
    }
    
    # Send to gateway
    response = requests.post(
        f"http://localhost:8080/api/process/{pod_id}",
        files=files
    )
    
    if response.status_code == 202:
        # Request queued
        queue_info = response.json()
        print(f"Request queued at position {queue_info['position']}")
    else:
        print(f"Error: {response.text}")
```

## Health Checking

The gateway ensures pods are ready by:
1. Verifying RunPod status is "RUNNING"
2. Checking the pod's `/health` endpoint
3. Accepting requests even during model loading

## Error Handling

- Retries on server errors (5xx)
- Skips retry on client errors (4xx)
- Logs all attempts and failures
- Queues requests during pod startup

## Cost Optimization

This gateway helps optimize costs by:
1. Using stopped pods (which don't incur charges)
2. Starting pods only when needed
3. Allowing your app to use cheaper stopped instances

## Requirements

- Python 3.7+
- RunPod API Key
- Existing RunPod instance (can be stopped)

## Security Note

Keep your RunPod API key secure and never expose it in client-side code. 
