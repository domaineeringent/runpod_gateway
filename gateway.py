import os
import time
import logging
import asyncio
import aiohttp
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
import requests
from typing import Optional, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = FastAPI(title="RunPod Gateway")

# Required Configuration
RUNPOD_API_KEY = os.getenv('RUNPOD_API_KEY')
if not RUNPOD_API_KEY:
    raise ValueError("Missing RUNPOD_API_KEY in .env file")

RUNPOD_API_BASE = "https://api.runpod.io/v2"
MAX_RETRIES = 3  # Maximum number of retries for failed requests
RETRY_DELAY = 5  # Seconds to wait between retries

# Cache pod information - now keyed by pod_id
pod_cache: Dict[str, Dict] = {}

# Queue per pod
request_queues: Dict[str, asyncio.Queue] = {}
MAX_QUEUE_SIZE = 50

async def get_queue_for_pod(pod_id: str) -> asyncio.Queue:
    """Get or create queue for specific pod."""
    if pod_id not in request_queues:
        request_queues[pod_id] = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
        # Start queue processor for this pod
        asyncio.create_task(process_queued_requests(pod_id))
    return request_queues[pod_id]

async def check_pod_health(pod_id: str) -> bool:
    """
    Check if the pod's API is accepting requests.
    Specifically checks our app's /health endpoint which tells us if models are loaded.
    """
    if pod_id not in pod_cache or not pod_cache[pod_id].get("endpoint_url"):
        return False
        
    try:
        async with aiohttp.ClientSession() as session:
            # First verify pod is actually running
            response = requests.get(
                f"{RUNPOD_API_BASE}/pod/{pod_id}",
                headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"}
            )
            if response.status_code != 200 or response.json()["status"] != "RUNNING":
                return False

            # Then check our app's health endpoint
            async with session.get(f"{pod_cache[pod_id]['endpoint_url']}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    # Our app accepts requests during "installing" state (models still loading)
                    # or "ready" state (fully loaded)
                    is_healthy = data["status"] in ["ready", "installing"]
                    logging.info(f"Pod {pod_id} health check: {data['status']}")
                    return is_healthy
    except Exception as e:
        logging.error(f"Health check failed for pod {pod_id}: {str(e)}")
        return False
    
    return False

async def ensure_pod_running(pod_id: str) -> tuple[bool, str]:
    """Ensures specific pod is running and ready to accept requests."""
    try:
        # Initialize pod cache entry if needed
        if pod_id not in pod_cache:
            pod_cache[pod_id] = {
                "endpoint_url": None,
                "status": "stopped",
                "last_error": None
            }

        # Check current pod status
        response = requests.get(
            f"{RUNPOD_API_BASE}/pod/{pod_id}",
            headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"}
        )
        
        if response.status_code == 200:
            pod_data = response.json()
            pod_status = pod_data["status"]
            
            # If running, verify our app is responding
            if pod_status == "RUNNING":
                if not pod_cache[pod_id]["endpoint_url"]:
                    pod_cache[pod_id]["endpoint_url"] = f"http://{pod_data['ip']}:8000"
                if await check_pod_health(pod_id):
                    return True, "Pod is running and healthy"
            
            # If stopped, start it
            elif pod_status == "STOPPED":
                logging.info(f"Starting pod {pod_id}...")
                start_response = requests.post(
                    f"{RUNPOD_API_BASE}/pod/{pod_id}/start",
                    headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"}
                )
                
                if start_response.status_code != 200:
                    return False, f"Failed to start pod: {start_response.text}"
                
                # Get pod IP after starting
                for _ in range(5):  # Wait for IP to be assigned
                    await asyncio.sleep(5)
                    pod_response = requests.get(
                        f"{RUNPOD_API_BASE}/pod/{pod_id}",
                        headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"}
                    )
                    if pod_response.status_code == 200:
                        pod_data = pod_response.json()
                        if pod_data.get("ip"):
                            pod_cache[pod_id]["endpoint_url"] = f"http://{pod_data['ip']}:8000"
                            break
                
                # Wait for our app to be ready (max 5 minutes)
                for _ in range(30):
                    if await check_pod_health(pod_id):
                        pod_cache[pod_id]["status"] = "running"
                        return True, "Pod is ready to accept requests"
                    await asyncio.sleep(10)
                    logging.info(f"Waiting for pod {pod_id} to be ready...")
                
                return False, "Pod startup timeout"
        
        return False, "Could not verify pod status"
        
    except Exception as e:
        pod_cache[pod_id]["last_error"] = str(e)
        return False, f"Error: {str(e)}"

async def process_queued_requests(pod_id: str):
    """Background task to process queued requests for a specific pod."""
    while True:
        try:
            # Get next request from this pod's queue
            file_data = await request_queues[pod_id].get()
            
            # Try processing with retries
            for attempt in range(MAX_RETRIES):
                try:
                    # Ensure pod is running
                    success, message = await ensure_pod_running(pod_id)
                    if not success:
                        logging.error(f"Pod {pod_id} startup failed: {message}")
                        if attempt == MAX_RETRIES - 1:  # Last attempt
                            break
                        await asyncio.sleep(RETRY_DELAY)
                        continue
                    
                    # Forward request to pod
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"{pod_cache[pod_id]['endpoint_url']}/api/transcribe",
                            data=file_data
                        ) as response:
                            if response.status in [200, 202]:  # Success
                                logging.info(f"Request processed for pod {pod_id}: {response.status}")
                                break
                            elif response.status >= 500:  # Server error, retry
                                if attempt == MAX_RETRIES - 1:  # Last attempt
                                    logging.error(f"Final attempt failed for pod {pod_id}: {response.status}")
                                else:
                                    logging.warning(f"Attempt {attempt + 1} failed, retrying...")
                                    await asyncio.sleep(RETRY_DELAY)
                            else:  # Client error or other, don't retry
                                logging.error(f"Request failed with status {response.status}, not retrying")
                                break
                                
                except Exception as e:
                    if attempt == MAX_RETRIES - 1:  # Last attempt
                        logging.error(f"Final attempt failed for pod {pod_id}: {e}")
                    else:
                        logging.warning(f"Attempt {attempt + 1} failed, retrying: {e}")
                        await asyncio.sleep(RETRY_DELAY)
                    
        except Exception as e:
            logging.error(f"Error processing request for pod {pod_id}: {e}")
        finally:
            request_queues[pod_id].task_done()

@app.post("/api/process/{pod_id}")
async def process_audio(
    pod_id: str,
    file: UploadFile = File(...),
):
    """Process audio through specific pod."""
    try:
        # Get or create queue for this pod
        queue = await get_queue_for_pod(pod_id)
        
        # Prepare file data
        data = aiohttp.FormData()
        data.add_field('file',
                      file.file,
                      filename=file.filename,
                      content_type=file.content_type)
        
        # Add to queue
        await queue.put(data)
        
        return JSONResponse(
            status_code=202,
            content={
                "status": "queued",
                "pod_id": pod_id,
                "position": queue.qsize(),
                "message": "Request queued for processing"
            }
        )
    
    except asyncio.QueueFull:
        raise HTTPException(status_code=429, detail="Queue is full")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status/{pod_id}")
async def get_status(pod_id: str):
    """Get status of specific pod."""
    if pod_id not in pod_cache:
        return {
            "pod_id": pod_id,
            "status": "unknown",
            "queue_size": 0
        }
    
    return {
        "pod_status": pod_cache[pod_id]["status"],
        "pod_id": pod_id,
        "endpoint_url": pod_cache[pod_id]["endpoint_url"],
        "queue_size": request_queues[pod_id].qsize() if pod_id in request_queues else 0,
        "last_error": pod_cache[pod_id]["last_error"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081) 
