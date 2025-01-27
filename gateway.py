import os
import time
import logging
import asyncio
import aiohttp
from fastapi import FastAPI, HTTPException
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

# Cache pod information
pod_cache: Dict[str, Dict] = {}

async def check_pod_ready(pod_id: str, health_url: str) -> bool:
    """Check if pod is running and app is responding."""
    try:
        # 1. Check if pod is running via RunPod API
        response = requests.get(
            f"{RUNPOD_API_BASE}/pod/{pod_id}",
            headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"}
        )
        
        if response.status_code != 200:
            return False
            
        pod_data = response.json()
        if pod_data["status"] != "RUNNING":
            return False
            
        # 2. If running, check if app is responding
        if health_url:  # Only check health if URL provided
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(health_url) as health_response:
                        return health_response.status == 200
            except:
                return False
                
        return True  # If no health URL, just return pod running status
        
    except Exception as e:
        logging.error(f"Error checking pod {pod_id}: {e}")
        return False

@app.post("/api/start/{pod_id}")
async def start_pod(pod_id: str, health_url: Optional[str] = None):
    """Start a pod and wait for it to be ready."""
    try:
        # Check if already running
        is_ready = await check_pod_ready(pod_id, health_url)
        if is_ready:
            return {"status": "ready", "message": "Pod is already running"}
            
        # Start the pod if needed
        response = requests.post(
            f"{RUNPOD_API_BASE}/pod/{pod_id}/start",
            headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"}
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to start pod: {response.text}"
            )
            
        # Wait for pod to be ready (max 2 minutes)
        for _ in range(24):  # 5 second intervals
            if await check_pod_ready(pod_id, health_url):
                return {
                    "status": "ready",
                    "message": "Pod started and ready"
                }
            await asyncio.sleep(5)
            
        return {
            "status": "starting",
            "message": "Pod is starting but not ready yet"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status/{pod_id}")
async def get_status(pod_id: str, health_url: Optional[str] = None):
    """Get pod status and readiness."""
    try:
        response = requests.get(
            f"{RUNPOD_API_BASE}/pod/{pod_id}",
            headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"}
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=404, detail="Pod not found")
            
        pod_data = response.json()
        is_ready = await check_pod_ready(pod_id, health_url)
        
        return {
            "pod_status": pod_data["status"],
            "ready": is_ready,
            "pod_id": pod_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080) 
