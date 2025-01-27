import os
import time
import logging
import asyncio
import aiohttp
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import requests
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = FastAPI()

# Configuration
RUNPOD_API_KEY = os.getenv('RUNPOD_API_KEY')
RUNPOD_TEMPLATE_ID = os.getenv('RUNPOD_TEMPLATE_ID')  # Your saved template ID
POD_GPU_COUNT = "1"
POD_TYPE = "SECURE"  # or "COMMUNITY"

# RunPod API endpoints
RUNPOD_API_BASE = "https://api.runpod.io/v2"

# Cache pod information
pod_cache = {
    "pod_id": None,
    "endpoint_url": None,
    "status": "stopped",
    "last_error": None
}

async def ensure_pod_running() -> tuple[bool, str]:
    """
    Ensure a pod is running and ready to accept requests.
    Returns (success, message)
    """
    try:
        # Check if pod exists and is running
        if pod_cache["pod_id"]:
            # Check pod status
            response = requests.get(
                f"{RUNPOD_API_BASE}/pod/{pod_cache['pod_id']}",
                headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"}
            )
            
            if response.status_code == 200:
                pod_status = response.json()["status"]
                if pod_status == "RUNNING":
                    return True, "Pod is running"
                elif pod_status in ["PENDING", "STARTING"]:
                    return False, "Pod is starting"
            
            # Pod not found or other status - start new pod
            pod_cache["pod_id"] = None
        
        # Start new pod if needed
        if not pod_cache["pod_id"]:
            response = requests.post(
                f"{RUNPOD_API_BASE}/pod/run",
                headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
                json={
                    "templateId": RUNPOD_TEMPLATE_ID,
                    "gpuCount": POD_GPU_COUNT,
                    "type": POD_TYPE,
                    "name": "demucs-whisper-pod"
                }
            )
            
            if response.status_code != 200:
                return False, f"Failed to start pod: {response.text}"
            
            pod_data = response.json()
            pod_cache["pod_id"] = pod_data["id"]
            pod_cache["endpoint_url"] = f"http://{pod_data['ip']}:8000"
            pod_cache["status"] = "starting"
            
            # Wait for pod to be ready
            max_retries = 30  # 5 minutes maximum wait
            for _ in range(max_retries):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(f"{pod_cache['endpoint_url']}/health") as health_response:
                            if health_response.status == 200:
                                health_data = await health_response.json()
                                if health_data["status"] in ["ready", "installing"]:
                                    pod_cache["status"] = "running"
                                    return True, "Pod is ready"
                except:
                    pass
                
                await asyncio.sleep(10)
            
            return False, "Pod startup timeout"
            
        return True, "Pod is running"
        
    except Exception as e:
        pod_cache["last_error"] = str(e)
        return False, f"Error ensuring pod is running: {str(e)}"

@app.post("/api/process")
async def process_audio(file: UploadFile = File(...)):
    """
    Main endpoint that handles audio processing requests.
    Ensures pod is running and forwards the request.
    """
    # First ensure pod is running
    success, message = await ensure_pod_running()
    if not success:
        raise HTTPException(status_code=503, detail=message)
    
    try:
        # Forward the request to the pod
        async with aiohttp.ClientSession() as session:
            # Create form data with file
            data = aiohttp.FormData()
            data.add_field('file',
                          file.file,
                          filename=file.filename,
                          content_type=file.content_type)
            
            # Forward request to pod
            async with session.post(
                f"{pod_cache['endpoint_url']}/api/transcribe",
                data=data
            ) as response:
                # Stream the response back
                return StreamingResponse(
                    response.content,
                    media_type=response.headers.get('content-type'),
                    headers={
                        'Content-Disposition': response.headers.get('content-disposition', '')
                    },
                    status_code=response.status
                )
    
    except Exception as e:
        pod_cache["last_error"] = str(e)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/api/status")
async def get_status():
    """Get current gateway and pod status."""
    return {
        "pod_status": pod_cache["status"],
        "pod_id": pod_cache["pod_id"],
        "endpoint_url": pod_cache["endpoint_url"],
        "last_error": pod_cache["last_error"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080) 
