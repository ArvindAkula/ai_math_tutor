#!/usr/bin/env python3
"""
Health check script for the math engine service.
This script is used by Docker health checks to verify service availability.
"""

import sys
import requests
import time
from typing import Dict, Any

def check_health() -> Dict[str, Any]:
    """
    Perform health check on the math engine service.
    
    Returns:
        Dict containing health status and details
    """
    try:
        # Check if the service is responding
        response = requests.get(
            "http://localhost:8001/health",
            timeout=5
        )
        
        if response.status_code == 200:
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "response_time": response.elapsed.total_seconds(),
                "details": response.json() if response.content else {}
            }
        else:
            return {
                "status": "unhealthy",
                "timestamp": time.time(),
                "error": f"HTTP {response.status_code}",
                "details": response.text
            }
            
    except requests.exceptions.ConnectionError:
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": "Connection refused",
            "details": "Service is not responding"
        }
    except requests.exceptions.Timeout:
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": "Request timeout",
            "details": "Service took too long to respond"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": "Unexpected error",
            "details": str(e)
        }

def main():
    """Main health check function."""
    health_status = check_health()
    
    if health_status["status"] == "healthy":
        print("Health check passed")
        sys.exit(0)
    else:
        print(f"Health check failed: {health_status['error']}")
        print(f"Details: {health_status['details']}")
        sys.exit(1)

if __name__ == "__main__":
    main()