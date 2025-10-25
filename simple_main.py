#!/usr/bin/env python3
"""
TSAI Jarvis - Simplified API Gateway

This is a simplified version of the TSAI Jarvis API Gateway that works
with our current module structure and tests the complete platform.
"""

import asyncio
import logging
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic models
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    services: Dict[str, str]

class TestRequest(BaseModel):
    test_type: str
    data: Optional[Dict[str, Any]] = None

class TestResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

# Initialize FastAPI app
app = FastAPI(
    title="TSAI Jarvis API Gateway",
    description="Core Intelligence Platform for TSAI Ecosystem",
    version="1.0.0"
)

# Global variables for service status
service_status = {
    "database": "unknown",
    "redis": "unknown",
    "temporal": "unknown"
}

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("🚀 TSAI Jarvis API Gateway starting...")
    
    # Test database connection
    try:
        import asyncpg
        conn = await asyncpg.connect('postgresql://temporal:temporal@localhost:5432/temporal')
        await conn.fetchval('SELECT 1')
        await conn.close()
        service_status["database"] = "connected"
        logger.info("✅ Database connection successful")
    except Exception as e:
        service_status["database"] = "failed"
        logger.error(f"❌ Database connection failed: {e}")
    
    # Test Redis connection
    try:
        import redis.asyncio as redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        await r.ping()
        await r.close()
        service_status["redis"] = "connected"
        logger.info("✅ Redis connection successful")
    except Exception as e:
        service_status["redis"] = "failed"
        logger.error(f"❌ Redis connection failed: {e}")
    
    # Test Temporal connection (optional)
    try:
        from temporalio.client import Client
        client = await Client.connect("localhost:7233")
        # Simple connection test - just try to connect
        service_status["temporal"] = "connected"
        logger.info("✅ Temporal connection successful")
    except Exception as e:
        service_status["temporal"] = "failed"
        logger.warning(f"⚠️ Temporal connection failed: {e}")
    
    logger.info("🎯 TSAI Jarvis API Gateway startup complete")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "TSAI Jarvis - Core Intelligence Platform",
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check"""
    from datetime import datetime
    
    overall_status = "healthy" if all(
        status in ["connected", "unknown"] for status in service_status.values()
    ) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
        services=service_status
    )

@app.get("/api/services", response_model=Dict[str, Any])
async def get_services():
    """Get TSAI services status"""
    try:
        import asyncpg
        conn = await asyncpg.connect('postgresql://temporal:temporal@localhost:5432/temporal')
        
        services = await conn.fetch("""
            SELECT service_name, service_type, status, endpoint_url 
            FROM tsai_services 
            ORDER BY service_name
        """)
        
        await conn.close()
        
        return {
            "tsai_services": [
                {
                    "name": service["service_name"],
                    "type": service["service_type"],
                    "status": service["status"],
                    "endpoint": service["endpoint_url"]
                }
                for service in services
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch services: {e}")

@app.get("/api/users", response_model=Dict[str, Any])
async def get_users():
    """Get users from database"""
    try:
        import asyncpg
        conn = await asyncpg.connect('postgresql://temporal:temporal@localhost:5432/temporal')
        
        users = await conn.fetch("""
            SELECT u.username, u.email, u.status, r.role_name
            FROM users u
            LEFT JOIN user_roles ur ON u.user_id = ur.user_id
            LEFT JOIN roles r ON ur.role_id = r.role_id
            ORDER BY u.username
        """)
        
        await conn.close()
        
        return {
            "users": [
                {
                    "username": user["username"],
                    "email": user["email"],
                    "status": user["status"],
                    "role": user["role_name"]
                }
                for user in users
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch users: {e}")

@app.post("/api/test", response_model=TestResponse)
async def run_test(request: TestRequest):
    """Run various platform tests"""
    try:
        if request.test_type == "database":
            import asyncpg
            conn = await asyncpg.connect('postgresql://temporal:temporal@localhost:5432/temporal')
            result = await conn.fetchval('SELECT COUNT(*) FROM users')
            await conn.close()
            
            return TestResponse(
                success=True,
                message="Database test successful",
                data={"user_count": result}
            )
        
        elif request.test_type == "redis":
            import redis.asyncio as redis
            r = redis.Redis(host='localhost', port=6379, db=0)
            await r.set("test_key", "test_value")
            value = await r.get("test_key")
            await r.delete("test_key")
            await r.close()
            
            return TestResponse(
                success=True,
                message="Redis test successful",
                data={"test_value": value.decode() if value else None}
            )
        
        elif request.test_type == "platform":
            # Test the complete platform
            import asyncpg
            import redis.asyncio as redis
            
            # Test database
            conn = await asyncpg.connect('postgresql://temporal:temporal@localhost:5432/temporal')
            user_count = await conn.fetchval('SELECT COUNT(*) FROM users')
            service_count = await conn.fetchval('SELECT COUNT(*) FROM tsai_services')
            await conn.close()
            
            # Test Redis
            r = redis.Redis(host='localhost', port=6379, db=0)
            await r.set("platform_test", "success")
            test_result = await r.get("platform_test")
            await r.delete("platform_test")
            await r.close()
            
            return TestResponse(
                success=True,
                message="Platform test successful",
                data={
                    "database": {"users": user_count, "services": service_count},
                    "redis": {"test_result": test_result.decode() if test_result else None},
                    "overall_status": "operational"
                }
            )
        
        else:
            return TestResponse(
                success=False,
                message=f"Unknown test type: {request.test_type}"
            )
    
    except Exception as e:
        return TestResponse(
            success=False,
            message=f"Test failed: {e}"
        )

@app.get("/api/analytics/hockey", response_model=Dict[str, Any])
async def get_hockey_analytics():
    """Get hockey analytics data structure"""
    try:
        import asyncpg
        conn = await asyncpg.connect('postgresql://temporal:temporal@localhost:5432/temporal')
        
        # Get table structure
        columns = await conn.fetch("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'hockey_analytics'
            ORDER BY ordinal_position
        """)
        
        await conn.close()
        
        return {
            "hockey_analytics_schema": [
                {
                    "column": col["column_name"],
                    "type": col["data_type"],
                    "nullable": col["is_nullable"] == "YES"
                }
                for col in columns
            ],
            "message": "Hockey analytics table ready for data collection"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get hockey analytics: {e}")

async def main():
    """Main function to run the API Gateway"""
    logger.info("🎯 Starting TSAI Jarvis API Gateway...")
    
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )
    
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
