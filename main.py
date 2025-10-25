#!/usr/bin/env python3
"""
TSAI Jarvis - Core Intelligence Platform
Main entry point for the TSAI Jarvis system
"""
import asyncio
import logging
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from api_gateway.intelligence_api.jarvis_api import jarvis_api
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/jarvis.log')
    ]
)

logger = logging.getLogger(__name__)

async def main():
    """Main entry point for TSAI Jarvis"""
    logger.info("🚀 Starting TSAI Jarvis - Core Intelligence Platform")
    
    try:
        # Initialize Jarvis
        await jarvis_api.initialize()
        
        # Start the server
        config = uvicorn.Config(
            app=jarvis_api.app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            reload=False
        )
        
        server = uvicorn.Server(config)
        await server.serve()
        
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down TSAI Jarvis...")
        await jarvis_api.shutdown()
    except Exception as e:
        logger.error(f"❌ TSAI Jarvis failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
