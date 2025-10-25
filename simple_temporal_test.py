#!/usr/bin/env python3
"""
Simple Temporal AI Test - Verify connection and basic workflow execution
"""

import asyncio
import logging
from datetime import datetime, timedelta
from temporalio import workflow, activity
from temporalio.client import Client
from temporalio.worker import Worker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple Activity
@activity.defn
async def simple_hockey_activity(game_id: str) -> str:
    """Simple hockey analytics activity"""
    logger.info(f"🏒 Processing hockey game: {game_id}")
    await asyncio.sleep(1)  # Simulate work
    return f"Hockey analysis completed for game {game_id}"

# Simple Workflow
@workflow.defn
class SimpleHockeyWorkflow:
    @workflow.run
    async def run(self, game_id: str) -> str:
        """Simple hockey workflow"""
        logger.info(f"🚀 Starting hockey workflow for game: {game_id}")
        
        result = await workflow.execute_activity(
            simple_hockey_activity,
            game_id,
            start_to_close_timeout=timedelta(minutes=1)
        )
        
        logger.info(f"✅ Hockey workflow completed: {result}")
        return result

async def test_temporal_connection():
    """Test basic Temporal connection"""
    try:
        client = await Client.connect("localhost:7233")
        logger.info("✅ Temporal connection successful")
        return True
    except Exception as e:
        logger.error(f"❌ Temporal connection failed: {e}")
        return False

async def test_simple_workflow():
    """Test a simple workflow execution"""
    try:
        client = await Client.connect("localhost:7233")
        
        # Start workflow
        workflow_handle = await client.start_workflow(
            SimpleHockeyWorkflow.run,
            "TEST_GAME_001",
            id=f"test-hockey-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            task_queue="test-tasks"
        )
        
        logger.info(f"🚀 Workflow started: {workflow_handle.id}")
        
        # Wait for result
        result = await workflow_handle.result()
        
        logger.info(f"✅ Workflow completed: {result}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Workflow test failed: {e}")
        return False

async def run_simple_worker():
    """Run a simple worker for testing"""
    try:
        client = await Client.connect("localhost:7233")
        
        worker = Worker(
            client,
            task_queue="test-tasks",
            workflows=[SimpleHockeyWorkflow],
            activities=[simple_hockey_activity]
        )
        
        logger.info("🔧 Simple Temporal Worker started")
        await worker.run()
        
    except Exception as e:
        logger.error(f"❌ Worker failed: {e}")

async def main():
    """Main test function"""
    logger.info("🎯 TSAI Jarvis - Simple Temporal AI Test")
    logger.info("=" * 50)
    
    # Test 1: Connection
    logger.info("\n1️⃣ Testing Temporal Connection")
    if await test_temporal_connection():
        logger.info("✅ Temporal connection successful")
    else:
        logger.info("❌ Temporal connection failed")
        return
    
    # Test 2: Simple Workflow (with worker in background)
    logger.info("\n2️⃣ Testing Simple Workflow")
    logger.info("🔧 Starting worker in background...")
    
    # Start worker in background
    worker_task = asyncio.create_task(run_simple_worker())
    await asyncio.sleep(2)  # Let worker start
    
    # Test workflow
    if await test_simple_workflow():
        logger.info("✅ Simple workflow test successful")
    else:
        logger.info("❌ Simple workflow test failed")
    
    # Stop worker
    worker_task.cancel()
    
    logger.info("\n🎉 Simple Temporal AI Test Complete!")

if __name__ == "__main__":
    asyncio.run(main())
