#!/usr/bin/env python3
"""
TSAI Jarvis - Watson Integration API
NLP reasoning and text analysis for hockey analytics
"""

import asyncio
import json
import logging
import sqlite3
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="TSAI Jarvis Watson Integration API",
    description="NLP reasoning and text analysis for hockey analytics",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Enums ---
class TextAnalysisType(str, Enum):
    SENTIMENT = "sentiment"
    ENTITY_RECOGNITION = "entity_recognition"
    TOPIC_MODELING = "topic_modeling"
    TEXT_SUMMARIZATION = "text_summarization"
    INTENT_CLASSIFICATION = "intent_classification"
    EMOTION_ANALYSIS = "emotion_analysis"
    KEYWORD_EXTRACTION = "keyword_extraction"

class ReasoningType(str, Enum):
    LOGICAL_INFERENCE = "logical_inference"
    CAUSAL_REASONING = "causal_reasoning"
    TEMPORAL_REASONING = "temporal_reasoning"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    PREDICTIVE_REASONING = "predictive_reasoning"

class DocumentType(str, Enum):
    GAME_REPORT = "game_report"
    PLAYER_INTERVIEW = "player_interview"
    COACH_COMMENTARY = "coach_commentary"
    FAN_SENTIMENT = "fan_sentiment"
    MEDIA_ARTICLE = "media_article"
    SOCIAL_MEDIA = "social_media"
    SCRAPING_DATA = "scraping_data"

class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# --- Pydantic Models ---
class TextDocument(BaseModel):
    document_id: str
    title: str
    content: str
    document_type: DocumentType
    source: str
    author: Optional[str] = None
    created_at: datetime
    metadata: Dict[str, Any] = {}

class TextAnalysis(BaseModel):
    analysis_id: str
    document_id: str
    analysis_type: TextAnalysisType
    status: ProcessingStatus
    results: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    processing_time: float = 0.0
    created_at: datetime
    completed_at: Optional[datetime] = None

class ReasoningTask(BaseModel):
    task_id: str
    reasoning_type: ReasoningType
    input_data: Dict[str, Any]
    context: Dict[str, Any] = {}
    status: ProcessingStatus
    result: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    reasoning_steps: List[Dict[str, Any]] = []
    created_at: datetime
    completed_at: Optional[datetime] = None

class SentimentAnalysis(BaseModel):
    document_id: str
    overall_sentiment: str  # positive, negative, neutral
    sentiment_score: float  # -1.0 to 1.0
    emotion_breakdown: Dict[str, float] = {}
    key_phrases: List[str] = []
    confidence: float = 0.0

class EntityExtraction(BaseModel):
    document_id: str
    entities: List[Dict[str, Any]] = []  # [{"text": "Auston Matthews", "type": "PERSON", "confidence": 0.95}]
    relationships: List[Dict[str, Any]] = []  # [{"entity1": "Auston Matthews", "entity2": "Toronto Maple Leafs", "relation": "plays_for"}]
    confidence: float = 0.0

class TopicModeling(BaseModel):
    document_id: str
    topics: List[Dict[str, Any]] = []  # [{"topic": "hockey strategy", "weight": 0.8, "keywords": ["offense", "defense"]}]
    topic_distribution: Dict[str, float] = {}
    coherence_score: float = 0.0

class TextSummary(BaseModel):
    document_id: str
    summary: str
    key_points: List[str] = []
    summary_length: int = 0
    compression_ratio: float = 0.0
    quality_score: float = 0.0

class ReasoningResult(BaseModel):
    task_id: str
    conclusion: str
    supporting_evidence: List[str] = []
    confidence: float = 0.0
    reasoning_chain: List[Dict[str, Any]] = []
    alternative_conclusions: List[str] = []

class NLPAnalytics(BaseModel):
    analytics_id: str
    time_period: str
    total_documents: int
    sentiment_distribution: Dict[str, int]
    top_entities: List[Dict[str, Any]]
    trending_topics: List[Dict[str, Any]]
    reasoning_accuracy: float
    processing_volume: int
    created_at: datetime

# --- Database Setup ---
def init_database():
    """Initialize the Watson database"""
    conn = sqlite3.connect('watson_integration.db')
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS text_documents (
            document_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            document_type TEXT NOT NULL,
            source TEXT NOT NULL,
            author TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT DEFAULT '{}'
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS text_analyses (
            analysis_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            analysis_type TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            results TEXT,
            confidence REAL DEFAULT 0.0,
            processing_time REAL DEFAULT 0.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reasoning_tasks (
            task_id TEXT PRIMARY KEY,
            reasoning_type TEXT NOT NULL,
            input_data TEXT NOT NULL,
            context TEXT DEFAULT '{}',
            status TEXT DEFAULT 'pending',
            result TEXT,
            confidence REAL DEFAULT 0.0,
            reasoning_steps TEXT DEFAULT '[]',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sentiment_analyses (
            document_id TEXT PRIMARY KEY,
            overall_sentiment TEXT NOT NULL,
            sentiment_score REAL NOT NULL,
            emotion_breakdown TEXT DEFAULT '{}',
            key_phrases TEXT DEFAULT '[]',
            confidence REAL DEFAULT 0.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS entity_extractions (
            document_id TEXT PRIMARY KEY,
            entities TEXT DEFAULT '[]',
            relationships TEXT DEFAULT '[]',
            confidence REAL DEFAULT 0.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS topic_modelings (
            document_id TEXT PRIMARY KEY,
            topics TEXT DEFAULT '[]',
            topic_distribution TEXT DEFAULT '{}',
            coherence_score REAL DEFAULT 0.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS text_summaries (
            document_id TEXT PRIMARY KEY,
            summary TEXT NOT NULL,
            key_points TEXT DEFAULT '[]',
            summary_length INTEGER DEFAULT 0,
            compression_ratio REAL DEFAULT 0.0,
            quality_score REAL DEFAULT 0.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reasoning_results (
            task_id TEXT PRIMARY KEY,
            conclusion TEXT NOT NULL,
            supporting_evidence TEXT DEFAULT '[]',
            confidence REAL DEFAULT 0.0,
            reasoning_chain TEXT DEFAULT '[]',
            alternative_conclusions TEXT DEFAULT '[]',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS nlp_analytics (
            analytics_id TEXT PRIMARY KEY,
            time_period TEXT NOT NULL,
            total_documents INTEGER NOT NULL,
            sentiment_distribution TEXT NOT NULL,
            top_entities TEXT NOT NULL,
            trending_topics TEXT NOT NULL,
            reasoning_accuracy REAL NOT NULL,
            processing_volume INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("Watson database initialized")

# Initialize database on startup
init_database()

# --- WebSocket Connection Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.analysis_connections: List[WebSocket] = []
        self.reasoning_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket, connection_type: str = "general"):
        await websocket.accept()
        if connection_type == "analysis":
            self.analysis_connections.append(websocket)
        elif connection_type == "reasoning":
            self.reasoning_connections.append(websocket)
        else:
            self.active_connections.append(websocket)
        logger.info(f"WebSocket connected: {connection_type}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.analysis_connections:
            self.analysis_connections.remove(websocket)
        if websocket in self.reasoning_connections:
            self.reasoning_connections.remove(websocket)
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info("WebSocket disconnected")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast_to_analyzers(self, message: str):
        for connection in self.analysis_connections:
            await connection.send_text(message)

    async def broadcast_to_reasoners(self, message: str):
        for connection in self.reasoning_connections:
            await connection.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# --- API Endpoints ---
@app.get("/health", summary="Health Check")
async def health_check():
    """Check the health of the Watson Integration API"""
    return {"status": "ok", "message": "TSAI Jarvis Watson Integration API is running"}

@app.get("/api/watson/status", summary="Get Watson Status")
async def get_watson_status():
    """Get the current status of the Watson integration"""
    conn = sqlite3.connect('watson_integration.db')
    cursor = conn.cursor()
    
    # Get total documents
    cursor.execute("SELECT COUNT(*) FROM text_documents")
    total_documents = cursor.fetchone()[0]
    
    # Get pending analyses
    cursor.execute("SELECT COUNT(*) FROM text_analyses WHERE status = 'pending'")
    pending_analyses = cursor.fetchone()[0]
    
    # Get active reasoning tasks
    cursor.execute("SELECT COUNT(*) FROM reasoning_tasks WHERE status IN ('pending', 'processing')")
    active_reasoning = cursor.fetchone()[0]
    
    # Get recent sentiment analyses
    cursor.execute("SELECT COUNT(*) FROM sentiment_analyses WHERE created_at > datetime('now', '-1 hour')")
    recent_sentiment = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        "status": "active",
        "total_documents": total_documents,
        "pending_analyses": pending_analyses,
        "active_reasoning": active_reasoning,
        "recent_sentiment": recent_sentiment,
        "analysis_connections": len(manager.analysis_connections),
        "reasoning_connections": len(manager.reasoning_connections),
        "total_connections": len(manager.active_connections)
    }

@app.post("/api/watson/documents", summary="Create Text Document")
async def create_text_document(document: TextDocument):
    """Create a new text document for analysis"""
    conn = sqlite3.connect('watson_integration.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO text_documents (document_id, title, content, document_type, 
                                      source, author, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            document.document_id,
            document.title,
            document.content,
            document.document_type,
            document.source,
            document.author,
            document.created_at,
            json.dumps(document.metadata)
        ))
        
        conn.commit()
        logger.info(f"Created text document: {document.document_id}")
        
        # Notify analysis connections about new document
        await manager.broadcast_to_analyzers(json.dumps({
            "type": "document_created",
            "document_id": document.document_id,
            "title": document.title,
            "document_type": document.document_type,
            "source": document.source
        }))
        
        return {"message": "Text document created successfully", "document_id": document.document_id}
        
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create document: {str(e)}")
    finally:
        conn.close()

@app.get("/api/watson/documents", summary="Get All Documents")
async def get_documents(document_type: Optional[str] = None, limit: int = 50):
    """Get text documents with optional filtering"""
    conn = sqlite3.connect('watson_integration.db')
    cursor = conn.cursor()
    
    query = "SELECT document_id, title, document_type, source, author, created_at FROM text_documents"
    params = []
    
    if document_type:
        query += " WHERE document_type = ?"
        params.append(document_type)
    
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    
    cursor.execute(query, params)
    
    documents = []
    for row in cursor.fetchall():
        documents.append({
            "document_id": row[0],
            "title": row[1],
            "document_type": row[2],
            "source": row[3],
            "author": row[4],
            "created_at": row[5]
        })
    
    conn.close()
    return {"documents": documents}

@app.post("/api/watson/analyze", summary="Analyze Text")
async def analyze_text(analysis: TextAnalysis):
    """Perform text analysis on a document"""
    conn = sqlite3.connect('watson_integration.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO text_analyses (analysis_id, document_id, analysis_type, status, 
                                    results, confidence, processing_time, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            analysis.analysis_id,
            analysis.document_id,
            analysis.analysis_type,
            analysis.status,
            json.dumps(analysis.results) if analysis.results else None,
            analysis.confidence,
            analysis.processing_time,
            analysis.created_at
        ))
        
        conn.commit()
        logger.info(f"Created text analysis: {analysis.analysis_id}")
        
        # Notify analysis connections about new analysis
        await manager.broadcast_to_analyzers(json.dumps({
            "type": "analysis_created",
            "analysis_id": analysis.analysis_id,
            "document_id": analysis.document_id,
            "analysis_type": analysis.analysis_type,
            "status": analysis.status
        }))
        
        return {"message": "Text analysis created successfully", "analysis_id": analysis.analysis_id}
        
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create analysis: {str(e)}")
    finally:
        conn.close()

@app.post("/api/watson/sentiment", summary="Analyze Sentiment")
async def analyze_sentiment(sentiment: SentimentAnalysis):
    """Analyze sentiment of a document"""
    conn = sqlite3.connect('watson_integration.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT OR REPLACE INTO sentiment_analyses (document_id, overall_sentiment, 
                                                     sentiment_score, emotion_breakdown, 
                                                     key_phrases, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            sentiment.document_id,
            sentiment.overall_sentiment,
            sentiment.sentiment_score,
            json.dumps(sentiment.emotion_breakdown),
            json.dumps(sentiment.key_phrases),
            sentiment.confidence,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        logger.info(f"Analyzed sentiment: {sentiment.document_id}")
        
        # Notify analysis connections about sentiment analysis
        await manager.broadcast_to_analyzers(json.dumps({
            "type": "sentiment_analysis",
            "document_id": sentiment.document_id,
            "overall_sentiment": sentiment.overall_sentiment,
            "sentiment_score": sentiment.sentiment_score,
            "confidence": sentiment.confidence
        }))
        
        return {"message": "Sentiment analysis completed successfully"}
        
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to analyze sentiment: {str(e)}")
    finally:
        conn.close()

@app.post("/api/watson/entities", summary="Extract Entities")
async def extract_entities(entities: EntityExtraction):
    """Extract entities from a document"""
    conn = sqlite3.connect('watson_integration.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT OR REPLACE INTO entity_extractions (document_id, entities, relationships, 
                                                     confidence, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            entities.document_id,
            json.dumps(entities.entities),
            json.dumps(entities.relationships),
            entities.confidence,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        logger.info(f"Extracted entities: {entities.document_id}")
        
        # Notify analysis connections about entity extraction
        await manager.broadcast_to_analyzers(json.dumps({
            "type": "entity_extraction",
            "document_id": entities.document_id,
            "entity_count": len(entities.entities),
            "relationship_count": len(entities.relationships),
            "confidence": entities.confidence
        }))
        
        return {"message": "Entity extraction completed successfully"}
        
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to extract entities: {str(e)}")
    finally:
        conn.close()

@app.post("/api/watson/topics", summary="Model Topics")
async def model_topics(topics: TopicModeling):
    """Perform topic modeling on a document"""
    conn = sqlite3.connect('watson_integration.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT OR REPLACE INTO topic_modelings (document_id, topics, topic_distribution, 
                                                  coherence_score, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            topics.document_id,
            json.dumps(topics.topics),
            json.dumps(topics.topic_distribution),
            topics.coherence_score,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        logger.info(f"Modeled topics: {topics.document_id}")
        
        # Notify analysis connections about topic modeling
        await manager.broadcast_to_analyzers(json.dumps({
            "type": "topic_modeling",
            "document_id": topics.document_id,
            "topic_count": len(topics.topics),
            "coherence_score": topics.coherence_score
        }))
        
        return {"message": "Topic modeling completed successfully"}
        
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to model topics: {str(e)}")
    finally:
        conn.close()

@app.post("/api/watson/summarize", summary="Summarize Text")
async def summarize_text(summary: TextSummary):
    """Summarize a document"""
    conn = sqlite3.connect('watson_integration.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT OR REPLACE INTO text_summaries (document_id, summary, key_points, 
                                                 summary_length, compression_ratio, 
                                                 quality_score, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            summary.document_id,
            summary.summary,
            json.dumps(summary.key_points),
            summary.summary_length,
            summary.compression_ratio,
            summary.quality_score,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        logger.info(f"Summarized text: {summary.document_id}")
        
        # Notify analysis connections about text summarization
        await manager.broadcast_to_analyzers(json.dumps({
            "type": "text_summarization",
            "document_id": summary.document_id,
            "summary_length": summary.summary_length,
            "compression_ratio": summary.compression_ratio,
            "quality_score": summary.quality_score
        }))
        
        return {"message": "Text summarization completed successfully"}
        
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to summarize text: {str(e)}")
    finally:
        conn.close()

@app.post("/api/watson/reasoning", summary="Perform Reasoning")
async def perform_reasoning(task: ReasoningTask):
    """Perform reasoning task"""
    conn = sqlite3.connect('watson_integration.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO reasoning_tasks (task_id, reasoning_type, input_data, context, 
                                       status, result, confidence, reasoning_steps, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            task.task_id,
            task.reasoning_type,
            json.dumps(task.input_data),
            json.dumps(task.context),
            task.status,
            json.dumps(task.result) if task.result else None,
            task.confidence,
            json.dumps(task.reasoning_steps),
            task.created_at
        ))
        
        conn.commit()
        logger.info(f"Created reasoning task: {task.task_id}")
        
        # Notify reasoning connections about new task
        await manager.broadcast_to_reasoners(json.dumps({
            "type": "reasoning_task",
            "task_id": task.task_id,
            "reasoning_type": task.reasoning_type,
            "status": task.status
        }))
        
        return {"message": "Reasoning task created successfully", "task_id": task.task_id}
        
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create reasoning task: {str(e)}")
    finally:
        conn.close()

@app.post("/api/watson/reasoning/result", summary="Submit Reasoning Result")
async def submit_reasoning_result(result: ReasoningResult):
    """Submit reasoning result"""
    conn = sqlite3.connect('watson_integration.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT OR REPLACE INTO reasoning_results (task_id, conclusion, supporting_evidence, 
                                                    confidence, reasoning_chain, 
                                                    alternative_conclusions, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            result.task_id,
            result.conclusion,
            json.dumps(result.supporting_evidence),
            result.confidence,
            json.dumps(result.reasoning_chain),
            json.dumps(result.alternative_conclusions),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        logger.info(f"Submitted reasoning result: {result.task_id}")
        
        # Notify reasoning connections about reasoning result
        await manager.broadcast_to_reasoners(json.dumps({
            "type": "reasoning_result",
            "task_id": result.task_id,
            "conclusion": result.conclusion,
            "confidence": result.confidence
        }))
        
        return {"message": "Reasoning result submitted successfully"}
        
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to submit reasoning result: {str(e)}")
    finally:
        conn.close()

@app.get("/api/watson/analytics", summary="Get NLP Analytics")
async def get_nlp_analytics(time_period: str = "24h"):
    """Get NLP analytics for a time period"""
    conn = sqlite3.connect('watson_integration.db')
    cursor = conn.cursor()
    
    # Calculate time filter
    if time_period == "24h":
        time_filter = "datetime('now', '-1 day')"
    elif time_period == "7d":
        time_filter = "datetime('now', '-7 days')"
    elif time_period == "30d":
        time_filter = "datetime('now', '-30 days')"
    else:
        time_filter = "datetime('now', '-1 day')"
    
    # Get sentiment distribution
    cursor.execute(f'''
        SELECT overall_sentiment, COUNT(*) as count
        FROM sentiment_analyses 
        WHERE created_at > {time_filter}
        GROUP BY overall_sentiment
    ''')
    
    sentiment_distribution = {}
    for row in cursor.fetchall():
        sentiment_distribution[row[0]] = row[1]
    
    # Get top entities
    cursor.execute(f'''
        SELECT entities FROM entity_extractions 
        WHERE created_at > {time_filter}
        LIMIT 100
    ''')
    
    all_entities = []
    for row in cursor.fetchall():
        entities = json.loads(row[0])
        all_entities.extend(entities)
    
    # Count entity frequency
    entity_counts = {}
    for entity in all_entities:
        entity_text = entity.get('text', '')
        if entity_text:
            entity_counts[entity_text] = entity_counts.get(entity_text, 0) + 1
    
    top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Get trending topics
    cursor.execute(f'''
        SELECT topics FROM topic_modelings 
        WHERE created_at > {time_filter}
        LIMIT 50
    ''')
    
    all_topics = []
    for row in cursor.fetchall():
        topics = json.loads(row[0])
        all_topics.extend(topics)
    
    # Count topic frequency
    topic_counts = {}
    for topic in all_topics:
        topic_name = topic.get('topic', '')
        if topic_name:
            topic_counts[topic_name] = topic_counts.get(topic_name, 0) + 1
    
    trending_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    conn.close()
    
    return {
        "time_period": time_period,
        "sentiment_distribution": sentiment_distribution,
        "top_entities": [{"entity": k, "count": v} for k, v in top_entities],
        "trending_topics": [{"topic": k, "count": v} for k, v in trending_topics],
        "total_documents": sum(sentiment_distribution.values()),
        "processing_volume": len(all_entities) + len(all_topics)
    }

# --- WebSocket Endpoints ---
@app.websocket("/ws/watson")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for general Watson communication"""
    await manager.connect(websocket, "general")
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages from clients
            message = json.loads(data)
            logger.info(f"Received message: {message}")
            
            # Echo back the message to all connected clients
            await manager.broadcast(json.dumps({
                "type": "echo",
                "message": message,
                "timestamp": datetime.now().isoformat()
            }))
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@app.websocket("/ws/watson/analysis")
async def analysis_websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for text analysis"""
    await manager.connect(websocket, "analysis")
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages from analysis clients
            message = json.loads(data)
            logger.info(f"Received analysis message: {message}")
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Analysis WebSocket error: {e}")
        manager.disconnect(websocket)

@app.websocket("/ws/watson/reasoning")
async def reasoning_websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for reasoning tasks"""
    await manager.connect(websocket, "reasoning")
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages from reasoning clients
            message = json.loads(data)
            logger.info(f"Received reasoning message: {message}")
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Reasoning WebSocket error: {e}")
        manager.disconnect(websocket)

# --- Background Tasks ---
async def simulate_nlp_processing():
    """Simulate NLP processing for demonstration"""
    while True:
        try:
            if manager.analysis_connections:
                # Simulate random NLP insights
                import random
                if random.random() < 0.2:  # 20% chance every 10 seconds
                    insights = [
                        "Positive sentiment detected in recent game reports",
                        "Key entities: Auston Matthews, Toronto Maple Leafs, Goal",
                        "Trending topic: Playoff strategy analysis",
                        "High confidence reasoning: Team performance correlation"
                    ]
                    
                    await manager.broadcast_to_analyzers(json.dumps({
                        "type": "nlp_insight",
                        "insight": random.choice(insights),
                        "timestamp": datetime.now().isoformat()
                    }))
            
        except Exception as e:
            logger.error(f"Error simulating NLP processing: {e}")
        
        await asyncio.sleep(10)  # Check every 10 seconds

@app.on_event("startup")
async def startup_event():
    """Start background tasks on startup"""
    logger.info("Starting Watson Integration API background tasks...")
    asyncio.create_task(simulate_nlp_processing())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8016)
