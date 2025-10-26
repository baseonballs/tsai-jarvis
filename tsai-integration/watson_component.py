#!/usr/bin/env python3
"""
TSAI Watson Component - NLP and AI Integration
"""

import os
import json
import time
import logging
import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

from tsai_component_base import TSAIComponent

class WatsonComponent(TSAIComponent):
    """TSAI Watson component for NLP and AI"""
    
    def __init__(self, jarvis_core: 'JarvisCoreServices' = None):
        super().__init__("watson", jarvis_core)
        self.active_nlp_tasks = {}
        self.ai_models = {}
        self.conversation_history = []
    
    def process_natural_language(self, text: str, nlp_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process natural language text"""
        try:
            # Start experiment
            experiment_id = self.start_experiment(
                "nlp-processing", 
                nlp_config or {}
            )
            
            # Log parameters
            self.log_params({
                'text_length': len(text),
                'language': nlp_config.get('language', 'en') if nlp_config else 'en',
                'processing_type': nlp_config.get('type', 'general') if nlp_config else 'general',
                'confidence_threshold': nlp_config.get('confidence_threshold', 0.7) if nlp_config else 0.7
            })
            
            # Start workflow
            workflow_id = asyncio.run(self.start_workflow(
                "nlp-processing", 
                {
                    'text': text,
                    'nlp_config': nlp_config or {},
                    'output_format': nlp_config.get('output_format', 'json') if nlp_config else 'json'
                }
            ))
            
            # Store NLP task info
            self.active_nlp_tasks[workflow_id] = {
                'experiment_id': experiment_id,
                'text_length': len(text),
                'language': nlp_config.get('language', 'en') if nlp_config else 'en',
                'started_at': datetime.now().isoformat(),
                'status': 'running',
                'processing_steps': []
            }
            
            # Log business metrics
            self.log_business_metrics({
                'user_engagement': ['nlp_processing_started'],
                'pipeline_success_rate': 1.0
            })
            
            self.logger.info(f"Started NLP processing: {workflow_id}")
            return {
                'workflow_id': workflow_id,
                'experiment_id': experiment_id,
                'status': 'started'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process natural language: {e}")
            raise
    
    def analyze_sentiment(self, text: str, sentiment_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze sentiment of text"""
        try:
            # Start experiment
            experiment_id = self.start_experiment(
                "sentiment-analysis", 
                sentiment_config or {}
            )
            
            # Log parameters
            self.log_params({
                'text_length': len(text),
                'sentiment_model': sentiment_config.get('model', 'default') if sentiment_config else 'default',
                'confidence_threshold': sentiment_config.get('confidence_threshold', 0.7) if sentiment_config else 0.7,
                'emotion_detection': sentiment_config.get('emotion_detection', False) if sentiment_config else False
            })
            
            # Start workflow
            workflow_id = asyncio.run(self.start_workflow(
                "sentiment-analysis", 
                {
                    'text': text,
                    'sentiment_config': sentiment_config or {},
                    'output_format': sentiment_config.get('output_format', 'json') if sentiment_config else 'json'
                }
            ))
            
            # Store sentiment analysis info
            self.active_nlp_tasks[workflow_id] = {
                'experiment_id': experiment_id,
                'text_length': len(text),
                'analysis_type': 'sentiment',
                'started_at': datetime.now().isoformat(),
                'status': 'running'
            }
            
            self.logger.info(f"Started sentiment analysis: {workflow_id}")
            return {
                'workflow_id': workflow_id,
                'experiment_id': experiment_id,
                'status': 'started'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to analyze sentiment: {e}")
            raise
    
    def extract_entities(self, text: str, entity_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Extract entities from text"""
        try:
            # Start experiment
            experiment_id = self.start_experiment(
                "entity-extraction", 
                entity_config or {}
            )
            
            # Log parameters
            self.log_params({
                'text_length': len(text),
                'entity_types': entity_config.get('entity_types', ['person', 'organization', 'location']) if entity_config else ['person', 'organization', 'location'],
                'confidence_threshold': entity_config.get('confidence_threshold', 0.7) if entity_config else 0.7,
                'extraction_depth': entity_config.get('depth', 'standard') if entity_config else 'standard'
            })
            
            # Start workflow
            workflow_id = asyncio.run(self.start_workflow(
                "entity-extraction", 
                {
                    'text': text,
                    'entity_config': entity_config or {},
                    'output_format': entity_config.get('output_format', 'json') if entity_config else 'json'
                }
            ))
            
            # Store entity extraction info
            self.active_nlp_tasks[workflow_id] = {
                'experiment_id': experiment_id,
                'text_length': len(text),
                'analysis_type': 'entity_extraction',
                'started_at': datetime.now().isoformat(),
                'status': 'running'
            }
            
            self.logger.info(f"Started entity extraction: {workflow_id}")
            return {
                'workflow_id': workflow_id,
                'experiment_id': experiment_id,
                'status': 'started'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to extract entities: {e}")
            raise
    
    def generate_text(self, prompt: str, generation_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate text using AI models"""
        try:
            # Start experiment
            experiment_id = self.start_experiment(
                "text-generation", 
                generation_config or {}
            )
            
            # Log parameters
            self.log_params({
                'prompt_length': len(prompt),
                'model_type': generation_config.get('model_type', 'gpt') if generation_config else 'gpt',
                'max_length': generation_config.get('max_length', 100) if generation_config else 100,
                'temperature': generation_config.get('temperature', 0.7) if generation_config else 0.7
            })
            
            # Start workflow
            workflow_id = asyncio.run(self.start_workflow(
                "text-generation", 
                {
                    'prompt': prompt,
                    'generation_config': generation_config or {},
                    'output_format': generation_config.get('output_format', 'text') if generation_config else 'text'
                }
            ))
            
            # Store text generation info
            self.active_nlp_tasks[workflow_id] = {
                'experiment_id': experiment_id,
                'prompt_length': len(prompt),
                'analysis_type': 'text_generation',
                'started_at': datetime.now().isoformat(),
                'status': 'running'
            }
            
            self.logger.info(f"Started text generation: {workflow_id}")
            return {
                'workflow_id': workflow_id,
                'experiment_id': experiment_id,
                'status': 'started'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate text: {e}")
            raise
    
    def translate_text(self, text: str, translation_config: Dict[str, Any]) -> Dict[str, Any]:
        """Translate text between languages"""
        try:
            # Start experiment
            experiment_id = self.start_experiment(
                "text-translation", 
                translation_config.get("parameters", {})
            )
            
            # Log parameters
            self.log_params({
                'text_length': len(text),
                'source_language': translation_config.get('source_language', 'auto'),
                'target_language': translation_config.get('target_language', 'en'),
                'translation_model': translation_config.get('model', 'default')
            })
            
            # Start workflow
            workflow_id = asyncio.run(self.start_workflow(
                "text-translation", 
                {
                    'text': text,
                    'translation_config': translation_config,
                    'output_format': translation_config.get('output_format', 'text')
                }
            ))
            
            # Store translation info
            self.active_nlp_tasks[workflow_id] = {
                'experiment_id': experiment_id,
                'text_length': len(text),
                'analysis_type': 'translation',
                'source_language': translation_config.get('source_language', 'auto'),
                'target_language': translation_config.get('target_language', 'en'),
                'started_at': datetime.now().isoformat(),
                'status': 'running'
            }
            
            self.logger.info(f"Started text translation: {workflow_id}")
            return {
                'workflow_id': workflow_id,
                'experiment_id': experiment_id,
                'status': 'started'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to translate text: {e}")
            raise
    
    def chat_with_ai(self, message: str, chat_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Chat with AI assistant"""
        try:
            # Start experiment
            experiment_id = self.start_experiment(
                "ai-chat", 
                chat_config or {}
            )
            
            # Log parameters
            self.log_params({
                'message_length': len(message),
                'chat_model': chat_config.get('model', 'gpt') if chat_config else 'gpt',
                'conversation_context': chat_config.get('context', 'general') if chat_config else 'general',
                'response_style': chat_config.get('style', 'helpful') if chat_config else 'helpful'
            })
            
            # Start workflow
            workflow_id = asyncio.run(self.start_workflow(
                "ai-chat", 
                {
                    'message': message,
                    'chat_config': chat_config or {},
                    'conversation_history': self.conversation_history[-10:],  # Last 10 messages
                    'output_format': chat_config.get('output_format', 'text') if chat_config else 'text'
                }
            ))
            
            # Store chat info
            self.active_nlp_tasks[workflow_id] = {
                'experiment_id': experiment_id,
                'message_length': len(message),
                'analysis_type': 'ai_chat',
                'started_at': datetime.now().isoformat(),
                'status': 'running'
            }
            
            # Add to conversation history
            self.conversation_history.append({
                'timestamp': datetime.now().isoformat(),
                'type': 'user',
                'message': message,
                'workflow_id': workflow_id
            })
            
            self.logger.info(f"Started AI chat: {workflow_id}")
            return {
                'workflow_id': workflow_id,
                'experiment_id': experiment_id,
                'status': 'started'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to chat with AI: {e}")
            raise
    
    def get_nlp_task_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get NLP task status"""
        try:
            if workflow_id in self.active_nlp_tasks:
                task_info = self.active_nlp_tasks[workflow_id]
                
                # Get workflow status
                status = asyncio.run(self.get_workflow_status(workflow_id))
                
                return {
                    'workflow_id': workflow_id,
                    'experiment_id': task_info['experiment_id'],
                    'task_type': task_info.get('analysis_type', 'unknown'),
                    'status': status['status'],
                    'started_at': task_info['started_at'],
                    'text_length': task_info.get('text_length', task_info.get('prompt_length', task_info.get('message_length', 0)))
                }
            else:
                return {'error': 'NLP task not found'}
                
        except Exception as e:
            self.logger.error(f"Failed to get NLP task status: {e}")
            return {'error': str(e)}
    
    def get_conversation_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get conversation history"""
        try:
            return self.conversation_history[-limit:] if limit else self.conversation_history
        except Exception as e:
            self.logger.error(f"Failed to get conversation history: {e}")
            return []
    
    def get_ai_models(self) -> Dict[str, Any]:
        """Get available AI models"""
        try:
            # Get models from storage
            models = self.list_models()
            
            # Filter for AI models
            ai_models = []
            for model in models:
                if any(keyword in model.get('name', '').lower() for keyword in ['nlp', 'ai', 'gpt', 'bert', 'transformer']):
                    ai_models.append(model)
            
            return {
                'ai_models': ai_models,
                'total_models': len(models),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get AI models: {e}")
            return {'error': str(e)}
    
    def get_nlp_metrics(self) -> Dict[str, Any]:
        """Get NLP-specific metrics"""
        try:
            # Get NLP task metrics
            nlp_metrics = {
                'active_tasks': len(self.active_nlp_tasks),
                'conversation_messages': len(self.conversation_history),
                'ai_models': len(self.get_ai_models().get('ai_models', [])),
                'avg_processing_time': self._calculate_avg_processing_time()
            }
            
            # Log business metrics
            self.log_business_metrics({
                'user_engagement': ['nlp_tasks_active'],
                'pipeline_success_rate': 1.0 if self.active_nlp_tasks else 0.0
            })
            
            return nlp_metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get NLP metrics: {e}")
            return {'error': str(e)}
    
    def _calculate_avg_processing_time(self) -> float:
        """Calculate average processing time"""
        if not self.active_nlp_tasks:
            return 0.0
        
        # Mock calculation
        return 2.5  # seconds
    
    def get_component_metrics(self) -> Dict[str, Any]:
        """Get component-specific metrics"""
        try:
            # Get system metrics
            system_metrics = {
                'cpu_usage': 40.2,  # Mock data
                'memory_usage': 1536 * 1024 * 1024,  # 1.5GB
                'disk_usage': 30 * 1024 * 1024 * 1024,  # 30GB
                'network_in': 800,
                'network_out': 600
            }
            
            # Log system metrics
            self.log_system_metrics(system_metrics)
            
            # Get business metrics
            business_metrics = {
                'active_nlp_tasks': len(self.active_nlp_tasks),
                'conversation_messages': len(self.conversation_history),
                'total_experiments': len(self.get_experiment_runs()),
                'total_artifacts': len(self.list_artifacts()),
                'total_models': len(self.list_models())
            }
            
            return {
                'component_name': self.component_name,
                'system_metrics': system_metrics,
                'business_metrics': business_metrics,
                'nlp_metrics': self.get_nlp_metrics(),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get component metrics: {e}")
            return {'error': str(e)}


def main():
    """Main function for Watson component"""
    
    # Initialize Watson component
    watson = WatsonComponent()
    
    # Health check
    health = watson.health_check()
    print(f"Watson Component Health: {health}")
    
    # Initialize component
    watson.initialize({
        'nlp_model': 'gpt-4',
        'language': 'en',
        'confidence_threshold': 0.7
    })
    
    # Start component
    watson.start()
    
    # Test natural language processing
    nlp_config = {
        'language': 'en',
        'type': 'general',
        'confidence_threshold': 0.7,
        'output_format': 'json'
    }
    
    result = watson.process_natural_language("Hello, how are you today?", nlp_config)
    print(f"✅ Started NLP processing: {result['workflow_id']}")
    
    # Test sentiment analysis
    sentiment_config = {
        'model': 'default',
        'confidence_threshold': 0.7,
        'emotion_detection': True,
        'output_format': 'json'
    }
    
    result = watson.analyze_sentiment("I love this product! It's amazing!", sentiment_config)
    print(f"✅ Started sentiment analysis: {result['workflow_id']}")
    
    # Test entity extraction
    entity_config = {
        'entity_types': ['person', 'organization', 'location'],
        'confidence_threshold': 0.7,
        'depth': 'standard',
        'output_format': 'json'
    }
    
    result = watson.extract_entities("John Smith works at Microsoft in Seattle.", entity_config)
    print(f"✅ Started entity extraction: {result['workflow_id']}")
    
    # Test text generation
    generation_config = {
        'model_type': 'gpt',
        'max_length': 100,
        'temperature': 0.7,
        'output_format': 'text'
    }
    
    result = watson.generate_text("Write a short story about", generation_config)
    print(f"✅ Started text generation: {result['workflow_id']}")
    
    # Test text translation
    translation_config = {
        'parameters': {
            'source_language': 'en',
            'target_language': 'es'
        },
        'source_language': 'en',
        'target_language': 'es',
        'model': 'default',
        'output_format': 'text'
    }
    
    result = watson.translate_text("Hello, how are you?", translation_config)
    print(f"✅ Started text translation: {result['workflow_id']}")
    
    # Test AI chat
    chat_config = {
        'model': 'gpt-4',
        'context': 'general',
        'style': 'helpful',
        'output_format': 'text'
    }
    
    result = watson.chat_with_ai("What is artificial intelligence?", chat_config)
    print(f"✅ Started AI chat: {result['workflow_id']}")
    
    # Test NLP task status
    status = watson.get_nlp_task_status(result['workflow_id'])
    print(f"✅ NLP task status: {status}")
    
    # Test conversation history
    history = watson.get_conversation_history(10)
    print(f"✅ Conversation history: {len(history)} messages")
    
    # Test AI models
    models = watson.get_ai_models()
    print(f"✅ AI models: {models}")
    
    # Get component metrics
    metrics = watson.get_component_metrics()
    print(f"✅ Component metrics: {metrics}")
    
    # Stop component
    watson.stop()
    print("✅ Watson component lifecycle completed")

if __name__ == "__main__":
    main()
