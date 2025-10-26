#!/usr/bin/env python3
"""
Integration Test Suite for Jarvis Core Services
"""

import os
import sys
import json
import time
import logging
import asyncio
import tempfile
from pathlib import Path
from typing import Dict, List, Any

# Add the services to the path
sys.path.append('/Volumes/Thorage/wip/tsai-jarvis/services')
sys.path.append('/Volumes/Thorage/wip/tsai-jarvis/tsai-integration')

# Import TSAI Integration
from tsai_integration import (
    TSAIComponent, ToolchainComponent, SpotlightComponent, 
    AutopilotComponent, SherlockComponent, WatsonComponent
)

class JarvisIntegrationTester:
    """Integration test suite for Jarvis Core Services"""
    
    def __init__(self):
        self.test_results = {}
        self.logger = logging.getLogger("JarvisIntegrationTester")
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for tests"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def test_complete_hockey_pipeline(self) -> bool:
        """Test complete hockey detection pipeline"""
        try:
            self.logger.info("🏒 Testing Complete Hockey Detection Pipeline...")
            
            # Initialize Toolchain component
            toolchain = ToolchainComponent()
            toolchain.initialize({
                'model_type': 'yolov8n',
                'target_accuracy': 0.85,
                'max_epochs': 100
            })
            toolchain.start()
            
            # Step 1: Import hockey media
            self.logger.info("📥 Step 1: Importing hockey media...")
            files = toolchain.import_hockey_media("google_drive")
            self.logger.info(f"✅ Imported {len(files)} hockey media files")
            
            # Step 2: Run hockey detection pipeline
            self.logger.info("🔍 Step 2: Running hockey detection pipeline...")
            config = {
                'parameters': {
                    'model_type': 'yolov8n',
                    'task': 'role_classification',
                    'target_accuracy': 0.85
                },
                'data_config': {
                    'dataset_path': '/data/hockey_players',
                    'train_split': 0.8,
                    'val_split': 0.1,
                    'test_split': 0.1
                },
                'training_config': {
                    'epochs': 100,
                    'batch_size': 32,
                    'learning_rate': 0.001
                },
                'evaluation_config': {
                    'metrics': ['accuracy', 'precision', 'recall', 'f1_score'],
                    'threshold': 0.5
                }
            }
            
            workflow_id = toolchain.run_hockey_detection_pipeline(config)
            if not workflow_id:
                self.logger.error("❌ Failed to start hockey detection pipeline")
                return False
            
            self.logger.info(f"✅ Started hockey detection pipeline: {workflow_id}")
            
            # Step 3: Run role classification
            self.logger.info("👥 Step 3: Running role classification...")
            role_config = {
                'parameters': {
                    'model_type': 'yolov8n',
                    'task': 'role_classification',
                    'target_accuracy': 0.85
                },
                'data_config': {
                    'dataset_path': '/data/hockey_players',
                    'role_types': ['player', 'goalie', 'referee']
                },
                'training_config': {
                    'epochs': 100,
                    'batch_size': 32
                }
            }
            
            role_workflow_id = asyncio.run(toolchain.run_hockey_role_classification(role_config))
            if not role_workflow_id:
                self.logger.error("❌ Failed to start role classification")
                return False
            
            self.logger.info(f"✅ Started role classification: {role_workflow_id}")
            
            # Step 4: Run equipment classification
            self.logger.info("🛡️ Step 4: Running equipment classification...")
            equipment_config = {
                'parameters': {
                    'model_type': 'yolov8n',
                    'task': 'equipment_classification',
                    'target_accuracy': 0.75
                },
                'data_config': {
                    'dataset_path': '/data/hockey_players',
                    'equipment_types': ['helmet', 'gloves', 'stick', 'skates', 'jersey']
                },
                'training_config': {
                    'epochs': 100,
                    'batch_size': 32
                }
            }
            
            equipment_workflow_id = toolchain.run_hockey_equipment_classification(equipment_config)
            if not equipment_workflow_id:
                self.logger.error("❌ Failed to start equipment classification")
                return False
            
            self.logger.info(f"✅ Started equipment classification: {equipment_workflow_id}")
            
            # Step 5: Run ice surface detection
            self.logger.info("🧊 Step 5: Running ice surface detection...")
            ice_config = {
                'parameters': {
                    'model_type': 'yolov8n',
                    'task': 'ice_surface_detection',
                    'target_accuracy': 0.90
                },
                'data_config': {
                    'dataset_path': '/data/hockey_players',
                    'surface_types': ['ice', 'non_ice']
                },
                'training_config': {
                    'epochs': 100,
                    'batch_size': 32
                }
            }
            
            ice_workflow_id = toolchain.run_hockey_ice_surface_detection(ice_config)
            if not ice_workflow_id:
                self.logger.error("❌ Failed to start ice surface detection")
                return False
            
            self.logger.info(f"✅ Started ice surface detection: {ice_workflow_id}")
            
            # Step 6: Export results
            self.logger.info("📤 Step 6: Exporting results...")
            results = [
                "hockey_detection_results.json",
                "role_classification_results.json",
                "equipment_classification_results.json",
                "ice_surface_detection_results.json"
            ]
            
            uploaded_files = toolchain.export_hockey_results(results, "google_drive")
            if not uploaded_files:
                self.logger.error("❌ Failed to export results")
                return False
            
            self.logger.info(f"✅ Exported {len(uploaded_files)} results")
            
            # Step 7: Get pipeline status
            self.logger.info("📊 Step 7: Getting pipeline status...")
            status = toolchain.get_pipeline_status(workflow_id)
            if not status or 'error' in status:
                self.logger.error(f"❌ Failed to get pipeline status: {status}")
                return False
            
            self.logger.info(f"✅ Pipeline status: {status['status']}")
            
            # Step 8: Get component metrics
            self.logger.info("📈 Step 8: Getting component metrics...")
            metrics = toolchain.get_component_metrics()
            if not metrics or 'error' in metrics:
                self.logger.error(f"❌ Failed to get component metrics: {metrics}")
                return False
            
            self.logger.info(f"✅ Component metrics: {metrics['business_metrics']}")
            
            # Stop component
            toolchain.stop()
            
            self.logger.info("✅ Complete Hockey Detection Pipeline test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Complete hockey pipeline test failed: {e}")
            return False
    
    def test_spotlight_computer_vision(self) -> bool:
        """Test Spotlight computer vision capabilities"""
        try:
            self.logger.info("👁️ Testing Spotlight Computer Vision...")
            
            # Initialize Spotlight component
            spotlight = SpotlightComponent()
            spotlight.initialize({
                'model_type': 'yolov8n',
                'detection_confidence': 0.5,
                'max_detections': 100
            })
            spotlight.start()
            
            # Test 1: Hockey video analysis
            self.logger.info("🎥 Testing hockey video analysis...")
            video_config = {
                'analysis_type': 'hockey_video',
                'target_fps': 30,
                'detection_confidence': 0.5,
                'output_format': 'json'
            }
            
            result = spotlight.analyze_hockey_video('/path/to/hockey_video.mp4', video_config)
            if not result.get('workflow_id'):
                self.logger.error("❌ Failed to start hockey video analysis")
                return False
            
            self.logger.info(f"✅ Started hockey video analysis: {result['workflow_id']}")
            
            # Test 2: Hockey image analysis
            self.logger.info("🖼️ Testing hockey image analysis...")
            image_config = {
                'analysis_type': 'hockey_image',
                'detection_confidence': 0.5,
                'max_detections': 100,
                'output_format': 'json'
            }
            
            result = spotlight.analyze_hockey_image('/path/to/hockey_image.jpg', image_config)
            if not result.get('workflow_id'):
                self.logger.error("❌ Failed to start hockey image analysis")
                return False
            
            self.logger.info(f"✅ Started hockey image analysis: {result['workflow_id']}")
            
            # Test 3: Player detection
            self.logger.info("👤 Testing player detection...")
            detection_config = {
                'detection_type': 'hockey_players',
                'confidence_threshold': 0.5,
                'nms_threshold': 0.4,
                'output_format': 'json'
            }
            
            result = spotlight.detect_hockey_players('/path/to/hockey_media.jpg', detection_config)
            if not result.get('workflow_id'):
                self.logger.error("❌ Failed to start player detection")
                return False
            
            self.logger.info(f"✅ Started player detection: {result['workflow_id']}")
            
            # Test 4: Role classification
            self.logger.info("🎭 Testing role classification...")
            classification_config = {
                'classification_type': 'hockey_roles',
                'target_accuracy': 0.85,
                'role_types': ['player', 'goalie', 'referee'],
                'output_format': 'json'
            }
            
            result = spotlight.classify_hockey_roles('/path/to/hockey_media.jpg', classification_config)
            if not result.get('workflow_id'):
                self.logger.error("❌ Failed to start role classification")
                return False
            
            self.logger.info(f"✅ Started role classification: {result['workflow_id']}")
            
            # Test 5: Media import/export
            self.logger.info("📁 Testing media import/export...")
            files = spotlight.import_user_media("google_drive")
            self.logger.info(f"✅ Imported {len(files)} media files")
            
            results = ["analysis1.json", "analysis2.json", "analysis3.json"]
            uploaded_files = spotlight.export_analysis_results(results, "google_drive")
            self.logger.info(f"✅ Exported {len(uploaded_files)} analysis results")
            
            # Test 6: Get analysis status
            self.logger.info("📊 Testing analysis status...")
            status = spotlight.get_analysis_status(result['workflow_id'])
            if not status or 'error' in status:
                self.logger.error(f"❌ Failed to get analysis status: {status}")
                return False
            
            self.logger.info(f"✅ Analysis status: {status['status']}")
            
            # Stop component
            spotlight.stop()
            
            self.logger.info("✅ Spotlight Computer Vision test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Spotlight computer vision test failed: {e}")
            return False
    
    def test_autopilot_autonomous_systems(self) -> bool:
        """Test Autopilot autonomous systems"""
        try:
            self.logger.info("🤖 Testing Autopilot Autonomous Systems...")
            
            # Initialize Autopilot component
            autopilot = AutopilotComponent()
            autopilot.initialize({
                'learning_rate': 0.001,
                'adaptation_threshold': 0.8,
                'max_iterations': 1000
            })
            autopilot.start()
            
            # Test 1: Autonomous hockey analysis
            self.logger.info("🏒 Testing autonomous hockey analysis...")
            analysis_config = {
                'parameters': {
                    'analysis_type': 'autonomous_hockey',
                    'learning_rate': 0.001,
                    'adaptation_threshold': 0.8
                },
                'analysis_config': {
                    'target_accuracy': 0.85,
                    'max_iterations': 1000
                },
                'learning_config': {
                    'model_type': 'reinforcement_learning',
                    'reward_function': 'performance_based'
                },
                'adaptation_config': {
                    'adaptation_threshold': 0.8,
                    'adaptation_rate': 0.1
                }
            }
            
            workflow_id = autopilot.start_autonomous_hockey_analysis(analysis_config)
            if not workflow_id:
                self.logger.error("❌ Failed to start autonomous hockey analysis")
                return False
            
            self.logger.info(f"✅ Started autonomous hockey analysis: {workflow_id}")
            
            # Test 2: Autonomous learning system
            self.logger.info("🧠 Testing autonomous learning system...")
            learning_config = {
                'parameters': {
                    'learning_type': 'autonomous',
                    'model_type': 'reinforcement_learning'
                },
                'learning_config': {
                    'exploration_rate': 0.1,
                    'exploitation_rate': 0.9
                },
                'model_config': {
                    'model_type': 'deep_q_network',
                    'hidden_layers': [128, 64, 32]
                },
                'environment_config': {
                    'state_space': 'continuous',
                    'action_space': 'discrete'
                }
            }
            
            learning_workflow_id = autopilot.start_autonomous_learning_system(learning_config)
            if not learning_workflow_id:
                self.logger.error("❌ Failed to start autonomous learning system")
                return False
            
            self.logger.info(f"✅ Started autonomous learning system: {learning_workflow_id}")
            
            # Test 3: Autonomous optimization
            self.logger.info("⚡ Testing autonomous optimization...")
            optimization_config = {
                'parameters': {
                    'optimization_type': 'autonomous',
                    'objective_function': 'performance_maximization'
                },
                'optimization_config': {
                    'algorithm': 'genetic_algorithm',
                    'population_size': 100,
                    'generations': 50
                },
                'objective_config': {
                    'primary_objective': 'accuracy',
                    'secondary_objective': 'efficiency'
                },
                'constraint_config': {
                    'max_computation_time': 3600,
                    'max_memory_usage': 8 * 1024 * 1024 * 1024  # 8GB
                }
            }
            
            optimization_workflow_id = autopilot.start_autonomous_optimization(optimization_config)
            if not optimization_workflow_id:
                self.logger.error("❌ Failed to start autonomous optimization")
                return False
            
            self.logger.info(f"✅ Started autonomous optimization: {optimization_workflow_id}")
            
            # Test 4: System adaptation
            self.logger.info("🔄 Testing system adaptation...")
            performance_metrics = {
                'accuracy': 0.75,
                'precision': 0.72,
                'recall': 0.78,
                'f1_score': 0.75
            }
            
            adapted = autopilot.adapt_system_parameters(workflow_id, performance_metrics)
            if not adapted:
                self.logger.error("❌ Failed to adapt system parameters")
                return False
            
            self.logger.info(f"✅ System adaptation: {adapted}")
            
            # Test 5: Get adaptation history
            self.logger.info("📚 Testing adaptation history...")
            history = autopilot.get_adaptation_history(workflow_id)
            self.logger.info(f"✅ Adaptation history: {len(history)} entries")
            
            # Test 6: Get learning models
            self.logger.info("🤖 Testing learning models...")
            models = autopilot.get_learning_models()
            self.logger.info(f"✅ Learning models: {models}")
            
            # Test 7: Get autonomous system status
            self.logger.info("📊 Testing autonomous system status...")
            status = autopilot.get_autonomous_system_status(workflow_id)
            if not status or 'error' in status:
                self.logger.error(f"❌ Failed to get autonomous system status: {status}")
                return False
            
            self.logger.info(f"✅ Autonomous system status: {status['status']}")
            
            # Stop component
            autopilot.stop()
            
            self.logger.info("✅ Autopilot Autonomous Systems test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Autopilot autonomous systems test failed: {e}")
            return False
    
    def test_sherlock_security_investigation(self) -> bool:
        """Test Sherlock security investigation"""
        try:
            self.logger.info("🕵️ Testing Sherlock Security Investigation...")
            
            # Initialize Sherlock component
            sherlock = SherlockComponent()
            sherlock.initialize({
                'investigation_type': 'security',
                'threat_level': 'medium',
                'monitoring_interval': 60
            })
            sherlock.start()
            
            # Test 1: Security investigation
            self.logger.info("🔍 Testing security investigation...")
            incident_config = {
                'parameters': {
                    'incident_type': 'data_breach',
                    'severity': 'high',
                    'threat_level': 'critical'
                },
                'incident_config': {
                    'affected_systems': ['database', 'api', 'web'],
                    'data_compromised': ['user_data', 'financial_data']
                },
                'investigation_config': {
                    'scope': 'full',
                    'forensic_analysis': True
                },
                'threat_config': {
                    'threat_indicators': ['suspicious_login', 'data_exfiltration'],
                    'confidence_threshold': 0.8
                }
            }
            
            workflow_id = sherlock.start_security_investigation(incident_config)
            if not workflow_id:
                self.logger.error("❌ Failed to start security investigation")
                return False
            
            self.logger.info(f"✅ Started security investigation: {workflow_id}")
            
            # Test 2: Threat analysis
            self.logger.info("⚠️ Testing threat analysis...")
            threat_data = {
                'parameters': {
                    'threat_type': 'malware',
                    'analysis_depth': 'deep'
                },
                'threat_type': 'malware',
                'indicators': ['suspicious_process', 'network_anomaly'],
                'confidence_threshold': 0.7,
                'analysis_config': {
                    'scan_depth': 'full',
                    'behavioral_analysis': True
                }
            }
            
            result = sherlock.analyze_security_threats(threat_data)
            if not result.get('workflow_id'):
                self.logger.error("❌ Failed to start threat analysis")
                return False
            
            self.logger.info(f"✅ Started threat analysis: {result['workflow_id']}")
            
            # Test 3: Data breach investigation
            self.logger.info("💥 Testing data breach investigation...")
            breach_config = {
                'parameters': {
                    'breach_type': 'unauthorized_access',
                    'data_affected': ['user_credentials', 'personal_data']
                },
                'breach_type': 'unauthorized_access',
                'data_affected': ['user_credentials', 'personal_data'],
                'scope': 'user_database',
                'compromise_level': 'high',
                'investigation_config': {
                    'forensic_analysis': True,
                    'timeline_reconstruction': True
                }
            }
            
            breach_workflow_id = sherlock.investigate_data_breach(breach_config)
            if not breach_workflow_id:
                self.logger.error("❌ Failed to start data breach investigation")
                return False
            
            self.logger.info(f"✅ Started data breach investigation: {breach_workflow_id}")
            
            # Test 4: Security monitoring
            self.logger.info("📡 Testing security monitoring...")
            monitoring_config = {
                'parameters': {
                    'type': 'continuous',
                    'scope': 'full_system'
                },
                'type': 'continuous',
                'scope': 'full_system',
                'alert_threshold': 0.8,
                'interval': 60,
                'alert_config': {
                    'notification_channels': ['email', 'slack'],
                    'escalation_levels': ['low', 'medium', 'high', 'critical']
                }
            }
            
            monitoring_workflow_id = sherlock.monitor_system_security(monitoring_config)
            if not monitoring_workflow_id:
                self.logger.error("❌ Failed to start security monitoring")
                return False
            
            self.logger.info(f"✅ Started security monitoring: {monitoring_workflow_id}")
            
            # Test 5: Forensic evidence collection
            self.logger.info("🔬 Testing forensic evidence collection...")
            evidence_config = {
                'parameters': {
                    'type': 'digital',
                    'scope': 'full'
                },
                'type': 'digital',
                'scope': 'full',
                'chain_of_custody': True,
                'format': 'standard',
                'sources': ['system_logs', 'network_traffic', 'user_activity']
            }
            
            evidence_files = sherlock.collect_forensic_evidence(evidence_config)
            if not evidence_files:
                self.logger.error("❌ Failed to collect forensic evidence")
                return False
            
            self.logger.info(f"✅ Collected {len(evidence_files)} evidence files")
            
            # Test 6: Security report generation
            self.logger.info("📄 Testing security report generation...")
            report_config = {
                'parameters': {
                    'type': 'investigation',
                    'format': 'pdf'
                },
                'type': 'investigation',
                'format': 'pdf',
                'scope': 'full',
                'confidentiality': 'internal'
            }
            
            report_id = sherlock.generate_security_report(report_config)
            if not report_id:
                self.logger.error("❌ Failed to generate security report")
                return False
            
            self.logger.info(f"✅ Generated security report: {report_id}")
            
            # Test 7: Get investigation status
            self.logger.info("📊 Testing investigation status...")
            status = sherlock.get_investigation_status(workflow_id)
            if not status or 'error' in status:
                self.logger.error(f"❌ Failed to get investigation status: {status}")
                return False
            
            self.logger.info(f"✅ Investigation status: {status['status']}")
            
            # Test 8: Get security incidents
            self.logger.info("🚨 Testing security incidents...")
            incidents = sherlock.get_security_incidents()
            self.logger.info(f"✅ Security incidents: {len(incidents)}")
            
            # Test 9: Get threat intelligence
            self.logger.info("🧠 Testing threat intelligence...")
            intelligence = sherlock.get_threat_intelligence()
            self.logger.info(f"✅ Threat intelligence: {intelligence}")
            
            # Stop component
            sherlock.stop()
            
            self.logger.info("✅ Sherlock Security Investigation test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Sherlock security investigation test failed: {e}")
            return False
    
    def test_watson_nlp_ai(self) -> bool:
        """Test Watson NLP and AI capabilities"""
        try:
            self.logger.info("🧠 Testing Watson NLP and AI...")
            
            # Initialize Watson component
            watson = WatsonComponent()
            watson.initialize({
                'nlp_model': 'gpt-4',
                'language': 'en',
                'confidence_threshold': 0.7
            })
            watson.start()
            
            # Test 1: Natural language processing
            self.logger.info("💬 Testing natural language processing...")
            nlp_config = {
                'language': 'en',
                'type': 'general',
                'confidence_threshold': 0.7,
                'output_format': 'json'
            }
            
            result = watson.process_natural_language("Hello, how are you today?", nlp_config)
            if not result.get('workflow_id'):
                self.logger.error("❌ Failed to start NLP processing")
                return False
            
            self.logger.info(f"✅ Started NLP processing: {result['workflow_id']}")
            
            # Test 2: Sentiment analysis
            self.logger.info("😊 Testing sentiment analysis...")
            sentiment_config = {
                'model': 'default',
                'confidence_threshold': 0.7,
                'emotion_detection': True,
                'output_format': 'json'
            }
            
            result = watson.analyze_sentiment("I love this product! It's amazing!", sentiment_config)
            if not result.get('workflow_id'):
                self.logger.error("❌ Failed to start sentiment analysis")
                return False
            
            self.logger.info(f"✅ Started sentiment analysis: {result['workflow_id']}")
            
            # Test 3: Entity extraction
            self.logger.info("🏷️ Testing entity extraction...")
            entity_config = {
                'entity_types': ['person', 'organization', 'location'],
                'confidence_threshold': 0.7,
                'depth': 'standard',
                'output_format': 'json'
            }
            
            result = watson.extract_entities("John Smith works at Microsoft in Seattle.", entity_config)
            if not result.get('workflow_id'):
                self.logger.error("❌ Failed to start entity extraction")
                return False
            
            self.logger.info(f"✅ Started entity extraction: {result['workflow_id']}")
            
            # Test 4: Text generation
            self.logger.info("✍️ Testing text generation...")
            generation_config = {
                'model_type': 'gpt',
                'max_length': 100,
                'temperature': 0.7,
                'output_format': 'text'
            }
            
            result = watson.generate_text("Write a short story about", generation_config)
            if not result.get('workflow_id'):
                self.logger.error("❌ Failed to start text generation")
                return False
            
            self.logger.info(f"✅ Started text generation: {result['workflow_id']}")
            
            # Test 5: Text translation
            self.logger.info("🌍 Testing text translation...")
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
            if not result.get('workflow_id'):
                self.logger.error("❌ Failed to start text translation")
                return False
            
            self.logger.info(f"✅ Started text translation: {result['workflow_id']}")
            
            # Test 6: AI chat
            self.logger.info("💭 Testing AI chat...")
            chat_config = {
                'model': 'gpt-4',
                'context': 'general',
                'style': 'helpful',
                'output_format': 'text'
            }
            
            result = watson.chat_with_ai("What is artificial intelligence?", chat_config)
            if not result.get('workflow_id'):
                self.logger.error("❌ Failed to start AI chat")
                return False
            
            self.logger.info(f"✅ Started AI chat: {result['workflow_id']}")
            
            # Test 7: Get NLP task status
            self.logger.info("📊 Testing NLP task status...")
            status = watson.get_nlp_task_status(result['workflow_id'])
            if not status or 'error' in status:
                self.logger.error(f"❌ Failed to get NLP task status: {status}")
                return False
            
            self.logger.info(f"✅ NLP task status: {status['status']}")
            
            # Test 8: Get conversation history
            self.logger.info("💬 Testing conversation history...")
            history = watson.get_conversation_history(10)
            self.logger.info(f"✅ Conversation history: {len(history)} messages")
            
            # Test 9: Get AI models
            self.logger.info("🤖 Testing AI models...")
            models = watson.get_ai_models()
            self.logger.info(f"✅ AI models: {models}")
            
            # Stop component
            watson.stop()
            
            self.logger.info("✅ Watson NLP and AI test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Watson NLP and AI test failed: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all integration tests"""
        self.logger.info("🚀 Starting Jarvis Integration Test Suite...")
        
        tests = {
            "Complete Hockey Pipeline": self.test_complete_hockey_pipeline,
            "Spotlight Computer Vision": self.test_spotlight_computer_vision,
            "Autopilot Autonomous Systems": self.test_autopilot_autonomous_systems,
            "Sherlock Security Investigation": self.test_sherlock_security_investigation,
            "Watson NLP and AI": self.test_watson_nlp_ai
        }
        
        results = {}
        for test_name, test_func in tests.items():
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Running {test_name} Integration Test")
            self.logger.info(f"{'='*60}")
            
            try:
                result = test_func()
                results[test_name] = result
                status = "✅ PASSED" if result else "❌ FAILED"
                self.logger.info(f"{test_name}: {status}")
            except Exception as e:
                self.logger.error(f"❌ {test_name} test crashed: {e}")
                results[test_name] = False
        
        return results
    
    def print_summary(self, results: Dict[str, bool]):
        """Print test summary"""
        self.logger.info(f"\n{'='*70}")
        self.logger.info("JARVIS INTEGRATION TEST SUMMARY")
        self.logger.info(f"{'='*70}")
        
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            self.logger.info(f"{test_name:<35} {status}")
        
        self.logger.info(f"{'='*70}")
        self.logger.info(f"TOTAL: {passed}/{total} integration tests passed")
        
        if passed == total:
            self.logger.info("🎉 ALL INTEGRATION TESTS PASSED! Jarvis Core Services are fully integrated.")
        else:
            self.logger.error(f"⚠️  {total - passed} integration tests failed. Please check the logs above.")
        
        self.logger.info(f"{'='*70}")


def main():
    """Main integration test function"""
    tester = JarvisIntegrationTester()
    results = tester.run_all_tests()
    tester.print_summary(results)
    
    # Return exit code based on results
    all_passed = all(results.values())
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
