#!/usr/bin/env python3
"""
TSAI Sherlock Component - Security and Investigation Integration
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

class SherlockComponent(TSAIComponent):
    """TSAI Sherlock component for security and investigation"""
    
    def __init__(self, jarvis_core: 'JarvisCoreServices' = None):
        super().__init__("sherlock", jarvis_core)
        self.active_investigations = {}
        self.security_incidents = {}
        self.threat_intelligence = {}
    
    def start_security_investigation(self, incident_config: Dict[str, Any]) -> str:
        """Start security investigation"""
        try:
            # Start experiment
            experiment_id = self.start_experiment(
                "security-investigation", 
                incident_config.get("parameters", {})
            )
            
            # Log parameters
            self.log_params({
                'investigation_type': 'security',
                'incident_severity': incident_config.get('severity', 'medium'),
                'threat_level': incident_config.get('threat_level', 'medium'),
                'investigation_scope': incident_config.get('scope', 'full')
            })
            
            # Start workflow
            workflow_id = asyncio.run(self.start_workflow(
                "security-investigation", 
                {
                    'incident_config': incident_config.get('incident_config', {}),
                    'investigation_config': incident_config.get('investigation_config', {}),
                    'threat_config': incident_config.get('threat_config', {})
                }
            ))
            
            # Store investigation info
            self.active_investigations[workflow_id] = {
                'experiment_id': experiment_id,
                'incident_type': incident_config.get('incident_type', 'unknown'),
                'severity': incident_config.get('severity', 'medium'),
                'started_at': datetime.now().isoformat(),
                'status': 'running',
                'evidence_count': 0
            }
            
            # Log business metrics
            self.log_business_metrics({
                'user_engagement': ['security_investigation_started'],
                'pipeline_success_rate': 1.0
            })
            
            self.logger.info(f"Started security investigation: {workflow_id}")
            return workflow_id
            
        except Exception as e:
            self.logger.error(f"Failed to start security investigation: {e}")
            raise
    
    def analyze_security_threats(self, threat_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze security threats"""
        try:
            # Start experiment
            experiment_id = self.start_experiment(
                "threat-analysis", 
                threat_data.get("parameters", {})
            )
            
            # Log parameters
            self.log_params({
                'threat_type': threat_data.get('threat_type', 'unknown'),
                'analysis_depth': threat_data.get('analysis_depth', 'standard'),
                'threat_indicators': threat_data.get('indicators', []),
                'confidence_threshold': threat_data.get('confidence_threshold', 0.7)
            })
            
            # Start workflow
            workflow_id = asyncio.run(self.start_workflow(
                "threat-analysis", 
                {
                    'threat_data': threat_data,
                    'analysis_config': threat_data.get('analysis_config', {}),
                    'response_config': threat_data.get('response_config', {})
                }
            ))
            
            # Store threat analysis info
            self.active_investigations[workflow_id] = {
                'experiment_id': experiment_id,
                'threat_type': threat_data.get('threat_type', 'unknown'),
                'started_at': datetime.now().isoformat(),
                'status': 'running',
                'threat_indicators': threat_data.get('indicators', [])
            }
            
            self.logger.info(f"Started threat analysis: {workflow_id}")
            return {
                'workflow_id': workflow_id,
                'experiment_id': experiment_id,
                'status': 'started'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to analyze security threats: {e}")
            raise
    
    def investigate_data_breach(self, breach_config: Dict[str, Any]) -> str:
        """Investigate data breach incident"""
        try:
            # Start experiment
            experiment_id = self.start_experiment(
                "data-breach-investigation", 
                breach_config.get("parameters", {})
            )
            
            # Log parameters
            self.log_params({
                'breach_type': breach_config.get('breach_type', 'unknown'),
                'data_affected': breach_config.get('data_affected', []),
                'breach_scope': breach_config.get('scope', 'unknown'),
                'compromise_level': breach_config.get('compromise_level', 'medium')
            })
            
            # Start workflow
            workflow_id = asyncio.run(self.start_workflow(
                "data-breach-investigation", 
                {
                    'breach_config': breach_config,
                    'investigation_config': breach_config.get('investigation_config', {}),
                    'forensic_config': breach_config.get('forensic_config', {})
                }
            ))
            
            # Store breach investigation info
            self.active_investigations[workflow_id] = {
                'experiment_id': experiment_id,
                'breach_type': breach_config.get('breach_type', 'unknown'),
                'started_at': datetime.now().isoformat(),
                'status': 'running',
                'data_affected': breach_config.get('data_affected', [])
            }
            
            self.logger.info(f"Started data breach investigation: {workflow_id}")
            return workflow_id
            
        except Exception as e:
            self.logger.error(f"Failed to investigate data breach: {e}")
            raise
    
    def monitor_system_security(self, monitoring_config: Dict[str, Any]) -> str:
        """Monitor system security continuously"""
        try:
            # Start experiment
            experiment_id = self.start_experiment(
                "security-monitoring", 
                monitoring_config.get("parameters", {})
            )
            
            # Log parameters
            self.log_params({
                'monitoring_type': monitoring_config.get('type', 'continuous'),
                'monitoring_scope': monitoring_config.get('scope', 'full'),
                'alert_threshold': monitoring_config.get('alert_threshold', 0.8),
                'monitoring_interval': monitoring_config.get('interval', 60)
            })
            
            # Start workflow
            workflow_id = asyncio.run(self.start_workflow(
                "security-monitoring", 
                {
                    'monitoring_config': monitoring_config,
                    'alert_config': monitoring_config.get('alert_config', {}),
                    'response_config': monitoring_config.get('response_config', {})
                }
            ))
            
            # Store monitoring info
            self.active_investigations[workflow_id] = {
                'experiment_id': experiment_id,
                'monitoring_type': monitoring_config.get('type', 'continuous'),
                'started_at': datetime.now().isoformat(),
                'status': 'running',
                'alerts_generated': 0
            }
            
            self.logger.info(f"Started security monitoring: {workflow_id}")
            return workflow_id
            
        except Exception as e:
            self.logger.error(f"Failed to start security monitoring: {e}")
            raise
    
    def collect_forensic_evidence(self, evidence_config: Dict[str, Any]) -> List[str]:
        """Collect forensic evidence"""
        try:
            # Start experiment
            experiment_id = self.start_experiment(
                "forensic-evidence-collection", 
                evidence_config.get("parameters", {})
            )
            
            # Log parameters
            self.log_params({
                'evidence_type': evidence_config.get('type', 'digital'),
                'collection_scope': evidence_config.get('scope', 'full'),
                'chain_of_custody': evidence_config.get('chain_of_custody', True),
                'evidence_format': evidence_config.get('format', 'standard')
            })
            
            # Collect evidence
            evidence_files = []
            for source in evidence_config.get('sources', []):
                evidence_file = f"evidence_{source}_{int(time.time())}.json"
                evidence_files.append(evidence_file)
                
                # Store evidence
                self.store_artifact(evidence_file, {
                    'evidence_type': evidence_config.get('type', 'digital'),
                    'source': source,
                    'collected_at': datetime.now().isoformat(),
                    'chain_of_custody': evidence_config.get('chain_of_custody', True)
                })
            
            # Log evidence collection metrics
            self.log_metrics({
                'evidence_collected': len(evidence_files),
                'collection_time': time.time(),
                'evidence_sources': len(evidence_config.get('sources', []))
            })
            
            # Log business metrics
            self.log_business_metrics({
                'user_engagement': ['evidence_collected'],
                'pipeline_success_rate': 1.0 if evidence_files else 0.0
            })
            
            self.logger.info(f"Collected {len(evidence_files)} evidence files")
            return evidence_files
            
        except Exception as e:
            self.logger.error(f"Failed to collect forensic evidence: {e}")
            return []
    
    def generate_security_report(self, report_config: Dict[str, Any]) -> str:
        """Generate security investigation report"""
        try:
            # Start experiment
            experiment_id = self.start_experiment(
                "security-report-generation", 
                report_config.get("parameters", {})
            )
            
            # Log parameters
            self.log_params({
                'report_type': report_config.get('type', 'investigation'),
                'report_format': report_config.get('format', 'pdf'),
                'report_scope': report_config.get('scope', 'full'),
                'confidentiality_level': report_config.get('confidentiality', 'internal')
            })
            
            # Generate report
            report_filename = f"security_report_{int(time.time())}.{report_config.get('format', 'pdf')}"
            
            # Store report
            report_id = self.store_artifact(report_filename, {
                'report_type': report_config.get('type', 'investigation'),
                'generated_at': datetime.now().isoformat(),
                'confidentiality': report_config.get('confidentiality', 'internal'),
                'scope': report_config.get('scope', 'full')
            })
            
            # Log report generation metrics
            self.log_metrics({
                'report_generated': 1.0,
                'report_type': report_config.get('type', 'investigation'),
                'generation_time': time.time()
            })
            
            self.logger.info(f"Generated security report: {report_id}")
            return report_id
            
        except Exception as e:
            self.logger.error(f"Failed to generate security report: {e}")
            raise
    
    def get_investigation_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get investigation status"""
        try:
            if workflow_id in self.active_investigations:
                investigation_info = self.active_investigations[workflow_id]
                
                # Get workflow status
                status = asyncio.run(self.get_workflow_status(workflow_id))
                
                return {
                    'workflow_id': workflow_id,
                    'experiment_id': investigation_info['experiment_id'],
                    'investigation_type': investigation_info.get('incident_type', investigation_info.get('threat_type', investigation_info.get('breach_type', 'unknown'))),
                    'status': status['status'],
                    'started_at': investigation_info['started_at'],
                    'evidence_count': investigation_info.get('evidence_count', 0),
                    'alerts_generated': investigation_info.get('alerts_generated', 0)
                }
            else:
                return {'error': 'Investigation not found'}
                
        except Exception as e:
            self.logger.error(f"Failed to get investigation status: {e}")
            return {'error': str(e)}
    
    def get_security_incidents(self) -> List[Dict[str, Any]]:
        """Get security incidents"""
        try:
            # Get incidents from storage
            incidents = []
            for workflow_id, info in self.active_investigations.items():
                incident = {
                    'workflow_id': workflow_id,
                    'incident_type': info.get('incident_type', info.get('threat_type', info.get('breach_type', 'unknown'))),
                    'severity': info.get('severity', 'medium'),
                    'started_at': info['started_at'],
                    'status': info['status']
                }
                incidents.append(incident)
            
            return incidents
            
        except Exception as e:
            self.logger.error(f"Failed to get security incidents: {e}")
            return []
    
    def get_threat_intelligence(self) -> Dict[str, Any]:
        """Get threat intelligence"""
        try:
            # Get threat intelligence from storage
            threats = []
            for workflow_id, info in self.active_investigations.items():
                if 'threat_indicators' in info:
                    threats.extend(info['threat_indicators'])
            
            return {
                'threat_indicators': threats,
                'total_threats': len(threats),
                'active_investigations': len(self.active_investigations),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get threat intelligence: {e}")
            return {'error': str(e)}
    
    def get_component_metrics(self) -> Dict[str, Any]:
        """Get component-specific metrics"""
        try:
            # Get system metrics
            system_metrics = {
                'cpu_usage': 25.3,  # Mock data
                'memory_usage': 512 * 1024 * 1024,  # 512MB
                'disk_usage': 25 * 1024 * 1024 * 1024,  # 25GB
                'network_in': 500,
                'network_out': 300
            }
            
            # Log system metrics
            self.log_system_metrics(system_metrics)
            
            # Get business metrics
            business_metrics = {
                'active_investigations': len(self.active_investigations),
                'security_incidents': len(self.get_security_incidents()),
                'total_experiments': len(self.get_experiment_runs()),
                'total_artifacts': len(self.list_artifacts()),
                'total_models': len(self.list_models())
            }
            
            return {
                'component_name': self.component_name,
                'system_metrics': system_metrics,
                'business_metrics': business_metrics,
                'security_metrics': {
                    'active_investigations': len(self.active_investigations),
                    'threat_indicators': len(self.get_threat_intelligence().get('threat_indicators', [])),
                    'evidence_collected': sum(info.get('evidence_count', 0) for info in self.active_investigations.values())
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get component metrics: {e}")
            return {'error': str(e)}


def main():
    """Main function for Sherlock component"""
    
    # Initialize Sherlock component
    sherlock = SherlockComponent()
    
    # Health check
    health = sherlock.health_check()
    print(f"Sherlock Component Health: {health}")
    
    # Initialize component
    sherlock.initialize({
        'investigation_type': 'security',
        'threat_level': 'medium',
        'monitoring_interval': 60
    })
    
    # Start component
    sherlock.start()
    
    # Test security investigation
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
    print(f"✅ Started security investigation: {workflow_id}")
    
    # Test threat analysis
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
    print(f"✅ Started threat analysis: {result['workflow_id']}")
    
    # Test data breach investigation
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
    
    workflow_id = sherlock.investigate_data_breach(breach_config)
    print(f"✅ Started data breach investigation: {workflow_id}")
    
    # Test security monitoring
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
    
    workflow_id = sherlock.monitor_system_security(monitoring_config)
    print(f"✅ Started security monitoring: {workflow_id}")
    
    # Test forensic evidence collection
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
    print(f"✅ Collected {len(evidence_files)} evidence files")
    
    # Test security report generation
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
    print(f"✅ Generated security report: {report_id}")
    
    # Test investigation status
    status = sherlock.get_investigation_status(workflow_id)
    print(f"✅ Investigation status: {status}")
    
    # Test security incidents
    incidents = sherlock.get_security_incidents()
    print(f"✅ Security incidents: {len(incidents)}")
    
    # Test threat intelligence
    intelligence = sherlock.get_threat_intelligence()
    print(f"✅ Threat intelligence: {intelligence}")
    
    # Get component metrics
    metrics = sherlock.get_component_metrics()
    print(f"✅ Component metrics: {metrics}")
    
    # Stop component
    sherlock.stop()
    print("✅ Sherlock component lifecycle completed")

if __name__ == "__main__":
    main()
