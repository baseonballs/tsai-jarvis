#!/usr/bin/env python3
"""
Jarvis Analytics Service - Prometheus/Grafana Integration
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta
import uuid

from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server
import requests
import yaml

class JarvisAnalyticsService:
    """Jarvis analytics service for metrics and monitoring"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.getenv('JARVIS_CONFIG_PATH', 'config/jarvis-core.yaml')
        self.config = self._load_config()
        self.prometheus_client = self._create_prometheus_client()
        self.grafana_client = self._create_grafana_client()
        self.logger = logging.getLogger(__name__)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Initialize metrics
        self._initialize_metrics()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config['jarvis_core']['services']['analytics']
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            # Return default config
            return {
                'type': 'prometheus',
                'endpoint': 'http://prometheus:9090',
                'dashboard': 'http://grafana:3000'
            }
    
    def _create_prometheus_client(self):
        """Create Prometheus client"""
        try:
            # Start Prometheus metrics server
            start_http_server(8080)
            self.logger.info("Started Prometheus metrics server on port 8080")
            return True
        except Exception as e:
            self.logger.error(f"Failed to create Prometheus client: {e}")
            return False
    
    def _create_grafana_client(self):
        """Create Grafana client"""
        try:
            # Test Grafana connection
            response = requests.get(f"{self.config['dashboard']}/api/health", timeout=5)
            if response.status_code == 200:
                self.logger.info("Grafana connection successful")
                return True
            else:
                self.logger.warning(f"Grafana connection failed: {response.status_code}")
                return False
        except Exception as e:
            self.logger.warning(f"Grafana connection failed: {e}")
            return False
    
    def _initialize_metrics(self):
        """Initialize Prometheus metrics"""
        try:
            # System Metrics
            self.cpu_usage = Gauge('jarvis_cpu_usage_percent', 'CPU usage percentage', ['component'])
            self.memory_usage = Gauge('jarvis_memory_usage_bytes', 'Memory usage in bytes', ['component'])
            self.disk_usage = Gauge('jarvis_disk_usage_bytes', 'Disk usage in bytes', ['component'])
            self.network_io = Counter('jarvis_network_io_bytes_total', 'Network I/O in bytes', ['component', 'direction'])
            
            # Application Metrics
            self.request_count = Counter('jarvis_requests_total', 'Total requests', ['component', 'endpoint', 'method', 'status'])
            self.request_duration = Histogram('jarvis_request_duration_seconds', 'Request duration', ['component', 'endpoint'])
            self.error_rate = Gauge('jarvis_error_rate', 'Error rate', ['component'])
            self.active_connections = Gauge('jarvis_active_connections', 'Active connections', ['component'])
            
            # Business Metrics
            self.experiment_count = Counter('jarvis_experiments_total', 'Total experiments', ['component', 'status'])
            self.model_training_time = Histogram('jarvis_model_training_duration_seconds', 'Model training duration', ['component', 'model_type'])
            self.pipeline_success_rate = Gauge('jarvis_pipeline_success_rate', 'Pipeline success rate', ['component', 'pipeline_type'])
            self.user_engagement = Counter('jarvis_user_engagement_total', 'User engagement', ['component', 'action'])
            
            # Workflow Metrics
            self.workflow_started = Counter('jarvis_workflow_started_total', 'Workflows started', ['component', 'workflow_type'])
            self.workflow_completed = Counter('jarvis_workflow_completed_total', 'Workflows completed', ['component', 'workflow_type', 'status'])
            self.workflow_duration = Histogram('jarvis_workflow_duration_seconds', 'Workflow duration', ['component', 'workflow_type'])
            
            # Activity Metrics
            self.activity_started = Counter('jarvis_activity_started_total', 'Activities started', ['component', 'activity_type'])
            self.activity_completed = Counter('jarvis_activity_completed_total', 'Activities completed', ['component', 'activity_type', 'status'])
            self.activity_duration = Histogram('jarvis_activity_duration_seconds', 'Activity duration', ['component', 'activity_type'])
            
            self.logger.info("Initialized Prometheus metrics")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize metrics: {e}")
    
    def log_metric(self, metric_name: str, value: float, 
                   labels: Dict[str, str] = None, metric_type: str = "gauge"):
        """Log metric to Prometheus"""
        try:
            labels = labels or {}
            
            if metric_type == "counter":
                counter = Counter(f'jarvis_{metric_name}_total', f'Total {metric_name}', list(labels.keys()))
                counter.labels(**labels).inc(value)
            elif metric_type == "histogram":
                histogram = Histogram(f'jarvis_{metric_name}_seconds', f'{metric_name} duration', list(labels.keys()))
                histogram.labels(**labels).observe(value)
            elif metric_type == "gauge":
                gauge = Gauge(f'jarvis_{metric_name}', f'{metric_name}', list(labels.keys()))
                gauge.labels(**labels).set(value)
            elif metric_type == "summary":
                summary = Summary(f'jarvis_{metric_name}_summary', f'{metric_name} summary', list(labels.keys()))
                summary.labels(**labels).observe(value)
            
            self.logger.info(f"Logged metric {metric_name}: {value}")
            
        except Exception as e:
            self.logger.error(f"Failed to log metric {metric_name}: {e}")
    
    def log_counter(self, counter_name: str, value: float = 1.0, 
                    labels: Dict[str, str] = None):
        """Log counter metric"""
        try:
            labels = labels or {}
            self.log_metric(counter_name, value, labels, "counter")
        except Exception as e:
            self.logger.error(f"Failed to log counter {counter_name}: {e}")
    
    def log_histogram(self, histogram_name: str, value: float, 
                      labels: Dict[str, str] = None):
        """Log histogram metric"""
        try:
            labels = labels or {}
            self.log_metric(histogram_name, value, labels, "histogram")
        except Exception as e:
            self.logger.error(f"Failed to log histogram {histogram_name}: {e}")
    
    def log_gauge(self, gauge_name: str, value: float, 
                  labels: Dict[str, str] = None):
        """Log gauge metric"""
        try:
            labels = labels or {}
            self.log_metric(gauge_name, value, labels, "gauge")
        except Exception as e:
            self.logger.error(f"Failed to log gauge {gauge_name}: {e}")
    
    def log_system_metrics(self, component: str, metrics: Dict[str, float]):
        """Log system metrics"""
        try:
            # CPU usage
            if 'cpu_usage' in metrics:
                self.cpu_usage.labels(component=component).set(metrics['cpu_usage'])
            
            # Memory usage
            if 'memory_usage' in metrics:
                self.memory_usage.labels(component=component).set(metrics['memory_usage'])
            
            # Disk usage
            if 'disk_usage' in metrics:
                self.disk_usage.labels(component=component).set(metrics['disk_usage'])
            
            # Network I/O
            if 'network_in' in metrics:
                self.network_io.labels(component=component, direction='in').inc(metrics['network_in'])
            if 'network_out' in metrics:
                self.network_io.labels(component=component, direction='out').inc(metrics['network_out'])
            
            self.logger.info(f"Logged system metrics for {component}")
            
        except Exception as e:
            self.logger.error(f"Failed to log system metrics for {component}: {e}")
    
    def log_application_metrics(self, component: str, endpoint: str, 
                              method: str, status: str, duration: float):
        """Log application metrics"""
        try:
            # Request count
            self.request_count.labels(
                component=component,
                endpoint=endpoint,
                method=method,
                status=status
            ).inc()
            
            # Request duration
            self.request_duration.labels(
                component=component,
                endpoint=endpoint
            ).observe(duration)
            
            # Error rate calculation
            if status.startswith('4') or status.startswith('5'):
                self.error_rate.labels(component=component).set(1.0)
            else:
                self.error_rate.labels(component=component).set(0.0)
            
            self.logger.info(f"Logged application metrics for {component}")
            
        except Exception as e:
            self.logger.error(f"Failed to log application metrics for {component}: {e}")
    
    def log_business_metrics(self, component: str, metrics: Dict[str, Any]):
        """Log business metrics"""
        try:
            # Experiment metrics
            if 'experiments' in metrics:
                for experiment in metrics['experiments']:
                    self.experiment_count.labels(
                        component=component,
                        status=experiment.get('status', 'unknown')
                    ).inc()
            
            # Model training metrics
            if 'model_training' in metrics:
                training = metrics['model_training']
                self.model_training_time.labels(
                    component=component,
                    model_type=training.get('model_type', 'unknown')
                ).observe(training.get('duration', 0))
            
            # Pipeline success rate
            if 'pipeline_success_rate' in metrics:
                self.pipeline_success_rate.labels(
                    component=component,
                    pipeline_type=metrics.get('pipeline_type', 'unknown')
                ).set(metrics['pipeline_success_rate'])
            
            # User engagement
            if 'user_engagement' in metrics:
                for action in metrics['user_engagement']:
                    self.user_engagement.labels(
                        component=component,
                        action=action
                    ).inc()
            
            self.logger.info(f"Logged business metrics for {component}")
            
        except Exception as e:
            self.logger.error(f"Failed to log business metrics for {component}: {e}")
    
    def log_workflow_metrics(self, component: str, workflow_type: str, 
                           status: str, duration: float):
        """Log workflow metrics"""
        try:
            # Workflow started
            self.workflow_started.labels(
                component=component,
                workflow_type=workflow_type
            ).inc()
            
            # Workflow completed
            self.workflow_completed.labels(
                component=component,
                workflow_type=workflow_type,
                status=status
            ).inc()
            
            # Workflow duration
            self.workflow_duration.labels(
                component=component,
                workflow_type=workflow_type
            ).observe(duration)
            
            self.logger.info(f"Logged workflow metrics for {component}")
            
        except Exception as e:
            self.logger.error(f"Failed to log workflow metrics for {component}: {e}")
    
    def log_activity_metrics(self, component: str, activity_type: str, 
                           status: str, duration: float):
        """Log activity metrics"""
        try:
            # Activity started
            self.activity_started.labels(
                component=component,
                activity_type=activity_type
            ).inc()
            
            # Activity completed
            self.activity_completed.labels(
                component=component,
                activity_type=activity_type,
                status=status
            ).inc()
            
            # Activity duration
            self.activity_duration.labels(
                component=component,
                activity_type=activity_type
            ).observe(duration)
            
            self.logger.info(f"Logged activity metrics for {component}")
            
        except Exception as e:
            self.logger.error(f"Failed to log activity metrics for {component}: {e}")
    
    def create_dashboard(self, dashboard_config: Dict[str, Any]) -> str:
        """Create Grafana dashboard"""
        try:
            if not self.grafana_client:
                self.logger.warning("Grafana client not available")
                return None
            
            # Create dashboard
            response = requests.post(
                f"{self.config['dashboard']}/api/dashboards/db",
                json=dashboard_config,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                dashboard_id = response.json().get('id')
                self.logger.info(f"Created Grafana dashboard: {dashboard_id}")
                return dashboard_id
            else:
                self.logger.error(f"Failed to create dashboard: {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to create dashboard: {e}")
            return None
    
    def get_dashboard(self, dashboard_id: str) -> Dict[str, Any]:
        """Get Grafana dashboard"""
        try:
            if not self.grafana_client:
                self.logger.warning("Grafana client not available")
                return None
            
            response = requests.get(
                f"{self.config['dashboard']}/api/dashboards/uid/{dashboard_id}",
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"Failed to get dashboard: {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get dashboard: {e}")
            return None
    
    def list_dashboards(self) -> List[Dict[str, Any]]:
        """List Grafana dashboards"""
        try:
            if not self.grafana_client:
                self.logger.warning("Grafana client not available")
                return []
            
            response = requests.get(
                f"{self.config['dashboard']}/api/search?type=dash-db",
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"Failed to list dashboards: {response.status_code}")
                return []
                
        except Exception as e:
            self.logger.error(f"Failed to list dashboards: {e}")
            return []
    
    def send_alert(self, alert_config: Dict[str, Any]):
        """Send alert notification"""
        try:
            # Get alert configuration
            alert_type = alert_config.get('type', 'email')
            message = alert_config.get('message', '')
            severity = alert_config.get('severity', 'info')
            
            if alert_type == 'email':
                self._send_email_alert(alert_config)
            elif alert_type == 'slack':
                self._send_slack_alert(alert_config)
            elif alert_type == 'webhook':
                self._send_webhook_alert(alert_config)
            else:
                self.logger.warning(f"Unknown alert type: {alert_type}")
            
            self.logger.info(f"Sent {alert_type} alert: {message}")
            
        except Exception as e:
            self.logger.error(f"Failed to send alert: {e}")
    
    def _send_email_alert(self, alert_config: Dict[str, Any]):
        """Send email alert"""
        # Placeholder for email alert implementation
        self.logger.info("Email alert sent")
    
    def _send_slack_alert(self, alert_config: Dict[str, Any]):
        """Send Slack alert"""
        # Placeholder for Slack alert implementation
        self.logger.info("Slack alert sent")
    
    def _send_webhook_alert(self, alert_config: Dict[str, Any]):
        """Send webhook alert"""
        # Placeholder for webhook alert implementation
        self.logger.info("Webhook alert sent")
    
    def get_metrics_summary(self, component: str = None) -> Dict[str, Any]:
        """Get metrics summary"""
        try:
            # This would typically query Prometheus for metrics
            # For now, return a placeholder summary
            summary = {
                'component': component or 'all',
                'total_requests': 0,
                'error_rate': 0.0,
                'avg_response_time': 0.0,
                'active_connections': 0,
                'timestamp': datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get metrics summary: {e}")
            return {'error': str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """Check analytics service health"""
        try:
            # Test Prometheus connection
            prometheus_status = "healthy" if self.prometheus_client else "unhealthy"
            
            # Test Grafana connection
            grafana_status = "healthy" if self.grafana_client else "unhealthy"
            
            return {
                'status': 'healthy' if prometheus_status == 'healthy' else 'unhealthy',
                'prometheus': prometheus_status,
                'grafana': grafana_status,
                'endpoint': self.config['endpoint'],
                'dashboard': self.config['dashboard'],
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

def main():
    """Main function for analytics service"""
    
    # Initialize analytics service
    analytics_service = JarvisAnalyticsService()
    
    # Health check
    health = analytics_service.health_check()
    print(f"Analytics Service Health: {health}")
    
    if health['status'] == 'healthy':
        print("✅ Analytics service is healthy")
        
        # Test metrics logging
        analytics_service.log_system_metrics("test", {
            'cpu_usage': 45.5,
            'memory_usage': 1024 * 1024 * 1024,  # 1GB
            'disk_usage': 50 * 1024 * 1024 * 1024,  # 50GB
            'network_in': 1000,
            'network_out': 2000
        })
        print("✅ Logged system metrics")
        
        # Test application metrics
        analytics_service.log_application_metrics(
            "test", "/api/test", "GET", "200", 0.5
        )
        print("✅ Logged application metrics")
        
        # Test business metrics
        analytics_service.log_business_metrics("test", {
            'experiments': [{'status': 'completed'}],
            'model_training': {'model_type': 'hockey-detection', 'duration': 3600},
            'pipeline_success_rate': 0.95,
            'user_engagement': ['login', 'upload', 'download']
        })
        print("✅ Logged business metrics")
        
        # Test workflow metrics
        analytics_service.log_workflow_metrics(
            "test", "hockey-detection", "completed", 7200
        )
        print("✅ Logged workflow metrics")
        
        # Test activity metrics
        analytics_service.log_activity_metrics(
            "test", "data-preparation", "completed", 1800
        )
        print("✅ Logged activity metrics")
        
        # Test metrics summary
        summary = analytics_service.get_metrics_summary("test")
        print(f"✅ Metrics summary: {summary}")
        
    else:
        print(f"❌ Analytics service is unhealthy: {health['error']}")

if __name__ == "__main__":
    main()
