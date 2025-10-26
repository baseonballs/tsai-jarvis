#!/bin/bash

# TSAI Jarvis Command Line Interface
# Production-ready shell script for managing TSAI Jarvis Hockey Analytics services
#
# Usage:
#   ./jarvis.sh start --dashboard  # Start dashboard and backend services
#   ./jarvis.sh start --all       # Start all services
#   ./jarvis.sh stop              # Stop all services
#   ./jarvis.sh status            # Check service status
#   ./jarvis.sh help              # Show help information

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Script configuration
readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$SCRIPT_DIR"

# Production configuration
PRODUCTION_MODE="${PRODUCTION_MODE:-false}"
LOG_LEVEL="${LOG_LEVEL:-info}"
MAX_RETRIES="${MAX_RETRIES:-3}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-30}"
SHUTDOWN_TIMEOUT="${SHUTDOWN_TIMEOUT:-15}"

# Service configuration (based on APPLICATION-PORTS-MAPPING.md)
readonly DASHBOARD_PORT="${DASHBOARD_PORT:-8000}"
readonly CORE_API_PORT="${CORE_API_PORT:-8001}"
readonly API_GATEWAY_PORT="${API_GATEWAY_PORT:-8002}"
readonly ENTERPRISE_INTEGRATION_PORT="${ENTERPRISE_INTEGRATION_PORT:-8007}"
readonly ADVANCED_VISUALIZATION_PORT="${ADVANCED_VISUALIZATION_PORT:-8008}"
readonly MACHINE_LEARNING_PORT="${MACHINE_LEARNING_PORT:-8009}"
readonly ENTERPRISE_SECURITY_PORT="${ENTERPRISE_SECURITY_PORT:-8010}"
readonly ADVANCED_ANALYTICS_PORT="${ADVANCED_ANALYTICS_PORT:-8011}"
readonly TOOLCHAIN_INTEGRATION_PORT="${TOOLCHAIN_INTEGRATION_PORT:-8013}"
readonly AUTOPILOT_INTEGRATION_PORT="${AUTOPILOT_INTEGRATION_PORT:-8014}"
readonly SPOTLIGHT_INTEGRATION_PORT="${SPOTLIGHT_INTEGRATION_PORT:-8015}"

# PID files for service management
readonly DASHBOARD_PID_FILE="/tmp/tsai-jarvis-dashboard.pid"
readonly CORE_API_PID_FILE="/tmp/tsai-jarvis-core-api.pid"
readonly API_GATEWAY_PID_FILE="/tmp/tsai-jarvis-api-gateway.pid"
readonly ENTERPRISE_INTEGRATION_PID_FILE="/tmp/tsai-jarvis-enterprise-integration.pid"
readonly ADVANCED_VISUALIZATION_PID_FILE="/tmp/tsai-jarvis-advanced-visualization.pid"
readonly MACHINE_LEARNING_PID_FILE="/tmp/tsai-jarvis-machine-learning.pid"
readonly ENTERPRISE_SECURITY_PID_FILE="/tmp/tsai-jarvis-enterprise-security.pid"
readonly ADVANCED_ANALYTICS_PID_FILE="/tmp/tsai-jarvis-advanced-analytics.pid"
readonly TOOLCHAIN_INTEGRATION_PID_FILE="/tmp/tsai-jarvis-toolchain-integration.pid"
readonly AUTOPILOT_INTEGRATION_PID_FILE="/tmp/tsai-jarvis-autopilot-integration.pid"
readonly SPOTLIGHT_INTEGRATION_PID_FILE="/tmp/tsai-jarvis-spotlight-integration.pid"

# Enhanced colors for output
readonly RED='\033[0;31m'
readonly BRIGHT_RED='\033[1;31m'
readonly GREEN='\033[0;32m'
readonly BRIGHT_GREEN='\033[1;32m'
readonly YELLOW='\033[1;33m'
readonly BRIGHT_YELLOW='\033[1;93m'
readonly BLUE='\033[0;34m'
readonly BRIGHT_BLUE='\033[1;34m'
readonly PURPLE='\033[0;35m'
readonly BRIGHT_PURPLE='\033[1;35m'
readonly CYAN='\033[0;36m'
readonly BRIGHT_CYAN='\033[1;36m'
readonly WHITE='\033[1;37m'
readonly GRAY='\033[0;37m'
readonly BOLD='\033[1m'
readonly DIM='\033[2m'
readonly NC='\033[0m' # No Color

# ASCII Art
show_banner() {
    echo -e "${BRIGHT_CYAN}"
    cat << 'EOF'
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║    ████████╗███████╗ █████╗ ██╗    ██╗ █████╗ ██████╗ ██╗   ║
    ║    ╚══██╔══╝██╔════╝██╔══██╗██║   ██╔╝██╔══██╗██╔══██╗██║   ║
    ║       ██║   █████╗  ███████║██║   ██║ ███████║██████╔╝██║   ║
    ║       ██║   ██╔══╝  ██╔══██║██║   ██║ ██╔══██║██╔══██╗██║   ║
    ║       ██║   ███████╗██║  ██║╚██████╔╝██║  ██║██║  ██║╚██████╗
    ║       ╚═╝   ╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝
    ║                                                              ║
    ║              🏒 AI-Powered Hockey Analytics Platform 🏒     ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

# Enhanced logging functions with ASCII art
log_info() {
    echo -e "${BRIGHT_BLUE}┌─ ${BOLD}INFO${NC} ${BRIGHT_BLUE}─────────────────────────────────────────────────────────┐${NC}"
    echo -e "${BRIGHT_BLUE}│${NC} ${BLUE}ℹ️  $1${NC}"
    echo -e "${BRIGHT_BLUE}└─────────────────────────────────────────────────────────────────┘${NC}"
}

log_success() {
    echo -e "${BRIGHT_GREEN}┌─ ${BOLD}SUCCESS${NC} ${BRIGHT_GREEN}─────────────────────────────────────────────────────┐${NC}"
    echo -e "${BRIGHT_GREEN}│${NC} ${GREEN}✅ $1${NC}"
    echo -e "${BRIGHT_GREEN}└─────────────────────────────────────────────────────────────────┘${NC}"
}

log_warning() {
    echo -e "${BRIGHT_YELLOW}┌─ ${BOLD}WARNING${NC} ${BRIGHT_YELLOW}─────────────────────────────────────────────────────┐${NC}"
    echo -e "${BRIGHT_YELLOW}│${NC} ${YELLOW}⚠️  $1${NC}"
    echo -e "${BRIGHT_YELLOW}└─────────────────────────────────────────────────────────────────┘${NC}"
}

log_error() {
    echo -e "${BRIGHT_RED}┌─ ${BOLD}ERROR${NC} ${BRIGHT_RED}─────────────────────────────────────────────────────────┐${NC}"
    echo -e "${BRIGHT_RED}│${NC} ${RED}❌ $1${NC}"
    echo -e "${BRIGHT_RED}└─────────────────────────────────────────────────────────────────┘${NC}" >&2
}

log_header() {
    echo -e "${BRIGHT_PURPLE}┌─ ${BOLD}$1${NC} ${BRIGHT_PURPLE}─────────────────────────────────────────────────────┐${NC}"
    echo -e "${BRIGHT_PURPLE}│${NC} ${PURPLE}🚀 $1${NC}"
    echo -e "${BRIGHT_PURPLE}└─────────────────────────────────────────────────────────────────┘${NC}"
}

log_step() {
    echo -e "${BRIGHT_CYAN}┌─ ${BOLD}STEP${NC} ${BRIGHT_CYAN}─────────────────────────────────────────────────────────┐${NC}"
    echo -e "${BRIGHT_CYAN}│${NC} ${CYAN}🔧 $1${NC}"
    echo -e "${BRIGHT_CYAN}└─────────────────────────────────────────────────────────────────┘${NC}"
}

log_service() {
    echo -e "${BRIGHT_GREEN}┌─ ${BOLD}SERVICE${NC} ${BRIGHT_GREEN}─────────────────────────────────────────────────────┐${NC}"
    echo -e "${BRIGHT_GREEN}│${NC} ${GREEN}⚙️  $1${NC}"
    echo -e "${BRIGHT_GREEN}└─────────────────────────────────────────────────────────────────┘${NC}"
}

log_url() {
    echo -e "${BRIGHT_CYAN}┌─ ${BOLD}URL${NC} ${BRIGHT_CYAN}─────────────────────────────────────────────────────────┐${NC}"
    echo -e "${BRIGHT_CYAN}│${NC} ${CYAN}🌐 $1${NC}"
    echo -e "${BRIGHT_CYAN}└─────────────────────────────────────────────────────────────────┘${NC}"
}

log_pid() {
    echo -e "${BRIGHT_PURPLE}┌─ ${BOLD}PROCESS${NC} ${BRIGHT_PURPLE}─────────────────────────────────────────────────────┐${NC}"
    echo -e "${BRIGHT_PURPLE}│${NC} ${PURPLE}🆔 $1${NC}"
    echo -e "${BRIGHT_PURPLE}└─────────────────────────────────────────────────────────────────┘${NC}"
}

# Error handling
error_exit() {
    log_error "$1"
    exit 1
}

# Check if running from correct directory
check_project_root() {
    log_step "Validating project structure..."
    
    # Check for essential project files
    local required_files=(
        "requirements.txt"
        "dashboard/package.json"
    )
    
    # Check for backend files
    local backend_files=(
        "hockey-analytics/enterprise_integration_api.py"
        "hockey-analytics/advanced_visualization_api.py"
        "hockey-analytics/machine_learning_api.py"
        "hockey-analytics/enterprise_security_api.py"
        "hockey-analytics/advanced_analytics_api.py"
    )
    
    local required_dirs=(
        "dashboard"
        "hockey-analytics"
    )
    
    # Check required files
    for file in "${required_files[@]}"; do
        if [[ ! -f "$PROJECT_ROOT/$file" ]]; then
            log_error "Required file not found: $file"
            log_info "Make sure you're running this script from the TSAI Jarvis project root"
            return 1
        fi
    done
    
    # Check backend files
    for file in "${backend_files[@]}"; do
        if [[ ! -f "$PROJECT_ROOT/$file" ]]; then
            log_error "Required backend file not found: $file"
            log_info "Make sure you're running this script from the TSAI Jarvis project root"
            return 1
        fi
    done
    
    # Check required directories
    for dir in "${required_dirs[@]}"; do
        if [[ ! -d "$PROJECT_ROOT/$dir" ]]; then
            log_error "Required directory not found: $dir"
            log_info "Make sure you're running this script from the TSAI Jarvis project root"
            return 1
        fi
    done
    
    log_success "Project structure validated"
    return 0
}

# Check if required tools are available
check_dependencies() {
    log_step "Checking system dependencies..."
    
    local missing_deps=()
    local optional_deps=()
    
    # Required dependencies
    if ! command -v python3 &> /dev/null; then
        missing_deps+=("python3")
    else
        local python_version
        python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
        log_info "Python version: $python_version"
    fi
    
    if ! command -v node &> /dev/null; then
        missing_deps+=("node")
    else
        local node_version
        node_version=$(node --version 2>&1)
        log_info "Node.js version: $node_version"
    fi
    
    if ! command -v npm &> /dev/null; then
        missing_deps+=("npm")
    else
        local npm_version
        npm_version=$(npm --version 2>&1)
        log_info "NPM version: $npm_version"
    fi
    
    # Optional dependencies
    if ! command -v uvicorn &> /dev/null; then
        optional_deps+=("uvicorn")
    else
        local uvicorn_version
        uvicorn_version=$(uvicorn --version 2>&1 | cut -d' ' -f2)
        log_info "Uvicorn version: $uvicorn_version"
    fi
    
    # Check system resources
    check_system_resources
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_error "Missing required dependencies: ${missing_deps[*]}"
        log_info "Please install them first:"
        for dep in "${missing_deps[@]}"; do
            case "$dep" in
                "python3")
                    log_info "  - Python 3: https://python.org/downloads/"
                    ;;
                "node")
                    log_info "  - Node.js: https://nodejs.org/downloads/ or use nvm"
                    ;;
                "npm")
                    log_info "  - NPM: comes with Node.js"
                    ;;
            esac
        done
        return 1
    fi
    
    if [[ ${#optional_deps[@]} -gt 0 ]]; then
        log_warning "Optional dependencies not found: ${optional_deps[*]}"
        log_info "Some features may not work without uvicorn"
    fi
    
    log_success "Dependencies check completed"
    return 0
}

# Check system resources
check_system_resources() {
    log_step "Checking system resources..."
    
    # Check available memory
    local total_mem
    local available_mem
    if command -v free &> /dev/null; then
        total_mem=$(free -m | awk 'NR==2{printf "%.0f", $2}')
        available_mem=$(free -m | awk 'NR==2{printf "%.0f", $7}')
        log_info "Total memory: ${total_mem}MB, Available: ${available_mem}MB"
        
        if [[ $available_mem -lt 2048 ]]; then
            log_warning "Low available memory (${available_mem}MB). Consider closing other applications."
        fi
    fi
    
    # Check disk space
    local disk_usage
    disk_usage=$(df -h "$PROJECT_ROOT" | awk 'NR==2{print $5}' | sed 's/%//')
    log_info "Disk usage: ${disk_usage}%"
    
    if [[ $disk_usage -gt 90 ]]; then
        log_warning "High disk usage (${disk_usage}%). Consider freeing up space."
    fi
    
    # Check CPU cores
    local cpu_cores
    cpu_cores=$(nproc 2>/dev/null || echo "unknown")
    log_info "CPU cores: $cpu_cores"
    
    log_success "System resources check completed"
}

# Check if a port is in use
is_port_in_use() {
    local port="$1"
    if lsof -Pi ":$port" -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Kill process by PID file
kill_by_pid_file() {
    local pid_file="$1"
    local service_name="$2"
    
    if [[ -f "$pid_file" ]]; then
        local pid
        pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            log_service "Stopping $service_name (PID: $pid)..."
            kill "$pid" 2>/dev/null || true
            sleep 2
            if kill -0 "$pid" 2>/dev/null; then
                log_warning "Force killing $service_name..."
                kill -9 "$pid" 2>/dev/null || true
            fi
            log_success "$service_name stopped successfully"
        else
            log_warning "$service_name was not running"
        fi
        rm -f "$pid_file"
    fi
}

# Enhanced function to check and clean up conflicting processes
check_and_cleanup_conflicts() {
    local service_name="$1"
    local port="$2"
    
    log_step "Checking for conflicts with $service_name on port $port..."
    
    # Check if port is in use
    if is_port_in_use "$port"; then
        log_warning "Port $port is already in use by another process"
        
        # Get process info
        local pid
        local process_info
        pid=$(lsof -Pi ":$port" -sTCP:LISTEN -t 2>/dev/null | head -1)
        
        if [[ -n "$pid" ]]; then
            process_info=$(ps -p "$pid" -o comm= 2>/dev/null || echo "unknown")
            log_warning "Process using port $port: PID $pid ($process_info)"
            
            # Check if it's our own process (from PID file)
            local pid_file=""
            case "$port" in
                "$DASHBOARD_PORT") pid_file="$DASHBOARD_PID_FILE" ;;
                "$CORE_API_PORT") pid_file="$CORE_API_PID_FILE" ;;
                "$API_GATEWAY_PORT") pid_file="$API_GATEWAY_PID_FILE" ;;
                "$ENTERPRISE_INTEGRATION_PORT") pid_file="$ENTERPRISE_INTEGRATION_PID_FILE" ;;
                "$ADVANCED_VISUALIZATION_PORT") pid_file="$ADVANCED_VISUALIZATION_PID_FILE" ;;
                "$MACHINE_LEARNING_PORT") pid_file="$MACHINE_LEARNING_PID_FILE" ;;
                "$ENTERPRISE_SECURITY_PORT") pid_file="$ENTERPRISE_SECURITY_PID_FILE" ;;
                "$ADVANCED_ANALYTICS_PORT") pid_file="$ADVANCED_ANALYTICS_PID_FILE" ;;
            esac
            
            if [[ -f "$pid_file" ]]; then
                local tracked_pid
                tracked_pid=$(cat "$pid_file" 2>/dev/null || echo "")
                if [[ "$pid" == "$tracked_pid" ]]; then
                    log_info "Port $port is used by our tracked $service_name process (PID: $pid)"
                    return 0  # This is our process, no conflict
                fi
            fi
            
            # For automated scenarios, we'll kill the conflicting process
            log_step "Automatically killing conflicting process on port $port..."
            kill_process_robust "$pid" "Conflicting process on port $port"
            
            # Wait a moment for port to be freed
            sleep 2
            
            # Verify port is now free
            if is_port_in_use "$port"; then
                log_error "Failed to free port $port. Please resolve manually."
                return 1
            else
                log_success "Port $port is now free"
                return 0
            fi
        else
            log_error "Could not identify process using port $port"
            return 1
        fi
    else
        log_success "Port $port is available for $service_name"
        return 0
    fi
}

# Enhanced function to verify service startup
verify_service_startup() {
    local service_name="$1"
    local port="$2"
    local pid_file="$3"
    local max_attempts=15
    local attempt=1
    
    log_step "Verifying $service_name startup..."
    
    while [[ $attempt -le $max_attempts ]]; do
        # Check if PID file exists and process is running
        if [[ -f "$pid_file" ]]; then
            local pid
            pid=$(cat "$pid_file" 2>/dev/null || echo "")
            if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
                # Check if port is responding
                if is_port_in_use "$port"; then
                    # Additional health check for HTTP services
                    if check_service_health "$port"; then
                        log_success "$service_name is running and healthy (PID: $pid, Port: $port)"
                        return 0
                    else
                        log_info "$service_name is running but not yet healthy... (attempt $attempt/$max_attempts)"
                    fi
                fi
            fi
        fi
        
        # Check if port is responding (even without PID file)
        if is_port_in_use "$port"; then
            # Additional health check for HTTP services
            if check_service_health "$port"; then
                log_success "$service_name is responding and healthy on port $port"
                return 0
            else
                log_info "$service_name is responding but not yet healthy... (attempt $attempt/$max_attempts)"
            fi
        fi
        
        log_info "Waiting for $service_name to start... (attempt $attempt/$max_attempts)"
        sleep 2
        ((attempt++))
    done
    
    log_error "$service_name failed to start after $max_attempts attempts"
    return 1
}

# Check if a service is healthy by making an HTTP request
check_service_health() {
    local port="$1"
    local max_attempts=3
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        # Try to connect to the service
        if curl -s --connect-timeout 5 --max-time 10 "http://localhost:$port" >/dev/null 2>&1; then
            return 0
        fi
        
        # For specific services, try health endpoints
        case "$port" in
            "8001"|"8007"|"8008"|"8009"|"8010"|"8011")
                if curl -s --connect-timeout 5 --max-time 10 "http://localhost:$port/health" >/dev/null 2>&1; then
                    return 0
                fi
                ;;
        esac
        
        sleep 1
        ((attempt++))
    done
    
    return 1
}

# Enhanced function to start service with comprehensive checks
start_service_robust() {
    local service_name="$1"
    local port="$2"
    local start_function="$3"
    local background_mode="$4"
    
    log_step "Starting $service_name..."
    
    # Check and clean up conflicts
    if ! check_and_cleanup_conflicts "$service_name" "$port"; then
        log_error "Failed to resolve conflicts for $service_name"
        return 1
    fi
    
    # Start the service
    if ! "$start_function" "$background_mode"; then
        log_error "Failed to start $service_name"
        return 1
    fi
    
    # Verify startup
    local pid_file=""
    case "$port" in
        "$DASHBOARD_PORT") pid_file="$DASHBOARD_PID_FILE" ;;
        "$CORE_API_PORT") pid_file="$CORE_API_PID_FILE" ;;
        "$API_GATEWAY_PORT") pid_file="$API_GATEWAY_PID_FILE" ;;
        "$ENTERPRISE_INTEGRATION_PORT") pid_file="$ENTERPRISE_INTEGRATION_PID_FILE" ;;
        "$ADVANCED_VISUALIZATION_PORT") pid_file="$ADVANCED_VISUALIZATION_PID_FILE" ;;
        "$MACHINE_LEARNING_PORT") pid_file="$MACHINE_LEARNING_PID_FILE" ;;
        "$ENTERPRISE_SECURITY_PORT") pid_file="$ENTERPRISE_SECURITY_PID_FILE" ;;
        "$ADVANCED_ANALYTICS_PORT") pid_file="$ADVANCED_ANALYTICS_PID_FILE" ;;
    esac
    
    if ! verify_service_startup "$service_name" "$port" "$pid_file"; then
        log_error "$service_name startup verification failed"
        return 1
    fi
    
    log_success "$service_name started successfully"
    return 0
}

# Start Dashboard
start_dashboard() {
    local background_mode="${1:-false}"
    
    log_step "Starting TSAI Jarvis Dashboard on port $DASHBOARD_PORT..."
    
    # Check if port is already in use
    if is_port_in_use "$DASHBOARD_PORT"; then
        log_warning "Dashboard port $DASHBOARD_PORT is already in use"
        return 1
    fi
    
    # Check if dashboard directory exists
    if [[ ! -d "$PROJECT_ROOT/dashboard" ]]; then
        log_error "Dashboard directory not found: $PROJECT_ROOT/dashboard"
        return 1
    fi
    
    # Navigate to dashboard directory
    cd "$PROJECT_ROOT/dashboard"
    
    # Check if node_modules exists
    if [[ ! -d "node_modules" ]]; then
        log_step "Installing dashboard dependencies..."
        if ! npm install >/dev/null 2>&1; then
            log_error "Failed to install dashboard dependencies"
            cd "$PROJECT_ROOT"
            return 1
        fi
        log_success "Dashboard dependencies installed"
    fi
    
    # Start dashboard
    log_step "Starting Next.js dashboard..."
    if [[ "$background_mode" == true ]]; then
        npm run dev >/dev/null 2>&1 &
        local dashboard_pid=$!
    else
        npm run dev &
        local dashboard_pid=$!
    fi
    
    # Store PID for tracking
    echo "$dashboard_pid" > "$DASHBOARD_PID_FILE"
    
    # Wait for dashboard to start
    local dashboard_attempts=0
    local max_dashboard_attempts=15
    while [[ $dashboard_attempts -lt $max_dashboard_attempts ]]; do
        if ! kill -0 "$dashboard_pid" 2>/dev/null; then
            log_error "Dashboard service failed to start"
            return 1
        fi
        
        if is_port_in_use "$DASHBOARD_PORT"; then
            log_success "Dashboard service started (PID: $dashboard_pid)"
            break
        fi
        
        sleep 2
        ((dashboard_attempts++))
    done
    
    if [[ $dashboard_attempts -eq $max_dashboard_attempts ]]; then
        log_error "Dashboard service failed to bind to port $DASHBOARD_PORT"
        kill "$dashboard_pid" 2>/dev/null || true
        return 1
    fi
    
    log_success "Dashboard started successfully"
    log_pid "Process ID: $dashboard_pid"
    log_url "URL: http://localhost:$DASHBOARD_PORT"
    
    # Return to project root
    cd "$PROJECT_ROOT"
}

# Start Core API
start_core_api() {
    local background_mode="${1:-false}"
    
    log_step "Starting Core API on port $CORE_API_PORT..."
    
    # Check if port is already in use
    if is_port_in_use "$CORE_API_PORT"; then
        log_warning "Core API port $CORE_API_PORT is already in use"
        return 1
    fi
    
    # Check if main_api.py exists
    if [[ ! -f "$PROJECT_ROOT/hockey-analytics/main_api.py" ]]; then
        log_error "Core API file not found: hockey-analytics/main_api.py"
        return 1
    fi
    
    # Ensure we're in the project root
    cd "$PROJECT_ROOT"
    
    # Check if virtual environment exists
    if [[ ! -d "venv" ]]; then
        log_step "Creating Python virtual environment..."
        if ! python3 -m venv venv; then
            log_error "Failed to create Python virtual environment"
            return 1
        fi
        log_success "Python virtual environment created"
    fi
    
    # Activate virtual environment
    log_step "Activating Python virtual environment..."
    if ! source venv/bin/activate; then
        log_error "Failed to activate Python virtual environment"
        return 1
    fi
    log_success "Python virtual environment activated"
    
    # Install Python dependencies
    if [[ -f "requirements.txt" ]]; then
        log_step "Installing Python dependencies..."
        if ! pip install -r requirements.txt >/dev/null 2>&1; then
            log_error "Failed to install Python dependencies"
            return 1
        fi
        log_success "Python dependencies installed"
    fi
    
    # Start Core API
    log_step "Starting Core API service..."
    if [[ "$background_mode" == true ]]; then
        uvicorn hockey-analytics.main_api:app --host 0.0.0.0 --port "$CORE_API_PORT" --reload >/dev/null 2>&1 &
        local core_api_pid=$!
    else
        uvicorn hockey-analytics.main_api:app --host 0.0.0.0 --port "$CORE_API_PORT" --reload &
        local core_api_pid=$!
    fi
    
    # Store PID for tracking
    echo "$core_api_pid" > "$CORE_API_PID_FILE"
    
    # Wait for Core API to start
    local core_api_attempts=0
    local max_core_api_attempts=15
    while [[ $core_api_attempts -lt $max_core_api_attempts ]]; do
        if ! kill -0 "$core_api_pid" 2>/dev/null; then
            log_error "Core API service failed to start"
            return 1
        fi
        
        if is_port_in_use "$CORE_API_PORT"; then
            log_success "Core API service started (PID: $core_api_pid)"
            break
        fi
        
        sleep 2
        ((core_api_attempts++))
    done
    
    if [[ $core_api_attempts -eq $max_core_api_attempts ]]; then
        log_error "Core API service failed to bind to port $CORE_API_PORT"
        kill "$core_api_pid" 2>/dev/null || true
        return 1
    fi
    
    log_success "Core API started successfully"
    log_pid "Process ID: $core_api_pid"
    log_url "API: http://localhost:$CORE_API_PORT"
    
    return 0
}

# Start Enterprise Integration API
start_enterprise_integration() {
    local background_mode="${1:-false}"
    
    log_step "Starting Enterprise Integration API on port $ENTERPRISE_INTEGRATION_PORT..."
    
    # Check if port is already in use
    if is_port_in_use "$ENTERPRISE_INTEGRATION_PORT"; then
        log_warning "Enterprise Integration port $ENTERPRISE_INTEGRATION_PORT is already in use"
        return 1
    fi
    
    # Check if API file exists
    if [[ ! -f "$PROJECT_ROOT/hockey-analytics/enterprise_integration_api.py" ]]; then
        log_error "Enterprise Integration API file not found"
        return 1
    fi
    
    # Ensure we're in the project root
    cd "$PROJECT_ROOT"
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Start Enterprise Integration API
    log_step "Starting Enterprise Integration API service..."
    if [[ "$background_mode" == true ]]; then
        uvicorn hockey-analytics.enterprise_integration_api:app --host 0.0.0.0 --port "$ENTERPRISE_INTEGRATION_PORT" --reload >/dev/null 2>&1 &
        local enterprise_pid=$!
    else
        uvicorn hockey-analytics.enterprise_integration_api:app --host 0.0.0.0 --port "$ENTERPRISE_INTEGRATION_PORT" --reload &
        local enterprise_pid=$!
    fi
    
    # Store PID for tracking
    echo "$enterprise_pid" > "$ENTERPRISE_INTEGRATION_PID_FILE"
    
    # Wait for Enterprise Integration API to start
    local enterprise_attempts=0
    local max_enterprise_attempts=15
    while [[ $enterprise_attempts -lt $max_enterprise_attempts ]]; do
        if ! kill -0 "$enterprise_pid" 2>/dev/null; then
            log_error "Enterprise Integration API service failed to start"
            return 1
        fi
        
        if is_port_in_use "$ENTERPRISE_INTEGRATION_PORT"; then
            log_success "Enterprise Integration API service started (PID: $enterprise_pid)"
            break
        fi
        
        sleep 2
        ((enterprise_attempts++))
    done
    
    if [[ $enterprise_attempts -eq $max_enterprise_attempts ]]; then
        log_error "Enterprise Integration API service failed to bind to port $ENTERPRISE_INTEGRATION_PORT"
        kill "$enterprise_pid" 2>/dev/null || true
        return 1
    fi
    
    log_success "Enterprise Integration API started successfully"
    log_pid "Process ID: $enterprise_pid"
    log_url "API: http://localhost:$ENTERPRISE_INTEGRATION_PORT"
    
    return 0
}

# Start Advanced Visualization API
start_advanced_visualization() {
    local background_mode="${1:-false}"
    
    log_step "Starting Advanced Visualization API on port $ADVANCED_VISUALIZATION_PORT..."
    
    # Check if port is already in use
    if is_port_in_use "$ADVANCED_VISUALIZATION_PORT"; then
        log_warning "Advanced Visualization port $ADVANCED_VISUALIZATION_PORT is already in use"
        return 1
    fi
    
    # Check if API file exists
    if [[ ! -f "$PROJECT_ROOT/hockey-analytics/advanced_visualization_api.py" ]]; then
        log_error "Advanced Visualization API file not found"
        return 1
    fi
    
    # Ensure we're in the project root
    cd "$PROJECT_ROOT"
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Start Advanced Visualization API
    log_step "Starting Advanced Visualization API service..."
    if [[ "$background_mode" == true ]]; then
        uvicorn hockey-analytics.advanced_visualization_api:app --host 0.0.0.0 --port "$ADVANCED_VISUALIZATION_PORT" --reload >/dev/null 2>&1 &
        local visualization_pid=$!
    else
        uvicorn hockey-analytics.advanced_visualization_api:app --host 0.0.0.0 --port "$ADVANCED_VISUALIZATION_PORT" --reload &
        local visualization_pid=$!
    fi
    
    # Store PID for tracking
    echo "$visualization_pid" > "$ADVANCED_VISUALIZATION_PID_FILE"
    
    # Wait for Advanced Visualization API to start
    local visualization_attempts=0
    local max_visualization_attempts=15
    while [[ $visualization_attempts -lt $max_visualization_attempts ]]; do
        if ! kill -0 "$visualization_pid" 2>/dev/null; then
            log_error "Advanced Visualization API service failed to start"
            return 1
        fi
        
        if is_port_in_use "$ADVANCED_VISUALIZATION_PORT"; then
            log_success "Advanced Visualization API service started (PID: $visualization_pid)"
            break
        fi
        
        sleep 2
        ((visualization_attempts++))
    done
    
    if [[ $visualization_attempts -eq $max_visualization_attempts ]]; then
        log_error "Advanced Visualization API service failed to bind to port $ADVANCED_VISUALIZATION_PORT"
        kill "$visualization_pid" 2>/dev/null || true
        return 1
    fi
    
    log_success "Advanced Visualization API started successfully"
    log_pid "Process ID: $visualization_pid"
    log_url "API: http://localhost:$ADVANCED_VISUALIZATION_PORT"
    
    return 0
}

# Start Machine Learning API
start_machine_learning() {
    local background_mode="${1:-false}"
    
    log_step "Starting Machine Learning API on port $MACHINE_LEARNING_PORT..."
    
    # Check if port is already in use
    if is_port_in_use "$MACHINE_LEARNING_PORT"; then
        log_warning "Machine Learning port $MACHINE_LEARNING_PORT is already in use"
        return 1
    fi
    
    # Check if API file exists
    if [[ ! -f "$PROJECT_ROOT/hockey-analytics/machine_learning_api.py" ]]; then
        log_error "Machine Learning API file not found"
        return 1
    fi
    
    # Ensure we're in the project root
    cd "$PROJECT_ROOT"
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Start Machine Learning API
    log_step "Starting Machine Learning API service..."
    if [[ "$background_mode" == true ]]; then
        uvicorn hockey-analytics.machine_learning_api:app --host 0.0.0.0 --port "$MACHINE_LEARNING_PORT" --reload >/dev/null 2>&1 &
        local ml_pid=$!
    else
        uvicorn hockey-analytics.machine_learning_api:app --host 0.0.0.0 --port "$MACHINE_LEARNING_PORT" --reload &
        local ml_pid=$!
    fi
    
    # Store PID for tracking
    echo "$ml_pid" > "$MACHINE_LEARNING_PID_FILE"
    
    # Wait for Machine Learning API to start
    local ml_attempts=0
    local max_ml_attempts=15
    while [[ $ml_attempts -lt $max_ml_attempts ]]; do
        if ! kill -0 "$ml_pid" 2>/dev/null; then
            log_error "Machine Learning API service failed to start"
            return 1
        fi
        
        if is_port_in_use "$MACHINE_LEARNING_PORT"; then
            log_success "Machine Learning API service started (PID: $ml_pid)"
            break
        fi
        
        sleep 2
        ((ml_attempts++))
    done
    
    if [[ $ml_attempts -eq $max_ml_attempts ]]; then
        log_error "Machine Learning API service failed to bind to port $MACHINE_LEARNING_PORT"
        kill "$ml_pid" 2>/dev/null || true
        return 1
    fi
    
    log_success "Machine Learning API started successfully"
    log_pid "Process ID: $ml_pid"
    log_url "API: http://localhost:$MACHINE_LEARNING_PORT"
    
    return 0
}

# Start Enterprise Security API
start_enterprise_security() {
    local background_mode="${1:-false}"
    
    log_step "Starting Enterprise Security API on port $ENTERPRISE_SECURITY_PORT..."
    
    # Check if port is already in use
    if is_port_in_use "$ENTERPRISE_SECURITY_PORT"; then
        log_warning "Enterprise Security port $ENTERPRISE_SECURITY_PORT is already in use"
        return 1
    fi
    
    # Check if API file exists
    if [[ ! -f "$PROJECT_ROOT/hockey-analytics/enterprise_security_api.py" ]]; then
        log_error "Enterprise Security API file not found"
        return 1
    fi
    
    # Ensure we're in the project root
    cd "$PROJECT_ROOT"
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Start Enterprise Security API
    log_step "Starting Enterprise Security API service..."
    if [[ "$background_mode" == true ]]; then
        uvicorn hockey-analytics.enterprise_security_api:app --host 0.0.0.0 --port "$ENTERPRISE_SECURITY_PORT" --reload >/dev/null 2>&1 &
        local security_pid=$!
    else
        uvicorn hockey-analytics.enterprise_security_api:app --host 0.0.0.0 --port "$ENTERPRISE_SECURITY_PORT" --reload &
        local security_pid=$!
    fi
    
    # Store PID for tracking
    echo "$security_pid" > "$ENTERPRISE_SECURITY_PID_FILE"
    
    # Wait for Enterprise Security API to start
    local security_attempts=0
    local max_security_attempts=15
    while [[ $security_attempts -lt $max_security_attempts ]]; do
        if ! kill -0 "$security_pid" 2>/dev/null; then
            log_error "Enterprise Security API service failed to start"
            return 1
        fi
        
        if is_port_in_use "$ENTERPRISE_SECURITY_PORT"; then
            log_success "Enterprise Security API service started (PID: $security_pid)"
            break
        fi
        
        sleep 2
        ((security_attempts++))
    done
    
    if [[ $security_attempts -eq $max_security_attempts ]]; then
        log_error "Enterprise Security API service failed to bind to port $ENTERPRISE_SECURITY_PORT"
        kill "$security_pid" 2>/dev/null || true
        return 1
    fi
    
    log_success "Enterprise Security API started successfully"
    log_pid "Process ID: $security_pid"
    log_url "API: http://localhost:$ENTERPRISE_SECURITY_PORT"
    
    return 0
}

# Start Advanced Analytics API
start_advanced_analytics() {
    local background_mode="${1:-false}"
    
    log_step "Starting Advanced Analytics API on port $ADVANCED_ANALYTICS_PORT..."
    
    # Check if port is already in use
    if is_port_in_use "$ADVANCED_ANALYTICS_PORT"; then
        log_warning "Advanced Analytics port $ADVANCED_ANALYTICS_PORT is already in use"
        return 1
    fi
    
    # Check if API file exists
    if [[ ! -f "$PROJECT_ROOT/hockey-analytics/advanced_analytics_api.py" ]]; then
        log_error "Advanced Analytics API file not found"
        return 1
    fi
    
    # Ensure we're in the project root
    cd "$PROJECT_ROOT"
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Start Advanced Analytics API
    log_step "Starting Advanced Analytics API service..."
    if [[ "$background_mode" == true ]]; then
        uvicorn hockey-analytics.advanced_analytics_api:app --host 0.0.0.0 --port "$ADVANCED_ANALYTICS_PORT" --reload >/dev/null 2>&1 &
        local analytics_pid=$!
    else
        uvicorn hockey-analytics.advanced_analytics_api:app --host 0.0.0.0 --port "$ADVANCED_ANALYTICS_PORT" --reload &
        local analytics_pid=$!
    fi
    
    # Store PID for tracking
    echo "$analytics_pid" > "$ADVANCED_ANALYTICS_PID_FILE"
    
    # Wait for Advanced Analytics API to start
    local analytics_attempts=0
    local max_analytics_attempts=15
    while [[ $analytics_attempts -lt $max_analytics_attempts ]]; do
        if ! kill -0 "$analytics_pid" 2>/dev/null; then
            log_error "Advanced Analytics API service failed to start"
            return 1
        fi
        
        if is_port_in_use "$ADVANCED_ANALYTICS_PORT"; then
            log_success "Advanced Analytics API service started (PID: $analytics_pid)"
            break
        fi
        
        sleep 2
        ((analytics_attempts++))
    done
    
    if [[ $analytics_attempts -eq $max_analytics_attempts ]]; then
        log_error "Advanced Analytics API service failed to bind to port $ADVANCED_ANALYTICS_PORT"
        kill "$analytics_pid" 2>/dev/null || true
        return 1
    fi
    
    log_success "Advanced Analytics API started successfully"
    log_pid "Process ID: $analytics_pid"
    log_url "API: http://localhost:$ADVANCED_ANALYTICS_PORT"
    
    return 0
}

# TSAI Ecosystem Integration Services
start_toolchain_integration() {
    local background_mode="${1:-false}"
    
    log_step "Starting Toolchain Integration API on port $TOOLCHAIN_INTEGRATION_PORT..."
    
    # Check if port is already in use
    if is_port_in_use "$TOOLCHAIN_INTEGRATION_PORT"; then
        log_warning "Toolchain Integration port $TOOLCHAIN_INTEGRATION_PORT is already in use"
        return 1
    fi
    
    # Check if API file exists
    if [[ ! -f "$PROJECT_ROOT/hockey-analytics/toolchain_integration_api.py" ]]; then
        log_error "Toolchain Integration API file not found"
        return 1
    fi
    
    # Ensure we're in the project root
    cd "$PROJECT_ROOT"
    
    # Activate virtual environment
    if [[ -f "venv/bin/activate" ]]; then
        source venv/bin/activate
    else
        log_error "Virtual environment not found"
        return 1
    fi
    
    # Start the service
    if [[ "$background_mode" == "true" ]]; then
        nohup python hockey-analytics/toolchain_integration_api.py > /tmp/tsai-jarvis-toolchain-integration.log 2>&1 &
        local toolchain_integration_pid=$!
        echo "$toolchain_integration_pid" > "$TOOLCHAIN_INTEGRATION_PID_FILE"
        log_success "Toolchain Integration API service started in background (PID: $toolchain_integration_pid)"
    else
        python hockey-analytics/toolchain_integration_api.py &
        local toolchain_integration_pid=$!
        echo "$toolchain_integration_pid" > "$TOOLCHAIN_INTEGRATION_PID_FILE"
        log_success "Toolchain Integration API service started (PID: $toolchain_integration_pid)"
    fi
    
    return 0
}

start_autopilot_integration() {
    local background_mode="${1:-false}"
    
    log_step "Starting Autopilot Integration API on port $AUTOPILOT_INTEGRATION_PORT..."
    
    # Check if port is already in use
    if is_port_in_use "$AUTOPILOT_INTEGRATION_PORT"; then
        log_warning "Autopilot Integration port $AUTOPILOT_INTEGRATION_PORT is already in use"
        return 1
    fi
    
    # Check if API file exists
    if [[ ! -f "$PROJECT_ROOT/hockey-analytics/autopilot_integration_api.py" ]]; then
        log_error "Autopilot Integration API file not found"
        return 1
    fi
    
    # Ensure we're in the project root
    cd "$PROJECT_ROOT"
    
    # Activate virtual environment
    if [[ -f "venv/bin/activate" ]]; then
        source venv/bin/activate
    else
        log_error "Virtual environment not found"
        return 1
    fi
    
    # Start the service
    if [[ "$background_mode" == "true" ]]; then
        nohup python hockey-analytics/autopilot_integration_api.py > /tmp/tsai-jarvis-autopilot-integration.log 2>&1 &
        local autopilot_integration_pid=$!
        echo "$autopilot_integration_pid" > "$AUTOPILOT_INTEGRATION_PID_FILE"
        log_success "Autopilot Integration API service started in background (PID: $autopilot_integration_pid)"
    else
        python hockey-analytics/autopilot_integration_api.py &
        local autopilot_integration_pid=$!
        echo "$autopilot_integration_pid" > "$AUTOPILOT_INTEGRATION_PID_FILE"
        log_success "Autopilot Integration API service started (PID: $autopilot_integration_pid)"
    fi
    
    return 0
}

start_spotlight_integration() {
    local background_mode="${1:-false}"
    
    log_step "Starting Spotlight Integration API on port $SPOTLIGHT_INTEGRATION_PORT..."
    
    # Check if port is already in use
    if is_port_in_use "$SPOTLIGHT_INTEGRATION_PORT"; then
        log_warning "Spotlight Integration port $SPOTLIGHT_INTEGRATION_PORT is already in use"
        return 1
    fi
    
    # Check if API file exists
    if [[ ! -f "$PROJECT_ROOT/hockey-analytics/spotlight_integration_api.py" ]]; then
        log_error "Spotlight Integration API file not found"
        return 1
    fi
    
    # Ensure we're in the project root
    cd "$PROJECT_ROOT"
    
    # Activate virtual environment
    if [[ -f "venv/bin/activate" ]]; then
        source venv/bin/activate
    else
        log_error "Virtual environment not found"
        return 1
    fi
    
    # Start the service
    if [[ "$background_mode" == "true" ]]; then
        nohup python hockey-analytics/spotlight_integration_api.py > /tmp/tsai-jarvis-spotlight-integration.log 2>&1 &
        local spotlight_integration_pid=$!
        echo "$spotlight_integration_pid" > "$SPOTLIGHT_INTEGRATION_PID_FILE"
        log_success "Spotlight Integration API service started in background (PID: $spotlight_integration_pid)"
    else
        python hockey-analytics/spotlight_integration_api.py &
        local spotlight_integration_pid=$!
        echo "$spotlight_integration_pid" > "$SPOTLIGHT_INTEGRATION_PID_FILE"
        log_success "Spotlight Integration API service started (PID: $spotlight_integration_pid)"
    fi
    
    return 0
}

# Start all services
start_all() {
    local background_mode=false
    local force_restart=false
    
    # Parse arguments
    for arg in "$@"; do
        case "$arg" in
            "--background"|"-b") background_mode=true ;;
            "--force"|"-f") force_restart=true ;;
        esac
    done
    
    if [[ "$background_mode" == false ]]; then
        show_banner
    fi
    log_header "Starting All TSAI Jarvis Services"
    
    # If force restart, stop everything first
    if [[ "$force_restart" == true ]]; then
        log_step "Force restart requested - stopping all services first..."
        stop_services >/dev/null 2>&1
        sleep 3
    fi
    
    # Define service startup order and dependencies
    local services=(
        "Dashboard:$DASHBOARD_PORT:start_dashboard"
        "Core API:$CORE_API_PORT:start_core_api"
        "Enterprise Integration:$ENTERPRISE_INTEGRATION_PORT:start_enterprise_integration"
        "Advanced Visualization:$ADVANCED_VISUALIZATION_PORT:start_advanced_visualization"
        "Machine Learning:$MACHINE_LEARNING_PORT:start_machine_learning"
        "Enterprise Security:$ENTERPRISE_SECURITY_PORT:start_enterprise_security"
        "Advanced Analytics:$ADVANCED_ANALYTICS_PORT:start_advanced_analytics"
        "Toolchain Integration:$TOOLCHAIN_INTEGRATION_PORT:start_toolchain_integration"
        "Autopilot Integration:$AUTOPILOT_INTEGRATION_PORT:start_autopilot_integration"
        "Spotlight Integration:$SPOTLIGHT_INTEGRATION_PORT:start_spotlight_integration"
    )
    
    # Clean up any remaining processes on our ports first
    log_step "Pre-startup port cleanup..."
    local all_ports=(
        "$DASHBOARD_PORT:Dashboard"
        "$CORE_API_PORT:Core API"
        "$API_GATEWAY_PORT:API Gateway"
        "$ENTERPRISE_INTEGRATION_PORT:Enterprise Integration"
        "$ADVANCED_VISUALIZATION_PORT:Advanced Visualization"
        "$MACHINE_LEARNING_PORT:Machine Learning"
        "$ENTERPRISE_SECURITY_PORT:Enterprise Security"
        "$ADVANCED_ANALYTICS_PORT:Advanced Analytics"
        "$TOOLCHAIN_INTEGRATION_PORT:Toolchain Integration"
        "$AUTOPILOT_INTEGRATION_PORT:Autopilot Integration"
        "$SPOTLIGHT_INTEGRATION_PORT:Spotlight Integration"
    )
    
    for port_info in "${all_ports[@]}"; do
        local port="${port_info%%:*}"
        local service="${port_info##*:}"
        if is_port_in_use "$port"; then
            log_warning "Found process on $service port $port - cleaning up..."
            kill_port_processes "$port" "$service"
        fi
    done
    
    local failed_services=()
    local started_services=()
    
    # Start services in order
    for service_info in "${services[@]}"; do
        local service_name="${service_info%%:*}"
        local port="${service_info#*:}"
        port="${port%%:*}"
        local start_function="${service_info##*:}"
        
        log_step "Starting $service_name..."
        
        if start_service_robust "$service_name" "$port" "$start_function" "$background_mode"; then
            started_services+=("$service_name")
            log_success "$service_name started successfully"
        else
            failed_services+=("$service_name")
            log_error "Failed to start $service_name"
            
            # If this is a critical service, stop everything
            if [[ "$service_name" == "Dashboard" ]]; then
                log_error "Critical service failed - stopping all services"
                stop_services >/dev/null 2>&1
                return 1
            fi
        fi
        
        # Brief pause between services
        sleep 1
    done
    
    # Report results
    if [[ ${#failed_services[@]} -eq 0 ]]; then
        if [[ "$background_mode" == false ]]; then
            echo -e "${BRIGHT_GREEN}"
            echo "┌─────────────────────────────────────────────────────────────────┐"
            echo "│                    🎉 All Services Started!                    │"
            echo "├─────────────────────────────────────────────────────────────────┤"
            echo "│  🎨 Dashboard:        http://localhost:8000                    │"
            echo "│  ⚙️  Core API:         http://localhost:8001                    │"
            echo "│  🏢 Enterprise:       http://localhost:8007                    │"
            echo "│  📊 Visualization:    http://localhost:8008                    │"
            echo "│  🤖 Machine Learning:  http://localhost:8009                    │"
            echo "│  🔒 Security:         http://localhost:8010                    │"
            echo "│  📈 Analytics:        http://localhost:8011                    │"
            echo "└─────────────────────────────────────────────────────────────────┘"
            echo -e "${NC}"
            
            log_warning "Press Ctrl+C to stop all services"
            
            # Set up cleanup on exit
            cleanup_all() {
                echo ""
                log_step "Stopping all services..."
                stop_services
                exit 0
            }
            
            trap cleanup_all SIGINT SIGTERM
            
            # Wait for processes
            wait
        else
            log_success "All services started in background"
            log_info "Use './jarvis.sh stop' to stop all services"
            log_info "Use './jarvis.sh status' to check service status"
        fi
    else
        log_warning "Some services failed to start:"
        for service in "${failed_services[@]}"; do
            log_warning "  - $service"
        done
        
        if [[ ${#started_services[@]} -gt 0 ]]; then
            log_info "Successfully started services:"
            for service in "${started_services[@]}"; do
                log_info "  - $service"
            done
            log_info "Use './jarvis.sh stop' to clean up running services"
        fi
        
        return 1
    fi
}

# Enhanced function to kill process by PID with robust cleanup
kill_process_robust() {
    local pid="$1"
    local service_name="$2"
    local max_attempts=3
    local attempt=1
    
    if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
        return 0
    fi
    
    log_step "Stopping $service_name (PID: $pid)..."
    
    while [[ $attempt -le $max_attempts ]]; do
        if [[ $attempt -eq 1 ]]; then
            # First attempt: graceful kill (SIGTERM)
            kill -TERM "$pid" 2>/dev/null || true
            sleep 3
        elif [[ $attempt -eq 2 ]]; then
            # Second attempt: SIGINT
            kill -INT "$pid" 2>/dev/null || true
            sleep 2
        else
            # Final attempt: SIGKILL
            log_warning "Force killing $service_name (PID: $pid)..."
            kill -9 "$pid" 2>/dev/null || true
            sleep 1
        fi
        
        # Check if process is still running
        if ! kill -0 "$pid" 2>/dev/null; then
            log_success "$service_name stopped successfully"
            return 0
        fi
        
        ((attempt++))
    done
    
    # If we get here, the process is still running
    log_error "Failed to stop $service_name (PID: $pid) after $max_attempts attempts"
    log_info "You may need to kill it manually: sudo kill -9 $pid"
    return 1
}

# Enhanced function to kill all processes on a specific port
kill_port_processes() {
    local port="$1"
    local service_name="$2"
    
    if ! is_port_in_use "$port"; then
        return 0
    fi
    
    log_step "Checking $service_name on port $port..."
    
    # Get all PIDs using this port
    local pids
    pids=$(lsof -ti:"$port" 2>/dev/null || true)
    
    if [[ -z "$pids" ]]; then
        log_info "$service_name not running on port $port"
        return 0
    fi
    
    # Kill each process
    for pid in $pids; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill_process_robust "$pid" "$service_name (port $port)"
        fi
    done
    
    # Verify port is free
    sleep 2
    if is_port_in_use "$port"; then
        log_warning "Port $port is still in use after cleanup attempts"
        return 1
    else
        log_success "Port $port is now free"
        return 0
    fi
}

# Enhanced function to kill processes by pattern
kill_processes_by_pattern() {
    local pattern="$1"
    local service_name="$2"
    
    log_step "Checking for $service_name processes..."
    
    # Find processes matching pattern
    local pids
    pids=$(pgrep -f "$pattern" 2>/dev/null || true)
    
    if [[ -z "$pids" ]]; then
        log_info "No $service_name processes found"
        return 0
    fi
    
    # Kill each process
    for pid in $pids; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill_process_robust "$pid" "$service_name"
        fi
    done
}

# Stop all services with comprehensive cleanup
stop_services() {
    show_banner
    log_header "Stopping TSAI Jarvis Services"
    
    echo -e "${BRIGHT_YELLOW}"
    echo "┌─────────────────────────────────────────────────────────────────┐"
    echo "│                    🛑 Comprehensive Shutdown                    │"
    echo "├─────────────────────────────────────────────────────────────────┤"
    echo "│  🔍 Scanning for all running services...                      │"
    echo "│  🧹 Gracefully shutting down processes...                     │"
    echo "│  🚫 Force killing stubborn processes...                       │"
    echo "│  📁 Removing temporary files...                               │"
    echo "│  🔒 Ensuring all ports are free...                            │"
    echo "└─────────────────────────────────────────────────────────────────┘"
    echo -e "${NC}"
    
    # Define all our service ports
    local service_ports=(
        "$DASHBOARD_PORT:Dashboard"
        "$CORE_API_PORT:Core API"
        "$API_GATEWAY_PORT:API Gateway"
        "$ENTERPRISE_INTEGRATION_PORT:Enterprise Integration"
        "$ADVANCED_VISUALIZATION_PORT:Advanced Visualization"
        "$MACHINE_LEARNING_PORT:Machine Learning"
        "$ENTERPRISE_SECURITY_PORT:Enterprise Security"
        "$ADVANCED_ANALYTICS_PORT:Advanced Analytics"
        "$TOOLCHAIN_INTEGRATION_PORT:Toolchain Integration"
        "$AUTOPILOT_INTEGRATION_PORT:Autopilot Integration"
        "$SPOTLIGHT_INTEGRATION_PORT:Spotlight Integration"
    )
    
    # Graceful shutdown of tracked processes first
    log_step "Gracefully shutting down tracked processes..."
    
    # Graceful shutdown of each tracked service
    if [[ -f "$DASHBOARD_PID_FILE" ]]; then
        local dashboard_pid=$(cat "$DASHBOARD_PID_FILE" 2>/dev/null || echo "")
        if [[ -n "$dashboard_pid" ]]; then
            graceful_shutdown "Dashboard" "$dashboard_pid" "$DASHBOARD_PORT"
        fi
    fi
    
    if [[ -f "$CORE_API_PID_FILE" ]]; then
        local core_api_pid=$(cat "$CORE_API_PID_FILE" 2>/dev/null || echo "")
        if [[ -n "$core_api_pid" ]]; then
            graceful_shutdown "Core API" "$core_api_pid" "$CORE_API_PORT"
        fi
    fi
    
    if [[ -f "$ENTERPRISE_INTEGRATION_PID_FILE" ]]; then
        local enterprise_pid=$(cat "$ENTERPRISE_INTEGRATION_PID_FILE" 2>/dev/null || echo "")
        if [[ -n "$enterprise_pid" ]]; then
            graceful_shutdown "Enterprise Integration" "$enterprise_pid" "$ENTERPRISE_INTEGRATION_PORT"
        fi
    fi
    
    if [[ -f "$ADVANCED_VISUALIZATION_PID_FILE" ]]; then
        local visualization_pid=$(cat "$ADVANCED_VISUALIZATION_PID_FILE" 2>/dev/null || echo "")
        if [[ -n "$visualization_pid" ]]; then
            graceful_shutdown "Advanced Visualization" "$visualization_pid" "$ADVANCED_VISUALIZATION_PORT"
        fi
    fi
    
    if [[ -f "$MACHINE_LEARNING_PID_FILE" ]]; then
        local ml_pid=$(cat "$MACHINE_LEARNING_PID_FILE" 2>/dev/null || echo "")
        if [[ -n "$ml_pid" ]]; then
            graceful_shutdown "Machine Learning" "$ml_pid" "$MACHINE_LEARNING_PORT"
        fi
    fi
    
    if [[ -f "$ENTERPRISE_SECURITY_PID_FILE" ]]; then
        local security_pid=$(cat "$ENTERPRISE_SECURITY_PID_FILE" 2>/dev/null || echo "")
        if [[ -n "$security_pid" ]]; then
            graceful_shutdown "Enterprise Security" "$security_pid" "$ENTERPRISE_SECURITY_PORT"
        fi
    fi
    
    if [[ -f "$ADVANCED_ANALYTICS_PID_FILE" ]]; then
        local analytics_pid=$(cat "$ADVANCED_ANALYTICS_PID_FILE" 2>/dev/null || echo "")
        if [[ -n "$analytics_pid" ]]; then
            graceful_shutdown "Advanced Analytics" "$analytics_pid" "$ADVANCED_ANALYTICS_PORT"
        fi
    fi
    
    # Kill all processes on our service ports
    log_step "Cleaning up all service ports..."
    for port_info in "${service_ports[@]}"; do
        local port="${port_info%%:*}"
        local service="${port_info##*:}"
        kill_port_processes "$port" "$service"
    done
    
    # Kill processes by pattern (catch any remaining)
    log_step "Killing processes by pattern..."
    kill_processes_by_pattern "uvicorn.*jarvis" "Uvicorn Jarvis"
    kill_processes_by_pattern "uvicorn.*hockey" "Uvicorn Hockey"
    kill_processes_by_pattern "node.*next" "Next.js Dashboard"
    kill_processes_by_pattern "npm.*dev" "NPM Development Server"
    
    # Kill any remaining processes on our ports (final cleanup)
    log_step "Final port cleanup..."
    for port_info in "${service_ports[@]}"; do
        local port="${port_info%%:*}"
        local service="${port_info##*:}"
        if is_port_in_use "$port"; then
            log_warning "Port $port still in use, force killing all processes..."
            local pids
            pids=$(lsof -ti:"$port" 2>/dev/null || true)
            for pid in $pids; do
                if [[ -n "$pid" ]]; then
                    log_warning "Force killing process $pid on port $port"
                    kill -9 "$pid" 2>/dev/null || true
                fi
            done
            sleep 1
        fi
    done
    
    # Clean up PID files
    log_step "Cleaning up temporary files..."
    rm -f "$DASHBOARD_PID_FILE" "$CORE_API_PID_FILE" "$API_GATEWAY_PID_FILE"
    rm -f "$ENTERPRISE_INTEGRATION_PID_FILE" "$ADVANCED_VISUALIZATION_PID_FILE"
    rm -f "$MACHINE_LEARNING_PID_FILE" "$ENTERPRISE_SECURITY_PID_FILE" "$ADVANCED_ANALYTICS_PID_FILE"
    log_success "Temporary files cleaned up"
    
    # Final verification
    log_step "Verifying all services are stopped..."
    local still_running=()
    for port_info in "${service_ports[@]}"; do
        local port="${port_info%%:*}"
        local service="${port_info##*:}"
        if is_port_in_use "$port"; then
            still_running+=("$service (port $port)")
        fi
    done
    
    if [[ ${#still_running[@]} -gt 0 ]]; then
        log_warning "Some services may still be running:"
        for service in "${still_running[@]}"; do
            log_warning "  - $service"
        done
        log_info "You may need to manually kill these processes"
    else
        log_success "All services successfully stopped"
    fi
    
    echo -e "${BRIGHT_GREEN}"
    echo "┌─────────────────────────────────────────────────────────────────┐"
    echo "│                    ✅ Comprehensive Shutdown Complete          │"
    echo "├─────────────────────────────────────────────────────────────────┤"
    echo "│  🎯 All TSAI Jarvis services have been stopped                 │"
    echo "│  🧹 All processes and resources have been cleaned up           │"
    echo "│  📁 All temporary files have been removed                      │"
    echo "│  🔒 All service ports have been freed                          │"
    echo "└─────────────────────────────────────────────────────────────────┘"
    echo -e "${NC}"
}

# Graceful shutdown function
graceful_shutdown() {
    local service_name="$1"
    local pid="$2"
    local port="$3"
    
    if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
        return 0
    fi
    
    log_step "Gracefully shutting down $service_name (PID: $pid)..."
    
    # Send SIGTERM for graceful shutdown
    kill -TERM "$pid" 2>/dev/null || true
    
    # Wait for graceful shutdown
    local max_wait=10
    local wait_time=0
    
    while [[ $wait_time -lt $max_wait ]]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            log_success "$service_name shut down gracefully"
            return 0
        fi
        
        # Check if port is still in use
        if [[ -n "$port" ]] && ! is_port_in_use "$port"; then
            log_success "$service_name released port $port"
            # Process might still be running but port is free
            sleep 1
            if ! kill -0 "$pid" 2>/dev/null; then
                log_success "$service_name shut down gracefully"
                return 0
            fi
        fi
        
        sleep 1
        ((wait_time++))
    done
    
    # If still running, force kill
    log_warning "$service_name did not shut down gracefully, force killing..."
    kill -9 "$pid" 2>/dev/null || true
    sleep 1
    
    if ! kill -0 "$pid" 2>/dev/null; then
        log_success "$service_name force stopped"
        return 0
    else
        log_error "Failed to stop $service_name (PID: $pid)"
        return 1
    fi
}

# Check service status
check_status() {
    show_banner
    log_header "TSAI Jarvis Service Status"
    
    echo -e "${BRIGHT_CYAN}"
    echo "┌─────────────────────────────────────────────────────────────────┐"
    echo "│                    📊 Service Status Report                     │"
    echo "├─────────────────────────────────────────────────────────────────┤"
    echo "│  🔍 Checking all TSAI Jarvis services...                     │"
    echo "└─────────────────────────────────────────────────────────────────┘"
    echo -e "${NC}"
    
    # Check Dashboard
    log_step "Checking Dashboard..."
    if is_port_in_use "$DASHBOARD_PORT"; then
        local dashboard_pid
        dashboard_pid=$(lsof -Pi ":$DASHBOARD_PORT" -sTCP:LISTEN -t 2>/dev/null | head -1)
        
        if [[ -f "$DASHBOARD_PID_FILE" ]]; then
            local tracked_pid
            tracked_pid=$(cat "$DASHBOARD_PID_FILE")
            if [[ "$dashboard_pid" == "$tracked_pid" ]] && kill -0 "$tracked_pid" 2>/dev/null; then
                log_success "Dashboard: Running (tracked)"
                log_pid "Process ID: $dashboard_pid"
                log_url "URL: http://localhost:$DASHBOARD_PORT"
            else
                log_warning "Dashboard: Port in use by different process (PID: $dashboard_pid)"
                log_info "This may be from a previous session. Use 'stop' to clean up."
            fi
        else
            log_warning "Dashboard: Port in use by untracked process (PID: $dashboard_pid)"
            log_info "This may be from a previous session. Use 'stop' to clean up."
        fi
    else
        log_warning "Dashboard: Not running"
    fi
    
    # Check Core API
    log_step "Checking Core API..."
    if is_port_in_use "$CORE_API_PORT"; then
        local core_api_pid
        core_api_pid=$(lsof -Pi ":$CORE_API_PORT" -sTCP:LISTEN -t 2>/dev/null | head -1)
        
        if [[ -f "$CORE_API_PID_FILE" ]]; then
            local tracked_pid
            tracked_pid=$(cat "$CORE_API_PID_FILE")
            if [[ "$core_api_pid" == "$tracked_pid" ]] && kill -0 "$tracked_pid" 2>/dev/null; then
                log_success "Core API: Running (tracked)"
                log_pid "Process ID: $core_api_pid"
                log_url "API: http://localhost:$CORE_API_PORT"
            else
                log_warning "Core API: Port in use by different process (PID: $core_api_pid)"
                log_info "This may be from a previous session. Use 'stop' to clean up."
            fi
        else
            log_warning "Core API: Port in use by untracked process (PID: $core_api_pid)"
            log_info "This may be from a previous session. Use 'stop' to clean up."
        fi
    else
        log_warning "Core API: Not running"
    fi
    
    # Check Enterprise Integration API
    log_step "Checking Enterprise Integration API..."
    if is_port_in_use "$ENTERPRISE_INTEGRATION_PORT"; then
        log_success "Enterprise Integration API: Running"
        log_url "API: http://localhost:$ENTERPRISE_INTEGRATION_PORT"
    else
        log_warning "Enterprise Integration API: Not running"
    fi
    
    # Check Advanced Visualization API
    log_step "Checking Advanced Visualization API..."
    if is_port_in_use "$ADVANCED_VISUALIZATION_PORT"; then
        log_success "Advanced Visualization API: Running"
        log_url "API: http://localhost:$ADVANCED_VISUALIZATION_PORT"
    else
        log_warning "Advanced Visualization API: Not running"
    fi
    
    # Check Machine Learning API
    log_step "Checking Machine Learning API..."
    if is_port_in_use "$MACHINE_LEARNING_PORT"; then
        log_success "Machine Learning API: Running"
        log_url "API: http://localhost:$MACHINE_LEARNING_PORT"
    else
        log_warning "Machine Learning API: Not running"
    fi
    
    # Check Enterprise Security API
    log_step "Checking Enterprise Security API..."
    if is_port_in_use "$ENTERPRISE_SECURITY_PORT"; then
        log_success "Enterprise Security API: Running"
        log_url "API: http://localhost:$ENTERPRISE_SECURITY_PORT"
    else
        log_warning "Enterprise Security API: Not running"
    fi
    
    # Check Advanced Analytics API
    log_step "Checking Advanced Analytics API..."
    if is_port_in_use "$ADVANCED_ANALYTICS_PORT"; then
        log_success "Advanced Analytics API: Running"
        log_url "API: http://localhost:$ADVANCED_ANALYTICS_PORT"
    else
        log_warning "Advanced Analytics API: Not running"
    fi
    
    # Check TSAI Ecosystem Integration Services
    log_step "Checking TSAI Ecosystem Integration Services..."
    
    # Toolchain Integration
    if is_port_in_use "$TOOLCHAIN_INTEGRATION_PORT"; then
        log_success "Toolchain Integration API: Running"
        log_url "API: http://localhost:$TOOLCHAIN_INTEGRATION_PORT"
    else
        log_warning "Toolchain Integration API: Not running"
    fi
    
    # Autopilot Integration
    if is_port_in_use "$AUTOPILOT_INTEGRATION_PORT"; then
        log_success "Autopilot Integration API: Running"
        log_url "API: http://localhost:$AUTOPILOT_INTEGRATION_PORT"
    else
        log_warning "Autopilot Integration API: Not running"
    fi
    
    # Spotlight Integration
    if is_port_in_use "$SPOTLIGHT_INTEGRATION_PORT"; then
        log_success "Spotlight Integration API: Running"
        log_url "API: http://localhost:$SPOTLIGHT_INTEGRATION_PORT"
    else
        log_warning "Spotlight Integration API: Not running"
    fi
    
    echo -e "${BRIGHT_GREEN}"
    echo "┌─────────────────────────────────────────────────────────────────┐"
    echo "│                    📋 Status Summary                            │"
    echo "├─────────────────────────────────────────────────────────────────┤"
    echo "│  🎯 Use 'start --dashboard' to start dashboard and APIs        │"
    echo "│  🚀 Use 'start --all' to start all services                    │"
    echo "│  🛑 Use 'stop' to stop all services                            │"
    echo "│  ❓ Use 'help' for more information                            │"
    echo "└─────────────────────────────────────────────────────────────────┘"
    echo -e "${NC}"
}

# Show short help information
show_help_short() {
    show_banner
    
    echo -e "${BRIGHT_CYAN}"
    echo "┌─────────────────────────────────────────────────────────────────┐"
    echo "│                    📖 TSAI Jarvis CLI                          │"
    echo "│  🏒 AI-Powered Hockey Analytics Platform                        │"
    echo "└─────────────────────────────────────────────────────────────────┘"
    echo -e "${NC}"
    
    echo -e "${BRIGHT_BLUE}"
    echo "┌─ ${BOLD}USAGE${NC} ${BRIGHT_BLUE}─────────────────────────────────────────────────────────┐"
    echo "│${NC} $SCRIPT_NAME <command> [options]"
    echo -e "${BRIGHT_BLUE}└─────────────────────────────────────────────────────────────────┘${NC}"
    
    echo -e "${BRIGHT_GREEN}"
    echo "┌─ ${BOLD}COMMANDS${NC} ${BRIGHT_GREEN}─────────────────────────────────────────────────────┐"
    echo "│${NC} start --dashboard  🎨 Start dashboard and backend services"
    echo "│${NC} start --all        🚀 Start all services"
    echo "│${NC} stop               🛑 Stop all services"
    echo "│${NC} status             📊 Check service status"
    echo "│${NC} help               ❓ Show help information"
    echo -e "${BRIGHT_GREEN}└─────────────────────────────────────────────────────────────────┘${NC}"
    
    echo -e "${BRIGHT_YELLOW}"
    echo "┌─ ${BOLD}QUICK START${NC} ${BRIGHT_YELLOW}───────────────────────────────────────────────┐"
    echo "│${NC} $SCRIPT_NAME start --all     # Start everything"
    echo "│${NC} $SCRIPT_NAME status          # Check what's running"
    echo "│${NC} $SCRIPT_NAME stop            # Stop everything"
    echo "│${NC} $SCRIPT_NAME help            # Full help"
    echo -e "${BRIGHT_YELLOW}└─────────────────────────────────────────────────────────────────┘${NC}"
    
    echo -e "${BRIGHT_PURPLE}"
    echo "┌─ ${BOLD}SERVICES${NC} ${BRIGHT_PURPLE}─────────────────────────────────────────────────────┐"
    echo "│${NC} 🎨 Dashboard: http://localhost:8000  ⚙️  Core API: http://localhost:8001"
    echo "│${NC} 🏢 Enterprise: http://localhost:8007  📊 Visualization: http://localhost:8008"
    echo "│${NC} 🤖 ML: http://localhost:8009  🔒 Security: http://localhost:8010"
    echo "│${NC} 📈 Analytics: http://localhost:8011"
    echo -e "${BRIGHT_PURPLE}└─────────────────────────────────────────────────────────────────┘${NC}"
}

# Show full help information
show_help() {
    show_banner
    
    echo -e "${BRIGHT_CYAN}"
    echo "┌─────────────────────────────────────────────────────────────────┐"
    echo "│                    📖 Command Line Interface                    │"
    echo "├─────────────────────────────────────────────────────────────────┤"
    echo "│  🏒 TSAI Jarvis - AI-Powered Hockey Analytics Platform        │"
    echo "└─────────────────────────────────────────────────────────────────┘"
    echo -e "${NC}"
    
    echo -e "${BRIGHT_BLUE}"
    echo "┌─ ${BOLD}USAGE${NC} ${BRIGHT_BLUE}─────────────────────────────────────────────────────────┐"
    echo "│${NC} $SCRIPT_NAME <command> [options]"
    echo -e "${BRIGHT_BLUE}└─────────────────────────────────────────────────────────────────┘${NC}"
    
    echo -e "${BRIGHT_GREEN}"
    echo "┌─ ${BOLD}COMMANDS${NC} ${BRIGHT_GREEN}─────────────────────────────────────────────────────┐"
    echo "│${NC} start --dashboard  🎨 Start dashboard and backend services"
    echo "│${NC} start --all        🚀 Start all services"
    echo "│${NC} stop               🛑 Stop all TSAI Jarvis services"
    echo "│${NC} status             📊 Check status of all services"
    echo "│${NC} help               ❓ Show this help message"
    echo -e "${BRIGHT_GREEN}└─────────────────────────────────────────────────────────────────┘${NC}"
    
    echo -e "${BRIGHT_YELLOW}"
    echo "┌─ ${BOLD}OPTIONS${NC} ${BRIGHT_YELLOW}─────────────────────────────────────────────────────┐"
    echo "│${NC} --dashboard       🎨 Start/stop dashboard and backend services"
    echo "│${NC} --all              🚀 Start/stop all services"
    echo "│${NC} --background, -b   🔄 Run services in background mode"
    echo "│${NC} --force, -f        🔄 Force restart (stop all, then start)"
    echo -e "${BRIGHT_YELLOW}└─────────────────────────────────────────────────────────────────┘${NC}"
    
    echo -e "${BRIGHT_PURPLE}"
    echo "┌─ ${BOLD}ENVIRONMENT VARIABLES${NC} ${BRIGHT_PURPLE}─────────────────────────────────────┐"
    echo "│${NC} DASHBOARD_PORT                    🎨 Dashboard port (default: 8000)"
    echo "│${NC} CORE_API_PORT                      ⚙️  Core API port (default: 8001)"
    echo "│${NC} ENTERPRISE_INTEGRATION_PORT        🏢 Enterprise port (default: 8007)"
    echo "│${NC} ADVANCED_VISUALIZATION_PORT        📊 Visualization port (default: 8008)"
    echo "│${NC} MACHINE_LEARNING_PORT              🤖 ML port (default: 8009)"
    echo "│${NC} ENTERPRISE_SECURITY_PORT           🔒 Security port (default: 8010)"
    echo "│${NC} ADVANCED_ANALYTICS_PORT            📈 Analytics port (default: 8011)"
    echo -e "${BRIGHT_PURPLE}└─────────────────────────────────────────────────────────────────┘${NC}"
    
    echo -e "${BRIGHT_CYAN}"
    echo "┌─ ${BOLD}EXAMPLES${NC} ${BRIGHT_CYAN}─────────────────────────────────────────────────────┐"
    echo "│${NC} $SCRIPT_NAME start --dashboard"
    echo "│${NC} $SCRIPT_NAME start --all"
    echo "│${NC} $SCRIPT_NAME start --all --background"
    echo "│${NC} $SCRIPT_NAME start --all --force"
    echo "│${NC} $SCRIPT_NAME stop"
    echo "│${NC} $SCRIPT_NAME status"
    echo -e "${BRIGHT_CYAN}└─────────────────────────────────────────────────────────────────┘${NC}"
    
    echo -e "${BRIGHT_GREEN}"
    echo "┌─ ${BOLD}SERVICES${NC} ${BRIGHT_GREEN}─────────────────────────────────────────────────────┐"
    echo "│${NC} 🎨 Dashboard:        http://localhost:8000"
    echo "│${NC} ⚙️  Core API:         http://localhost:8001"
    echo "│${NC} 🏢 Enterprise:        http://localhost:8007"
    echo "│${NC} 📊 Visualization:     http://localhost:8008"
    echo "│${NC} 🤖 Machine Learning:  http://localhost:8009"
    echo "│${NC} 🔒 Security:          http://localhost:8010"
    echo "│${NC} 📈 Analytics:         http://localhost:8011"
    echo -e "${BRIGHT_GREEN}└─────────────────────────────────────────────────────────────────┘${NC}"
    
    echo -e "${BRIGHT_PURPLE}"
    echo "┌─ ${BOLD}FEATURES${NC} ${BRIGHT_PURPLE}─────────────────────────────────────────────────────┐"
    echo "│${NC} 🏒 Real-time hockey player detection and tracking"
    echo "│${NC} 🤖 AI/ML model training and management"
    echo "│${NC} 📊 Modern dashboard with live monitoring"
    echo "│${NC} 🎨 Advanced 3D visualization and heat maps"
    echo "│${NC} 🔒 Enterprise-grade security and authentication"
    echo "│${NC} 📈 Advanced analytics and predictive modeling"
    echo "│${NC} 🚀 Production-ready microservices architecture"
    echo "│${NC} 🔄 Real-time WebSocket communication"
    echo -e "${BRIGHT_PURPLE}└─────────────────────────────────────────────────────────────────┘${NC}"
    
    echo -e "${BRIGHT_BLUE}"
    echo "┌─ ${BOLD}LINKS${NC} ${BRIGHT_BLUE}─────────────────────────────────────────────────────────┐"
    echo "│${NC} 🌐 GitHub: https://github.com/baseonballs/tsai-jarvis"
    echo "│${NC} 🎨 Dashboard: http://localhost:8000 (when running)"
    echo -e "${BRIGHT_BLUE}└─────────────────────────────────────────────────────────────────┘${NC}"
}

# Main function with enhanced error handling
main() {
    # Set up error handling
    set -euo pipefail
    
    # Trap errors and provide helpful messages
    trap 'log_error "Script failed at line $LINENO. Command: $BASH_COMMAND"' ERR
    
    # Check if we're in the right directory
    if ! check_project_root; then
        exit 1
    fi
    
    # Check dependencies
    if ! check_dependencies; then
        exit 1
    fi
    
    # Validate arguments
    if [[ $# -eq 0 ]]; then
        show_help_short
        exit 0
    fi
    
    # Parse command line arguments
    case "${1:-}" in
        "start")
            case "${2:-}" in
                "--dashboard")
                    if ! start_dashboard "${@:3}"; then
                        log_error "Failed to start dashboard"
                        exit 1
                    fi
                    ;;
                "--all")
                    if ! start_all "${@:3}"; then
                        log_error "Failed to start all services"
                        exit 1
                    fi
                    ;;
                "")
                    error_exit "Missing option for 'start' command. Use --dashboard or --all"
                    ;;
                *)
                    error_exit "Unknown option for 'start' command: $2. Use --dashboard or --all"
                    ;;
            esac
            ;;
        "stop")
            if ! stop_services; then
                log_error "Failed to stop services"
                exit 1
            fi
            ;;
        "status")
            if ! check_status; then
                log_error "Failed to check service status"
                exit 1
            fi
            ;;
        "help"|"--help"|"-h")
            show_help
            ;;
        *)
            error_exit "Unknown command: $1. Use 'help' to see available commands."
            ;;
    esac
}

# Run main function with all arguments
main "$@"
