# FastAPI Backend Starter Pack

AI-powered platform for data insights and analytics - Backend API Services

## Features

**Structured Logging & Observability**
- JSON structured logging with correlation IDs
- OpenTelemetry distributed tracing
- Prometheus metrics integration
- Request/response logging middleware

**Error Handling**
- Comprehensive custom exceptions
- Global exception handlers
- Standardized error responses
- Error tracking and logging

**Testing Framework**
- Pytest with async support
- Test fixtures and utilities
- Coverage reporting
- Integration testing setup

**Dockerized Deployment**
- Multi-stage Docker builds
- Docker Compose orchestration
- Prometheus + Grafana monitoring
- PostgreSQL and Redis services

## Technology Stack

- **Framework**: FastAPI
- **Database**: PostgreSQL with AsyncPG
- **Cache**: Redis
- **Monitoring**: Prometheus + Grafana
- **Tracing**: OpenTelemetry
- **Logging**: Structlog
- **Testing**: Pytest
- **Containerization**: Docker + Docker Compose

## Quick Start

### Prerequisites

- Python 3.12+
- Docker and Docker Compose
- PostgreSQL (for local development)
- Redis (for local development)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd aipal-backend-services
   ```

2. **Install dependencies**
   ```bash
   make install
   ```

3. **Start services with Docker**
   ```bash
   make up
   ```

4. **Or run locally for development**
   ```bash
   make db-up    # Start only database services
   make dev      # Start development server
   ```

### Development Workflow

```bash
# Install dependencies
make install

# Start development server
make dev

# Run tests
make test

# Run tests with coverage
make test-cov

# Format code
make format

# Run linting
make lint

# Run all quality checks
make check

# View logs
make logs
```

## API Documentation

Once the server is running, visit:
- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

## Monitoring

- **Prometheus**: http://localhost:9091
- **Grafana**: http://localhost:3000 (admin/admin)
- **Application Metrics**: http://localhost:9090/metrics

## Health Checks

- **Health**: http://localhost:8000/health
- **Readiness**: http://localhost:8000/ready

## Configuration

The application uses environment variables for configuration. Key settings:

```bash
# Application
ENVIRONMENT=development
DEBUG=true
HOST=0.0.0.0
PORT=8000

# Database
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/aipal

# Redis
REDIS_URL=redis://localhost:6379/0

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json  # or 'text' for development

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
```

## Project Structure

```
aipal-backend-services/
 api/                    # API routes and endpoints
 core/                   # Core application logic
    config.py          # Configuration management
    logging.py         # Structured logging setup
    observability.py   # OpenTelemetry configuration
    server.py          # FastAPI application factory
    exceptions/        # Custom exceptions and handlers
    middlewares/       # Custom middlewares
    utils/             # Utility functions
 tests/                  # Test files
 monitoring/             # Prometheus and Grafana config
 scripts/               # Database and deployment scripts
 Dockerfile             # Multi-stage Docker build
 docker-compose.yml     # Service orchestration
 Makefile              # Development commands
 main.py               # Application entry point
```

## Testing

The project includes comprehensive test coverage:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=core --cov=api --cov-report=html

# Run specific test file
pytest tests/test_health.py

# Run tests with verbose output
pytest -v
```

## Logging

The application uses structured logging with:
- JSON format for production
- Human-readable format for development
- Correlation IDs for request tracing
- Service information in all logs

## Error Handling

Robust error handling includes:
- Custom exception classes
- Global exception handlers
- Standardized error responses
- Detailed error logging

## Deployment

### Docker Compose (Recommended)

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f aipal-backend

# Stop services
docker-compose down
```

### Production Deployment

1. Set appropriate environment variables
2. Use a production WSGI server (configured in Docker)
3. Set up SSL/TLS termination
4. Configure monitoring and alerting
5. Set up log aggregation

## Contributing

1. Follow the existing code style
2. Write tests for new functionality
3. Run `make check` before committing
4. Use conventional commit messages

## License

[Add your license information here]
