# PhishNet - AI-Powered Phishing Detection

[![API Status](https://img.shields.io/badge/API-Live-brightgreen)](https://phishnet-api.onrender.com/health)
[![Web App](https://img.shields.io/badge/Web-Live-blue)](https://epbaid.github.io/phishnet/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Production-ready phishing detection API using ensemble machine learning with LLM-powered explanations.**

> **Live Demo**: [https://epbaid.github.io/phishnet/](https://epbaid.github.io/phishnet/)

## Features

- **Multi-Model Ensemble**: Combines URL, WHOIS, and DNS analysis for robust detection
- **90%+ Accuracy**: Trained on 40,000+ URLs from 13 threat intelligence sources
- **LLM Explanations**: Groq-powered natural language explanations for every prediction
- **Production API**: FastAPI with rate limiting, input validation, and structured logging
- **Continuous Learning**: 24/7 data collection with daily model retraining
- **Browser Extension**: Chrome extension for real-time URL checking
- **Docker Deployment**: Multi-stage builds with health checks

## Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/EpbAiD/phishnet.git
cd phishnet

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment config
cp .env.example .env
```

### 2. Run API Server

```bash
# Start the API
uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# Or use Docker
docker-compose up
```

### 3. Test the API

```bash
# Check health
curl http://localhost:8000/health

# Predict phishing risk
curl -X POST "http://localhost:8000/predict/ensemble" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'
```

**Interactive API Docs**: http://localhost:8000/docs

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Collection                          │
│  Kaggle + OpenPhish + URLhaus + Majestic + Tranco (40K URLs)│
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                 Feature Engineering                          │
│  • URL Features (39): domain, path, subdomain analysis       │
│  • WHOIS Features (15): domain age, registrar, privacy       │
│  • DNS Features: MX records, nameservers, SPF/DMARC         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   Model Training                             │
│  • 9 models tested (CatBoost, XGBoost, LightGBM, etc.)      │
│  • 5-fold stratified cross-validation                       │
│  • Best: CatBoost (881KB, 90% accuracy)                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                 Production API (FastAPI)                     │
│  • /predict/url: URL-only (fast, <50ms)                     │
│  • /predict/whois: Domain reputation (1-10s)                │
│  • /predict/ensemble: Best accuracy (weighted)               │
│  • /explain: Human-readable explanations (SHAP + LLM)       │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
phishnet/
├── .github/
│   └── workflows/
│       └── ci.yml              # CI/CD pipeline (GitHub Actions)
├── src/
│   ├── api/
│   │   ├── app.py             # FastAPI application
│   │   ├── security.py        # Rate limiting, validation
│   │   ├── model_loader.py    # Model caching
│   │   └── model_metadata.py  # Model versioning
│   ├── features/
│   │   ├── url_features.py    # URL feature extraction
│   │   ├── whois.py           # WHOIS feature extraction
│   │   └── dns_ipwhois.py     # DNS feature extraction
│   ├── training/
│   │   ├── url_train.py       # URL model training
│   │   ├── whois_train.py     # WHOIS model training
│   │   └── model_zoo.py       # Model configurations
│   ├── utils/
│   │   └── logger.py          # Structured logging
│   └── config.py              # Configuration management
├── tests/
│   ├── test_url_validator.py # Unit tests
│   ├── test_config.py         # Config tests
│   └── test_model_metadata.py # Metadata tests
├── models/                    # Trained models (.pkl files)
├── data/
│   ├── raw/                   # Raw datasets
│   └── processed/             # Preprocessed features
├── Dockerfile                 # Multi-stage Docker build
├── docker-compose.yml         # Docker orchestration
├── requirements.txt           # Production dependencies
├── requirements-dev.txt       # Development dependencies
├── pytest.ini                 # Pytest configuration
└── .env.example              # Environment config template
```

## API Endpoints

| Endpoint | Method | Speed | Accuracy | Use Case |
|----------|--------|-------|----------|----------|
| `/health` | GET | Instant | N/A | Health check |
| `/predict/url` | POST | 10-50ms | 90% | Fast URL screening |
| `/predict/whois` | POST | 1-10s | 90% | Domain reputation |
| `/predict/ensemble` | POST | 1-10s | **Best** | Production use |
| `/explain` | POST | 2-15s | N/A | Human explanations |

### Example Request

```python
import requests

response = requests.post(
    "http://localhost:8000/predict/ensemble",
    json={"url": "https://paypal-verify.suspicious-domain.xyz"}
)

print(response.json())
# {
#   "url": "https://paypal-verify.suspicious-domain.xyz",
#   "risk_score": 0.987,
#   "is_phishing": true,
#   "threshold": 0.5,
#   "model_name": "ensemble_url+whois_v1",
#   "latency_ms": 1243.5
# }
```

## Development

### Run Tests

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run unit tests with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_url_validator.py -v

# Run with markers
pytest -m security  # Only security tests
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/

# Security scan
bandit -r src/
```

### Training New Models

```bash
# 1. Collect data
python src/data_fetch/fetch.py

# 2. Extract features
python src/features/url_features.py --mode dataset

# 3. Build model-ready dataset
python src/data_prep/dataset_builder.py

# 4. Train models
python src/training/url_train.py
python src/training/whois_train.py

# 5. Evaluate models
python evaluation/model_comparison/test_model_selection.py
```

## Performance Metrics

| Model | Accuracy | Precision | Recall | F1 Score | Size |
|-------|----------|-----------|--------|----------|------|
| **URL (CatBoost)** | 90% | 88% | 92% | 90% | 881KB |
| **WHOIS (CatBoost)** | 90% | 89% | 91% | 90% | 875KB |
| **Ensemble** | **92%** | **90%** | **93%** | **91%** | - |

**Latency Benchmarks:**
- URL-only: 10-50ms
- WHOIS: 1,000-10,000ms (network-dependent)
- Ensemble: ~1,500ms average

## Configuration

Environment variables (`.env`):

```bash
# API
API_HOST=0.0.0.0
API_PORT=8000

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# Model Versioning
MODEL_VERSION=v1.0.0

# Ensemble Weights
ENSEMBLE_URL_WEIGHT=0.6
ENSEMBLE_WHOIS_WEIGHT=0.4

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

## Security

- **Input Validation**: URL length limits, pattern blocking (XSS, injection)
- **Rate Limiting**: 100 requests/minute per IP (configurable)
- **CORS**: Restricted origins (configurable)
- **API Keys**: Optional authentication (disabled by default)

## Deployment

### Docker

```bash
# Build image
docker build -t phishing-detection:latest .

# Run container
docker run -d -p 8000:8000 phishing-detection:latest
```

### Docker Compose

```bash
docker-compose up -d
```

### Production Checklist

- [ ] Enable API key authentication (`API_KEY_ENABLED=true`)
- [ ] Configure CORS origins (`CORS_ORIGINS=https://yourdomain.com`)
- [ ] Set log level to WARNING/ERROR
- [ ] Enable metrics collection (`ENABLE_METRICS=true`)
- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Configure backups for models/
- [ ] Set up alerts for errors/performance

## Technical Highlights

### 1. Temporal Drift Fix
**Problem**: Training data from 2010-2015 caused 20% false positives on modern URLs.

**Solution**: Augmented dataset with 102 modern tech URLs (GitHub, StackOverflow, SaaS platforms).

**Result**: Accuracy improved from 80% → 90%.

### 2. Ensemble Learning
**Approach**: Weighted average of URL (60%) + WHOIS (40%) models.

**Rationale**:
- URL model: Fast, reliable baseline
- WHOIS model: Adds domain reputation context

**Result**: Best overall accuracy with acceptable latency.

### 3. MLOps Best Practices
- **Model Versioning**: Metadata tracking (version, accuracy, trained date)
- **CI/CD Pipeline**: GitHub Actions for automated testing
- **Structured Logging**: JSON logs with correlation IDs
- **Configuration Management**: Environment-based configs
- **Security**: Rate limiting, input validation

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

**Data Sources:**
- [Kaggle Phishing URLs](https://www.kaggle.com/datasets/hassaanmustafavi/phishing-urls-dataset)
- [OpenPhish](https://openphish.com/feed.txt)
- [URLhaus](https://urlhaus.abuse.ch)
- [Majestic Million](https://majestic.com/reports/majestic-million)
- [Tranco List](https://tranco-list.eu)

**ML Libraries:**
- CatBoost, XGBoost, LightGBM
- scikit-learn, pandas, numpy
- FastAPI, uvicorn, pydantic

## Contact

**Author**: Eeshan Bhanap
**GitHub**: [@EpbAiD](https://github.com/EpbAiD)

---

**Built for cybersecurity research and education.**
