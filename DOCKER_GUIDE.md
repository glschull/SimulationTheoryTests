# Docker Setup Guide - Simulation Theory Test Kit

## 🐳 Quick Start

### Option 1: Run Complete Test Suite
```bash
# Build and run the complete analysis
docker-compose up simulation-tests

# Or with Docker directly
docker build -t simulation-theory-test-kit .
docker run -v $(pwd)/results:/app/results simulation-theory-test-kit
```

### Option 2: Interactive Development
```bash
# Start interactive development container
docker-compose up -d simulation-dev
docker-compose exec simulation-dev bash

# Inside container, you can run:
python main_runner.py --all
python quality_assurance.py
```

### Option 3: Jupyter Notebook Interface
```bash
# Start Jupyter server
docker-compose up simulation-jupyter

# Open browser to: http://localhost:8888
# No token required for local development
```

## 🏗️ Building the Container

### Standard Build
```bash
docker build -t simulation-theory-test-kit .
```

### Multi-platform Build (for sharing)
```bash
docker buildx build --platform linux/amd64,linux/arm64 -t simulation-theory-test-kit .
```

## 🚀 Running Tests

### Run All Tests
```bash
docker run \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/data:/app/data \
  simulation-theory-test-kit
```

### Run Specific Components
```bash
# Only generate data
docker run simulation-theory-test-kit python main_runner.py --generate-data

# Only run tests (no visualization)
docker run simulation-theory-test-kit python main_runner.py --run-tests --no-visualize

# Quality assurance
docker run simulation-theory-test-kit python quality_assurance.py
```

## 📊 Accessing Results

Results are automatically saved to mounted volumes:
- `./results/` - Analysis results, plots, JSON files
- `./data/` - Generated scientific datasets

## 🔧 Environment Variables

```bash
# Customize the analysis
docker run \
  -e MPLBACKEND=Agg \
  -e PYTHONUNBUFFERED=1 \
  -v $(pwd)/results:/app/results \
  simulation-theory-test-kit
```

## 🐛 Troubleshooting

### Memory Issues
```bash
# Increase Docker memory limit to 4GB+
docker run --memory=4g simulation-theory-test-kit
```

### Permission Issues
```bash
# Fix result file permissions
sudo chown -R $USER:$USER results/
```

### Rebuild After Changes
```bash
# Force rebuild
docker-compose build --no-cache simulation-tests
```

## 🌐 Sharing Your Container

### Save to File
```bash
docker save simulation-theory-test-kit > simulation-theory-test-kit.tar
```

### Load from File
```bash
docker load < simulation-theory-test-kit.tar
```

### Push to Registry
```bash
docker tag simulation-theory-test-kit your-registry/simulation-theory-test-kit
docker push your-registry/simulation-theory-test-kit
```

## 📋 Verification

To verify the container works correctly:

```bash
# Test build
docker build -t simulation-theory-test-kit .

# Test run (should complete without errors)
docker run --rm simulation-theory-test-kit python -c "import main_runner; print('✅ Container working!')"

# Full test (will take ~60 seconds)
docker run --rm -v $(pwd)/test-results:/app/results simulation-theory-test-kit
```

## 🎯 Production Use

For running in production or CI/CD:

```bash
# Production run with specific tag
docker run \
  --name simulation-theory-production \
  -v /data/simulation-results:/app/results \
  simulation-theory-test-kit:1.0
```

---
*Container ensures reproducible results across different systems and environments.*
