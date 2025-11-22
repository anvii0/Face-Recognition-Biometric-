# FaceScan | Biometric Face Recognition Platform

<div align="center">

**A secure, intelligent face-recognition system for dependable identity verification.**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?logo=javascript&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## Overview

**FaceScan** is a full-stack biometric identity platform built to recognize, match, and verify people through facial analysis. It combines a browser-based capture experience with a Python backend, configurable recognition pipeline, challenge-response checks, and cloud signal integration.

The system is designed as a practical foundation for secure-entry and identity-verification scenarios, with modular components that make matching logic, storage, vision processing, and decision policies easy to extend.

## Highlights

- **Live camera capture** through a responsive web interface
- **Face enrollment and matching** using a dedicated biometric pipeline
- **Configurable decision thresholds** for verification confidence
- **Challenge-based validation** to improve resistance to simple presentation attacks
- **Cloud signal support** for distributed or server-backed deployments
- **Benchmark interface** for comparing recognition behavior and outcomes
- **Modular architecture** across vision, matching, storage, decisions, and API layers
- **Human-readable configuration** for tuning the system without changing application code

## How It Works

```text
Browser Camera
      |
      v
Image Capture + Preprocessing
      |
      v
Face Detection + Feature Extraction
      |
      v
Stored Identity Comparison
      |
      v
Challenge / Confidence Evaluation
      |
      v
Allow, Review, or Reject Decision
```

## Project Structure

```text
.
├── code/
│   ├── backend/
│   │   ├── main.py          # API entry point
│   │   ├── pipeline.py      # End-to-end recognition workflow
│   │   ├── vision.py        # Face and image processing
│   │   ├── matcher.py       # Identity matching
│   │   ├── people.py        # Person and enrollment management
│   │   ├── store.py         # Data persistence
│   │   ├── challenge.py     # Challenge-response checks
│   │   ├── decision.py      # Verification decisions
│   │   └── cloud_signals.py # Cloud-integrated signals
│   ├── frontend/
│   │   ├── index.html       # Main capture interface
│   │   ├── bench.html       # Recognition benchmark interface
│   │   └── *.js / *.css     # Browser application assets
│   ├── docs/
│   │   ├── architecture.md  # System design
│   │   └── plan.md          # Implementation plan
│   └── scripts/
│       └── aws_resources.py # AWS resource utilities
├── overview-images/         # Project screenshots
└── FaceScan_Team_Orbit.pdf  # Presentation deck
```

## Technology

| Area | Technology |
|---|---|
| Backend | Python, FastAPI |
| Frontend | HTML, CSS, JavaScript |
| Computer vision | Configurable face-detection and recognition modules |
| Configuration | YAML |
| Cloud integration | Optional AWS-related resources |
| Deployment | Local Uvicorn server or a Python-compatible hosting environment |

## Getting Started

### Prerequisites

- Python 3.10 or newer
- A webcam-enabled modern browser
- `pip` and a virtual environment

### Installation

```bash
git clone https://github.com/anvii0/Face-Recognition-Biometric-.git
cd Face-Recognition-Biometric-/code

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Run the Application

Start the backend from the `code` directory:

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Open the frontend through a static file server from the `code/frontend` directory:

```bash
python3 -m http.server 8080
```

Then visit `http://localhost:8080`.

> The API and frontend ports may need to match the endpoints configured in the frontend JavaScript files.

## Configuration

Runtime settings are maintained in `code/config.yaml`. Review and tune this file before deployment, including:

- Recognition and matching thresholds
- Challenge requirements
- Pipeline behavior
- Storage settings
- Cloud or external integration settings

Do not commit secrets, credentials, private images, or biometric datasets to the repository. Use environment variables or a secure secret manager for production credentials.

## Security and Responsible Use

Face recognition is sensitive technology. Deploy FaceScan only with informed consent and appropriate legal authorization.

Recommended practices include:

- Encrypt biometric templates and personal data at rest and in transit
- Restrict access with authentication and least-privilege authorization
- Maintain audit logs for enrollment, verification, and administrative actions
- Avoid single-factor reliance when high-impact decisions are involved
- Provide a documented appeal or alternative verification process
- Establish data retention and deletion policies
- Test against bias, spoofing, and false-acceptance risks
- Never commit real biometric samples or private identity data

## Screenshots

| | | |
|---|---|---|
| ![Overview 1](overview-images/1.jpg) | ![Overview 2](overview-images/2.jpg) | ![Overview 3](overview-images/3.jpg) |

## Documentation

- [System architecture](code/docs/architecture.md)
- [Implementation plan](code/docs/plan.md)
- [Project presentation](FaceScan_Team_Orbit.pdf)

## Development

Development dependencies are listed in `code/requirements-dev.txt`. Run tests or validation commands from the repository root or the `code` directory, as defined by the project tooling.

## Contributing

Contributions are welcome. Before submitting changes:

1. Create a focused feature or fix branch.
2. Keep recognition and security behavior configurable and testable.
3. Use representative, consent-based test data.
4. Document configuration or API changes.
5. Verify that no private data or credentials are included.

## License

This project is available under the MIT License. See `LICENSE` if present in the repository, or add a license file before production or public distribution.
