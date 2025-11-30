# Development Guide

This guide helps contributors set up the project locally, run unit tests, and work on features.

## Prerequisites
- Python 3.9+ (3.11 recommended)
- Node.js 18+ (for frontend)
- Git

## Setup (backend)
```bash
git clone https://github.com/IsaacsonShoko/PI_IMAGING.git
cd PI_IMAGING
python3 -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux / macOS
# source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env for local dev (you can set simulation mode True)
``` 

## Running locally (simulate hardware)
```bash
# Start backend
python app.py
# In another terminal: start camera simulator (if available)
python camera_system.py --simulate True
``` 

## Running frontend (development)
```bash
cd frontend
npm install
npm run dev
``` 

## Tests
- Unit tests are in `Unit_test_code/` directory. To run a particular test:
```bash
python -m pytest Unit_test_code/EdgeAI_Detection_Test.py
``` 

## Code style
- Follow PEP8 for Python
- Use consistent type hints where practical

## Common tasks
- Add an API route: edit `app.py`, add route and document in `docs/API_DOCUMENTATION.md`
- Modify model pipeline: edit `camera_system.py` and update `docs/EDGE_IMPULSE_INTEGRATION.md` with changes

## Debugging tips
- Use `logging` module set to DEBUG to trace behavior
- Use `pdb.set_trace()` to step through functions

## Pull request checklist
- [ ] Tests added/updated
- [ ] Documentation updated (docs/*)
- [ ] `requirements.txt` updated if dependencies changed
- [ ] `frontend` build passes (if frontend changed)

## Branching strategy
- `main`: stable release
- `dev`: active development
- feature branches: `feature/<short-desc>`

