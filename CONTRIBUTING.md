# Contributing

Thanks for your interest in contributing to **ArxiTex**.

## Quick start

1. Fork the repo and create a feature branch.
2. Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

3. Run tests:

```bash
pytest -q
```

## Code style
- Prefer small, well-scoped pull requests.
- Keep functions and modules focused.
- Add/update tests when changing behavior.

## Web changes

```bash
cd web
npm install
npm run lint
npm run build
```
