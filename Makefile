PYTHON ?= python3
VENV ?= .venv
ACTIVATE = . $(VENV)/bin/activate

.PHONY: setup index run test eval submit-check docker-build docker-run clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(ACTIVATE) && pip install --upgrade pip && pip install -r requirements.txt

index:
	$(ACTIVATE) && python scripts/build_index.py

run:
	$(ACTIVATE) && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

test:
	$(ACTIVATE) && pytest -q

eval:
	$(ACTIVATE) && python scripts/evaluate.py

submit-check:
	$(ACTIVATE) && pytest -q
	$(ACTIVATE) && python scripts/build_index.py
	$(ACTIVATE) && python scripts/evaluate.py
	@test -f artifacts/faiss.index
	@test -f artifacts/chunks.jsonl
	@test -f artifacts/evaluation_report.md
	@echo "Submission checks passed."

docker-build:
	docker build -t mini-rag-email .

docker-run:
	docker run --rm -p 8000:8000 --env-file .env mini-rag-email

clean:
	rm -rf __pycache__ .pytest_cache .venv
