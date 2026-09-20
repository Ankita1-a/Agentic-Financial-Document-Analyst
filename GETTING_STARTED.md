# Getting started: raw PDFs to CI/CD

A complete, start-to-finish walkthrough for someone opening this project
folder for the first time — from ingesting raw annual report PDFs through
a working local app, Docker containers, the full stack via
`docker compose`, and a green GitHub Actions pipeline.

Every command below assumes **port 8000** for the backend throughout, and
that you're running commands from the project root unless a step says
otherwise.

This is the *build it* guide. For diagnosing something that's already
built but not behaving (a specific error, a specific phase failing), see
`VERIFY.md` instead — it's structured as a layered checklist rather than a
linear walkthrough, and has more troubleshooting depth per step.

---

## 0. Prerequisites

```bash
python3 --version      # 3.10+
docker --version
docker ps               # confirms the Docker daemon is actually running
git --version
```

You'll also need a free Mistral API key from **console.mistral.ai**.

---

## 1. Set up your local Python environment

This is for running ingestion and testing things directly, *before*
Docker enters the picture — the fastest way to catch a real bug is
outside a container, not inside one.

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Note: the root `requirements.txt` covers *everything* — ingestion,
the agent, and both services — since local development needs all of it
in one environment. `backend/requirements.txt` and `frontend/requirements.txt`
are separate, deliberately minimal subsets used only for the Docker
builds later (Step 7) — you don't need to touch those directly here.

```bash
export MISTRAL_API_KEY=your-key-here
```

Verify the key works before going any further — this bypasses the whole
project and talks to Mistral directly, so if it fails, the problem is the
key itself, not anything we're about to build:
```bash
curl https://api.mistral.ai/v1/models -H "Authorization: Bearer $MISTRAL_API_KEY"
```
Expect a real JSON list of models back, not an error.

---

## 2. Add your raw PDFs

Place your annual report PDFs somewhere under `data/raw_pdfs/`, then edit
`manifest.json` to point at them:

```json
[
  {"pdf_path": "data/raw_pdfs/TATASTEEL/2026/tatasteel-iar-2025-26.pdf", "company": "Tata Steel", "fiscal_year": 2025},
  {"pdf_path": "data/raw_pdfs/LT/2026/LT_FY2026.pdf", "company": "L&T", "fiscal_year": 2025}
]
```

Add one entry per report. `fiscal_year` should match the year the report
itself covers (used for trend/CAGR calculations later, not the calendar
year you're running this in).

---

## 3. Build the index (ingestion)

This is the expensive step — it runs real Mistral calls for structured
extraction, so it costs real API usage and takes real time (minutes, not
seconds, for a large report).

```bash
python3 -m ingest.build_index manifest.json data/financials.sqlite data/chroma_db
```

Watch the output: it prints per-report progress (parsing, batch
extraction, critic verification) and a summary at the end, including any
facts the critic couldn't verify against their source page — worth a
glance, not necessarily a blocker.

**If you're re-running this** (added a new report, or re-ingesting after a
fix): it's idempotent — safe to run again. It'll upsert facts and replace
that company/year's vector chunks rather than duplicating them. PDF
*parsing* is cached (`<pdf>.parsed_cache.json` next to each PDF); LLM
*extraction* is not, so a re-run still costs fresh API calls for
extraction specifically.

---

## 4. Verify ingestion actually worked

```bash
python3 -c "from stores.financials_db import list_companies; print(list_companies('data/financials.sqlite'))"
```
Expect your real companies back, not an empty list.

```bash
python3 -c "
import chromadb
client = chromadb.PersistentClient(path='data/chroma_db')
for c in client.list_collections():
    print(c.name, client.get_collection(c.name).count())
"
```
Expect a real chunk count — dozens to hundreds depending on report size.
A count of `1` here specifically means something ingested a stale test
chunk into this collection before your real data — delete `data/chroma_db`
and re-run Step 3 if you see this.

---

## 5. Run the smoke tests

Every module carries its own offline test suite (mocked LLM calls, no
network) — confirms the actual code is correct before adding Docker or a
live server into the mix:

```bash
python3 stores/financials_db.py
python3 stores/vectorstore.py
python3 agent/verifier.py
python3 agent/tools.py
python3 agent/orchestrator_langgraph.py
python3 ingest/extract.py
python3 ingest/build_index.py
```
Each should end with a line like `All X smoke tests passed`. Stop and fix
here if any fail — nothing downstream will work either, and it's much
easier to debug at this layer than inside a container later.

---

## 6. Run the backend locally (no Docker yet)

```bash
python3 -m uvicorn backend.main:app --reload --port 8000
```
Run this from the project root — `backend.main:app` is a dotted module
path that needs the root directory on `sys.path`.

**In a second terminal**, with that server still running:
```bash
curl http://localhost:8000/health
# expect: {"status":"ok"}

curl http://localhost:8000/companies
# expect: your real ingested companies

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"thread_id":"setup-check","question":"what companies do we have data for?"}'
# expect: a JSON body with "answer" and "trace" — your first real
# end-to-end question, so give it a few seconds
```

Also worth knowing: **http://localhost:8000/docs** is FastAPI's
auto-generated interactive API explorer — a browser-based way to send
`/chat` requests without hand-writing curl JSON.

Stop the server (`Ctrl+C`) before moving on.

---

## 7. Run the frontend locally (no Docker yet)

**Terminal 1** (backend, same as Step 6):
```bash
python3 -m uvicorn backend.main:app --port 8000
```

**Terminal 2**:
```bash
export BACKEND_URL=http://localhost:8000
export MISTRAL_API_KEY=your-key-here
streamlit run frontend/app.py
```
Open the printed local URL. Confirm the sidebar shows your real ingested
companies, then ask a real question and confirm you get a real answer with
a working "🔧 Plan & tool calls" expander.

Stop both (`Ctrl+C` in each) before moving on.

---

## 8. Build the Docker images

```bash
docker build -f backend/Dockerfile -t financial-analyst-backend .
docker build -f frontend/Dockerfile -t financial-analyst-frontend .
```
Both should end with something like `naming to docker.io/library/...`, no
red error text. If a build fails, the error names the exact line that broke.

> **Mac-specific gotcha, if you renamed/created either Dockerfile
> yourself rather than using the ones provided**: macOS's filesystem is
> case-*insensitive* by default, but Docker's build engine runs inside a
> Linux VM and is case-*sensitive*. A file saved as `DockerFile` looks
> identical to `Dockerfile` in Finder or `ls`, but `docker build -f
> backend/Dockerfile` will genuinely fail to find it. If you hit `open
> Dockerfile: no such file or directory`, this is almost always why —
> check with `ls backend/`.

---

## 9. Run each container individually

Isolates a Dockerfile problem from a docker-compose networking problem —
worth doing once even though Step 10 will run both together.

```bash
docker run --rm -p 8000:8000 -e MISTRAL_API_KEY=$MISTRAL_API_KEY financial-analyst-backend
```
In another terminal, repeat the `curl` checks from Step 6. `Ctrl+C` to stop.

```bash
docker run --rm -p 8501:8501 \
  --add-host=host.docker.internal:host-gateway \
  -e BACKEND_URL=http://host.docker.internal:8000 \
  -e MISTRAL_API_KEY=$MISTRAL_API_KEY \
  financial-analyst-frontend
```
`host.docker.internal` is Docker Desktop's DNS name for "the machine
running Docker" — needed here because this frontend container is running
standalone and needs to reach a backend on your actual Mac, not inside
docker-compose's shared network. Start the backend again first if you
stopped it.

---

## 10. Run the full stack with docker-compose

```bash
docker compose up --build
```
Expect both services to build and start, with backend's healthcheck
passing before frontend even starts (check with `docker compose ps` in
another terminal — backend should show `healthy`).

Open **http://localhost:8501** and run through a real conversation —
a company lookup, a trend/comparison question, and something that needs
`run_python` or `web_search` — to exercise each capability at least once
through the full real stack.

```bash
docker compose down
```
when done.

> **If port 8000 is already taken** by something else on your machine
> (common on Macs — a leftover process or container from earlier testing):
> `lsof -i :8000` shows what's holding it; `docker ps` shows if it's a
> stray container specifically. Stop whatever it is rather than switching
> ports, to keep every command in this guide accurate as written.

---

## 11. Push to GitHub

```bash
git init                                  # if not already a repo
git add .
git commit -m "Financial analyst agent: ingestion, agent, Docker, CI/CD"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```
The repo needs to be **public** for GitHub Container Registry's free tier
(used in Step 12) to work without extra configuration.

Double check `.gitignore` excludes `venv/` and any `__pycache__/` before
committing, if one doesn't already exist:
```bash
printf 'venv/\n__pycache__/\n*.pyc\n.env\n' >> .gitignore
```

---

## 12. GitHub Actions CI/CD

Nothing to configure — `.github/workflows/ci.yml` runs automatically once
it's in the repo and you push. It does two things, in order:

1. **`test` job** — installs both services' dependencies plus the
   ingestion-only ones, runs every module's smoke test from Step 5, and
   compiles both apps. Runs on every push *and* every pull request.
2. **`build` job** (only after `test` passes, only on push to `main`) —
   builds both Docker images and pushes them to `ghcr.io` (GitHub's free
   container registry — no extra account or token setup needed, it uses
   the workflow's own built-in `GITHUB_TOKEN`).

Go to your repo on GitHub → **Actions** tab. You should see a new run
appear within a few seconds of your push. **Expected**: `test` goes green
first, then `build` starts and also goes green.

Confirm the images actually published: your GitHub profile/org →
**Packages** tab should show two new packages
(`<repo-name>/backend`, `<repo-name>/frontend`).

**If `build` fails specifically at the image-push step** with a
permissions error: repo **Settings → Actions → General → Workflow
permissions**, and confirm "Read and write permissions" is selected —
some repos default to read-only, which blocks the workflow's token from
pushing to `ghcr.io` even when everything else about it is correct.

---

## You're done — what this covers, and what it doesn't

At this point: real data is ingested, the agent works locally and in
Docker, the full stack runs together via `docker compose`, and every push
to `main` automatically tests and publishes both images. That's the scope
of this document, matching what was asked for.

**Not covered here** (a deliberate boundary, not an oversight):
- Actually deploying the containers somewhere publicly reachable (Render,
  etc.) — see `DEPLOY.md`, though note it currently describes an earlier,
  single-process version of this app and needs a rewrite for the
  backend+frontend split before it's fully accurate again.
- Deeper troubleshooting for a specific failure at a specific phase — see
  `VERIFY.md`.
- Project architecture, the agent's capabilities, and the real bugs found
  along the way — see `README.md`.
