# Nyaya AI — Indian Legal Assistant
# Setup and Run Guide

## What this system does
- Accepts any Indian law question or crime report
- Uses InLegalBERT (law-ai/InLegalBERT) for legal text understanding
- Connects to IndianKanoon LIVE API (3 crore+ legal documents)
- Uses verified BNS 2023 punishment table (no hallucinated numbers)
- Generates grounded answers via Claude with cite-or-abstain rules
- Returns: offences, imprisonment years, fine amount, bail status, precedents

## Files
- legal_engine.py   — Core: InLegalBERT + IndianKanoon + RAG + Claude
- app.py            — FastAPI server (REST API)
- index.html        — Chat UI (open in browser, works standalone in demo mode)
- requirements.txt  — Python dependencies

## Install

pip install -r requirements.txt

## Configure API Keys

export ANTHROPIC_API_KEY="sk-ant-..."
export INDIANKANOON_API_TOKEN="your_token_here"

# Get IndianKanoon API token:
# 1. Go to https://api.indiankanoon.org/signup/
# 2. Sign up — get free Rs 500 credit immediately
# 3. Non-commercial use: free Rs 10,000/month (requires verification)
# 4. Must display "Powered by IndianKanoon" logo in your UI

# Get Anthropic API key:
# https://console.anthropic.com

## Run the backend

uvicorn app:app --reload --port 8000

## Open the UI

Open index.html in your browser (works in demo mode without backend too)

## API endpoints

POST http://localhost:8000/chat
  Body: { "message": "What is punishment for murder?", "history": [] }

POST http://localhost:8000/analyse
  Body: { "report_text": "On 15 Sep 2024, accused X stabbed victim Y..." }

POST http://localhost:8000/analyse/upload
  Form: file=<your_fir.pdf or transcript.txt>

GET  http://localhost:8000/punishments/murder
GET  http://localhost:8000/punishments/theft

## Example crime report analysis

curl -X POST http://localhost:8000/analyse \
  -H "Content-Type: application/json" \
  -d '{
    "report_text": "On 20 September 2024, the accused Ramesh Kumar
    entered the house of the complainant and stabbed him multiple
    times with a knife. The victim died on the way to hospital.
    Witnesses present confirmed the act was premeditated."
  }'

Expected output:
- Applicable law: BNS 2023 (incident after July 2024)
- Offence: Murder [BNS Section 101]
- Punishment: Death OR Life Imprisonment + Fine
- Bail: Non-bailable
- Court: Sessions Court
- Relevant precedents from IndianKanoon

## How hallucination is controlled

1. PUNISHMENT NUMBERS never come from LLM memory
   → Hardcoded in BNS_PUNISHMENT_TABLE from official gazette
   → LLM only reads these verified numbers, never generates them

2. CASE CITATIONS come from IndianKanoon API (live, real)
   → Not from LLM training memory
   → If API returns nothing, system says "no precedents found"

3. CITE-OR-ABSTAIN system prompt
   → Claude must cite every legal claim with a section number
   → If it cannot cite, it must say "not in provided documents"

4. DATE-AWARE routing
   → Incident before July 2024 → IPC 1860
   → Incident after July 2024 → BNS 2023
   → Prevents wrong-law hallucinations

5. TEMPERATURE = 0
   → Deterministic output for legal facts
