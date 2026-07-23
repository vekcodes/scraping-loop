"""FastAPI app — Vercel's entrypoint.

Vercel detects the `app` ASGI instance in this file and deploys it as a
serverless function automatically.
"""

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from app.crawler import check_properties
from app.schemas import CheckRequest, CheckResponse

load_dotenv()

app = FastAPI(
    title="Property Count Checker",
    description="Given a company website URL, estimates how many properties "
    "the company manages and whether it manages 10 or more.",
    version="1.0.0",
)


@app.get("/")
async def root():
    return {"status": "ok", "endpoint": "POST /check-properties"}


@app.post("/check-properties", response_model=CheckResponse)
async def check_properties_endpoint(req: CheckRequest) -> CheckResponse:
    url = (req.url or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="url is required")
    try:
        return await check_properties(url)
    except Exception as exc:  # pragma: no cover - defensive top-level guard
        raise HTTPException(status_code=502, detail=f"check failed: {exc}")
