# Property Count Checker

A FastAPI service with one endpoint. Given a company website URL, it returns
whether the company manages 10+ properties, the property count, and how
confident that count is — distinguishing individual property listings from
category/collection groupings.

It scrapes pages via [r.jina.ai](https://r.jina.ai) (which renders JS on its
own infrastructure and returns clean markdown) and classifies each page with
OpenAI `gpt-4o-mini`. No headless browser runs in this code, so it fits
Vercel's serverless model.

## API

### `GET /`
Health check.

### `POST /check-properties`

Request:
```json
{ "url": "https://example-vacation-rentals.com" }
```

Response:
```json
{
  "has_10_plus_properties": true,
  "property_count": 47,
  "count_type": "confirmed",
  "source": "summed across 3 listing pages",
  "pages_checked": ["https://.../", "https://.../properties"]
}
```

`count_type` is `"confirmed"` when every listing page produced a count and no
fetch failed; otherwise `"estimated"`. If no signal is found anywhere,
`property_count` is `null` and `source` is `"no property count signal found"`.

## Local development

```bash
cd property-checker
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then fill in OPENAI_API_KEY
uvicorn api.index:app --reload
```

Then:
```bash
curl -X POST http://localhost:8000/check-properties \
  -H "Content-Type: application/json" \
  -d '{"url": "https://some-property-manager.com"}'
```

Interactive docs at http://localhost:8000/docs.

## Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `OPENAI_API_KEY` | GPT-4o-mini classification calls | required |
| `JINA_API_KEY` | Optional — raises r.jina.ai rate limit above ~20 RPM | none |
| `MAX_PAGES_PER_SITE` | Caps total pages fetched per request | 8 |
| `MAX_CRAWL_DEPTH` | Caps how deep into categories it drills | 2 |
| `JINA_TIMEOUT_SECONDS` | Per-page fetch timeout | 15 |

## Deploy to Vercel

1. Push this repo to GitHub.
2. In Vercel: **New Project** → import the repo → it auto-detects FastAPI.
3. Add the environment variables above in the Vercel dashboard.
4. Deploy.
5. Test:
   ```bash
   curl -X POST https://your-app.vercel.app/check-properties \
     -H "Content-Type: application/json" \
     -d '{"url": "https://some-property-manager.com"}'
   ```

**Note:** Vercel's Hobby plan is personal/non-commercial. For production lead
qualification, move to Vercel Pro. If crawls approach the 60s limit, enable
Fluid Compute (free on Hobby) to raise the ceiling to 300s.
# scraping-loop
