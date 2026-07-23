# Property Count Checker

A FastAPI service with one endpoint. Given a company website URL, it returns
whether the company manages 10+ properties, the property count, and how
confident that count is — distinguishing individual property listings from
category/collection groupings.

It scrapes pages via [r.jina.ai](https://r.jina.ai) (which renders JS on its
own infrastructure and returns clean markdown) and uses OpenAI to classify
pages and find where the listings live. No headless browser runs in this code,
so it fits Vercel's serverless model.

## How the count is derived

Counting does **not** rely on the LLM eyeballing a page (it double-counts
grid/list renders and miscounts image links). Instead the count is
**deterministic**: every property links to its own detail page, so the service
extracts each unique detail-page link and counts the distinct ones. This is
stable across the common site permutations:

- path slugs — `/stowaway`, `/white-house-cottage`
- query-id links — `/details.aspx?PropertyID=391025`
- nested detail paths — `/rentals/beach-villa`
- properties shown as cards with a name + photo linking to a detail page

Across category pages the distinct links are **unioned** (so a site split into
"Beachfront / Downtown" sums correctly), and across pagination they are
**deduped**. The LLM still reads any explicitly stated total ("47 properties")
or pagination total ("Showing 12 of 84") and those refine the deterministic
count when the site advertises more than could be scraped.

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
  "property_count": 52,
  "count_type": "confirmed",
  "source": "count from https://.../properties (basis: counted_items)",
  "pages_checked": ["https://.../", "https://.../properties"],
  "breakdown": [
    { "url": "https://.../properties", "count": 52, "count_basis": "counted_items" }
  ]
}
```

- `count_type` — `"confirmed"` when the number is either explicitly stated by
  the site or came from a complete deterministic card count (no fetch failures,
  not undercut by pagination); otherwise `"estimated"`.
- `breakdown` — per-listing-page contribution, so the total is auditable.
- `count_basis` — how a number was derived: `stated_total`, `pagination`,
  `counted_items` (distinct detail links / cards), or `unknown`.
- If no signal is found anywhere, `property_count` is `null` and `source` is
  `"no property count signal found"`.

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
| `CLASSIFIER_MODEL` | LLM used to classify pages; upgrade (e.g. `gpt-4o`) for tricky sites | gpt-4o-mini |
| `MAX_CONTENT_CHARS` | Page markdown sent to the model; larger avoids truncated undercounts | 48000 |

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
