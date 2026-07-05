# Deploying VolCast publicly (free tier)

Two pieces: the **API** (Docker image → Render) and the **dashboard**
(Next.js → Vercel). Both deploy straight from the GitHub repo — the serving
seed data (model artifact, feature store, OHLCV history) is committed for
exactly this reason.

## 0. Push to GitHub

```bash
git add -A && git commit -m "deploy prep"
git push origin main
```

## 1. API on Render

1. [render.com](https://render.com) → **New → Blueprint** → select this repo.
   Render reads [render.yaml](render.yaml) and builds the Dockerfile.
2. Wait for the first deploy (image build ~5–10 min; free instances also
   cold-start after idle, so the first request can take ~1 min).
3. Note your URL, e.g. `https://volcast-api.onrender.com`, and check
   `https://volcast-api.onrender.com/health` returns `"status": "ok"`.

> Free tier notes: 512 MB RAM is enough for the INT8 model + feature store,
> but leave `SERVE_QUANTIZE=1`. There is no Redis — the API detects that and
> serves compute-on-request (the `/health` field shows `"redis": "unavailable"`).

## 2. Dashboard on Vercel

1. [vercel.com](https://vercel.com) → **Add New → Project** → import the repo.
2. Set **Root Directory** to `web/`.
3. Add the environment variable
   `NEXT_PUBLIC_API_URL = https://volcast-api.onrender.com` (your Render URL).
4. Deploy → you get e.g. `https://volcast.vercel.app`.

## 3. Point CORS at the dashboard

Back on Render: **Environment → UI_ORIGINS** →
`https://volcast.vercel.app` (your actual Vercel URL) → save (auto-redeploys).

Done. Put the Vercel link at the top of the README.

## Keeping the deployed model fresh (optional)

The committed seed data is a snapshot. To refresh it:

```bash
python -m scripts.run_ingest && python -m scripts.build_dataset
python -m scripts.export_model && python -m scripts.run_monitoring
git add data && git commit -m "data: refresh serving snapshot" && git push
```

Both Render and Vercel redeploy automatically on push.
