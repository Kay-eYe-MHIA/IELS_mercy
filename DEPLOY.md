# Deploying the mobile app (backend on Render + frontend on Vercel)

Two pieces, two free hosts:

- **`backend/`** — FastAPI wrapper around `stock_predictor`, runs the real
  XGBoost walk-forward analysis. Needs a normal always-on host, not Vercel
  serverless (xgboost + pandas + scikit-learn are ~400MB, well past
  Vercel's ~250MB serverless function limit).
- **`frontend/`** — a single static `index.html`, no build step. This is
  what Vercel hosts, and what you open on your phone.

## 1. Backend on Render (free tier)

1. Go to [render.com](https://render.com) and sign in with GitHub.
2. **New → Blueprint**, pick this repo. Render reads `render.yaml` at the
   repo root and configures the service automatically (build command,
   start command, health check).
3. Deploy. First build installs xgboost/pandas/etc, so it takes a few
   minutes.
4. Once live, note the URL Render gives you, e.g.
   `https://stock-predictor-api.onrender.com`.
5. Sanity check: open `https://<your-app>.onrender.com/api/health` in a
   browser — should return `{"status":"ok"}`.

**Free tier note:** the service spins down after ~15 minutes of no
traffic. The first request after that can take 30–60s to wake up — the
frontend's status message already accounts for this.

## 2. Frontend on Vercel

1. Go to [vercel.com](https://vercel.com) and sign in with GitHub.
2. **Add New → Project**, import this repo.
3. In the project's configure screen, set:
   - **Root Directory**: `frontend`
   - **Framework Preset**: `Other` (it's plain HTML/CSS/JS, no build step)
4. Deploy. Vercel gives you a URL like `https://your-app.vercel.app`.

## 3. Connect them

1. Open your Vercel URL on your phone (add it to your home screen for an
   app-like feel: browser share menu → "Add to Home Screen").
2. In the **Backend** card at the top, paste your Render URL (e.g.
   `https://stock-predictor-api.onrender.com`). It's saved in the browser
   so you only enter it once.
3. Pick a symbol and data source and tap **Run analysis**.
   - `Demo (synthetic)` needs no network on the backend side — good first
     test that the two pieces are wired up correctly.
   - `Yahoo Finance` needs no API key; Bursa Malaysia tickers use `.KL`
     (e.g. `4456.KL` for DNEX).
   - `EODHD` needs an API key (free tier at eodhd.com/register); Bursa
     tickers use `.KLSE` (e.g. `4456.KLSE`).

## Updating after future changes

Both Render and Vercel auto-redeploy on push to this branch once
connected — just `git push` and both sides pick it up within a minute or
two.
