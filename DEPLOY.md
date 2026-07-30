# Deploying Calorie Lens

The app is offline-first and the backend is optional, so the cheapest way to
run Calorie Lens is not to run anything. This document describes three tiers,
in increasing order of cost, and says plainly what each one buys.

Prices are approximate and were checked in mid-2026. Hosting prices move;
confirm the current figure before committing.

| Tier | What it adds | Realistic monthly cost |
| --- | --- | ---: |
| **0** | Nothing. The app on the device. | **$0** |
| **1** | Cross-device sync and accounts | **$0 – $5** |
| **2** | Remote AI language parsing | Tier 1 + a few cents to a few dollars |

Tier 0 is the recommended launch configuration.

---

## Tier 0 — no server at all ($0)

Build the app, install it, and stop. There is no server to deploy, no database
to back up, no API key to rotate, and no bill.

### What works

Everything except cross-device sync:

- Voice logging. Speech is transcribed by the operating system on the device
  (`expo-speech-recognition`, see `mobile/src/lib/speech.ts`). No audio is
  uploaded, no account is needed, and there is no per-use cost.
- Typed logging, in English, Hindi, or Hinglish.
- All food and exercise estimates. The USDA FoodData Central catalog and the
  2024 Compendium MET values are bundled in the app; the deterministic parser
  and engine run on the device.
- Clarifying questions when a portion, bowl size, duration, or intensity is
  missing — the app asks rather than guessing, with no model involved.
- The whole Train tab: 100-exercise library, routines, live logging, rest
  timers, PR detection, records, and charts.
- The Coach tab: the day's focus, your plan, and insights computed
  arithmetically from your own entries by `mobile/src/lib/insights.ts`.
- Apple Health / Apple Watch sync on iOS and Health Connect on Android.
- The encrypted local vault, in device-encrypted SQLite.
- Portable JSON backup export and restore, which is also how you move data to a
  new phone at this tier.

On the auth screen the user taps **Continue without an account** and never sees
a login again.

### What does not

- **Cross-device sync.** One phone holds the data. Moving to a new device means
  exporting a backup and restoring it.
- **Account recovery.** If the device is lost and there is no backup, the data
  is gone. The vault is encrypted with a device-held key; nobody can recover it
  for the user, by design.
- Signup, login, password recovery, and server-side backup — all of which are
  account features and therefore need Tier 1.

### Setup

Leave `EXPO_PUBLIC_API_URL` unset in the production build. `apiUrl()` in
`mobile/src/lib/api-client.ts` returns an empty string when it is unset outside
development, the auth screen shows "Account service is not configured in this
release", and the app runs entirely offline.

---

## Tier 1 — sync only

Run `api.py`, the FastAPI app in the repository root, with SQLite on a
persistent volume. That is the whole backend. It serves signup, login, recovery
codes, password change, account deletion, the encrypted vault sync, and backup
export/restore.

The mobile app pushes its data as JSON over HTTPS; the server encrypts it at
rest with AES-256-GCM under the master key before writing it to SQLite. This is
encryption at rest, not end-to-end encryption — whoever holds the master key
and the database can read vaults.

### Environment variables

These are the names actually read by the code. Everything has a default except
`CALORIE_LENS_MASTER_KEY`, which has a default you do not want in production.

Read by `account_store.py`:

| Variable | Default | Notes |
| --- | --- | --- |
| `CALORIE_LENS_DB_PATH` | `data/calorie_lens.db` | SQLite file. **Must** be on a persistent volume. |
| `CALORIE_LENS_KEY_PATH` | `<db directory>/calorie_lens.master.key` | Where a generated key is written when `CALORIE_LENS_MASTER_KEY` is unset. |
| `CALORIE_LENS_MASTER_KEY` | *(unset)* | URL-safe base64 that decodes to exactly **32 bytes**. Set this in production. |

Read by `api.py`:

| Variable | Default | Notes |
| --- | --- | --- |
| `CALORIE_LENS_ALLOWED_ORIGINS` | `*` | Comma-separated browser origins. Native apps are not subject to CORS, so `*` is fine for a mobile-only deployment. Setting anything else also turns on `allow_credentials`. |
| `GOOGLE_API_KEY` | *(unset)* | Tier 2 only. The API starts and serves accounts, sync, and backup without it. |
| `GEMINI_TEXT_MODEL` | `gemini-3.1-flash-lite` | Tier 2 only. |
| `CALORIE_LENS_AI_DAILY_LIMIT` | `8` | Tier 2 only. Total AI requests per account per UTC day. |
| `CALORIE_LENS_AI_TEXT_DAILY_LIMIT` | `4` | Tier 2 only. |
| `CALORIE_LENS_AI_AUDIO_DAILY_LIMIT` | `4` | Tier 2 only, and unused by the current app. |
| `CALORIE_LENS_AI_COACH_DAILY_LIMIT` | `2` | Tier 2 only, and unused by the current app. |

`api.py` calls `load_dotenv()`, so a `.env` file beside it works locally.
`.env.example` in the repository root lists the same names.

Generate a master key:

```bash
python -c "import base64,secrets; print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode())"
```

### The persistence requirement

This is the part that goes wrong on cheap hosts, so it is worth being blunt
about:

- **The SQLite file must live on a volume that survives restarts and
  redeploys.** Most platforms give containers an ephemeral filesystem. If
  `CALORIE_LENS_DB_PATH` points at ephemeral storage, every account and every
  synced vault disappears on the next deploy, restart, or crash.
- **Set `CALORIE_LENS_MASTER_KEY` explicitly.** If it is unset, `AccountStore`
  generates a key and writes it to `CALORIE_LENS_KEY_PATH`. On ephemeral
  storage that key is regenerated on every restart, and every previously stored
  vault becomes permanently undecryptable — while the accounts themselves still
  appear to exist. Losing or rotating this key destroys all vault data.
- SQLite runs in WAL mode, so the directory holding the database must be
  writable — the `-wal` and `-shm` files live beside it.
- One instance only. SQLite over a single volume does not survive being scaled
  horizontally.

Point `CALORIE_LENS_DB_PATH` at the mount, for example `/data/calorie_lens.db`
with a volume mounted at `/data`.

### A minimal image

There is no Dockerfile in the repository. This one is enough, and deliberately
skips Streamlit, Pillow, and the Google AI SDK, none of which the sync API
needs:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir \
      "fastapi" "uvicorn[standard]" "python-multipart" \
      "cryptography" "python-dotenv"
COPY api.py account_store.py ./
COPY calorie_engine ./calorie_engine
ENV CALORIE_LENS_DB_PATH=/data/calorie_lens.db
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

For Tier 2, add `google-genai` to the `pip install` line.

Sanity check before deploying anywhere:

```bash
docker build -t calorie-lens-api .
docker run --rm -p 8000:8000 -v calorie-lens-data:/data \
  -e CALORIE_LENS_MASTER_KEY=your_urlsafe_base64_32_byte_key \
  calorie-lens-api
curl localhost:8000/health
```

`/health` returns `{"ok": true, "accounts_enabled": true, ...}` and
`"ai_enabled": false` until a `GOOGLE_API_KEY` is set. That last field is a
useful confirmation that you are not paying for anything.

### Option A — cheap always-on: Fly.io (~$2–4/month)

Fly bills per second for a small shared-CPU machine plus a volume. A
`shared-cpu-1x` with 256 MB and a 1 GB volume lands around $2–4/month at
current list prices, and it does not sleep. Volumes are real block storage
attached to the machine, which is what SQLite needs.

```bash
# One-time
fly auth login
fly launch --no-deploy          # accept the app name, decline Postgres/Redis

# Persistent storage for SQLite, in the same region as the app
fly volumes create calorie_lens_data --size 1 --region <your-region>

# Secrets (these are not in fly.toml and not in git)
fly secrets set CALORIE_LENS_MASTER_KEY="$(python -c 'import base64,secrets; print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode())')"

fly deploy
```

Add the mount and the database path to `fly.toml`:

```toml
[[mounts]]
  source = "calorie_lens_data"
  destination = "/data"

[env]
  CALORIE_LENS_DB_PATH = "/data/calorie_lens.db"

[http_service]
  internal_port = 8000
  force_https = true
  auto_stop_machines = false     # see the note below
  auto_start_machines = true
  min_machines_running = 1
```

Fly terminates TLS for you, so the app is reachable at
`https://<app-name>.fly.dev` with no certificate work.

Setting `auto_stop_machines = true` and `min_machines_running = 0` cuts the
compute bill to near zero by stopping the machine when idle, at the price of a
cold start on the next request (see the caveat below). The volume is billed
either way. Do not scale past one machine.

Comparable always-on alternatives, if you would rather run a plain VPS and
manage systemd, nginx, and certbot yourself: Hetzner CX22 (~€3.80/month),
DigitalOcean or Vultr's smallest droplet (~$4–6/month). These are ordinary
servers with ordinary disks, so persistence is not something you have to think
about.

### Option B — free tier: Oracle Cloud Always Free ($0)

Oracle's Always Free tier includes Ampere ARM instances and persistent block
storage at no cost, indefinitely, with no sleeping. It is the only genuinely
free option in this list that satisfies the persistence requirement.

Rough shape:

1. Create an Always Free VM instance (Ampere A1, Ubuntu). The boot volume is
   persistent block storage — no extra volume is needed.
2. Open port 443 in both the VCN security list and the instance firewall.
   Oracle's default rules block almost everything; this is the step people miss.
3. On the box:

   ```bash
   sudo apt update && sudo apt install -y python3-pip nginx certbot python3-certbot-nginx
   pip install fastapi "uvicorn[standard]" python-multipart cryptography python-dotenv
   ```

4. Put the repository in `/opt/calorie-lens`, create
   `/opt/calorie-lens/.env` with `CALORIE_LENS_MASTER_KEY` and
   `CALORIE_LENS_DB_PATH=/opt/calorie-lens/data/calorie_lens.db`, and run
   `uvicorn api:app --host 127.0.0.1 --port 8000` under a systemd unit with
   `Restart=always`.
5. Terminate TLS with nginx plus certbot on a domain you own. The app must be
   served over HTTPS — iOS App Transport Security blocks plain HTTP in release
   builds.

The honest caveats: Always Free ARM capacity is frequently unavailable in
popular regions and can take repeated attempts to provision, idle instances can
be reclaimed if you never use them, and you own the patching, backups, and
certificate renewals. It is $0 in money and non-zero in attention.

**Free tiers that sleep.** Render, Koyeb, and similar free web-service tiers
idle a container out after roughly 15 minutes of inactivity and cold-start it on
the next request, which typically takes 30–60 seconds. For this app that is
more tolerable than usual: sync happens in the background, and every write is
already saved locally first, so a slow sync is invisible unless the user is
watching the sync indicator. **But check the disk story before choosing one.**
Render's free instance type has no persistent disk at all — the filesystem is
rebuilt on every deploy and every restart, which means every account and every
synced vault is destroyed. A free tier without a persistent volume is fine for
a demo and unusable for real data.

### Point the app at it

Set `EXPO_PUBLIC_API_URL` in `mobile/.env` (see `mobile/.env.example`) before
building:

```bash
EXPO_PUBLIC_API_URL=https://your-app.fly.dev
```

`EXPO_PUBLIC_*` values are inlined at build time, so a production build must be
rebuilt to change the URL. Use HTTPS: iOS blocks cleartext HTTP in release
builds. In development the app discovers the API from the Metro bundler address
automatically, so you usually do not need this locally.

Verify from the phone's network:

```bash
curl https://your-app.fly.dev/health
```

The auth screen shows a live reachability indicator with the exact URL it is
trying, which is the fastest way to confirm the build picked up the value.

---

## Tier 2 — optional AI parsing

Remote AI language parsing is off by default. Turning it on requires **both** a
build-time flag and a server-side key, and it requires Tier 1 — the
`/v1/parse-command` endpoint is authenticated, so the user must be signed in to
an account for the app to call it at all.

```bash
# mobile/.env — build-time
EXPO_PUBLIC_ENABLE_AI_PARSING=1

# server-side
GOOGLE_API_KEY=your_google_ai_key
```

### What it buys

Only one thing: foods the bundled catalog does not recognize. With the flag
off, `parseFitnessCommand` in `mobile/src/lib/nutrition.ts` returns a
clarifying question — asking for label calories, or for the main parts with
amounts. With it on, and only when the local parser has already failed on a
food or on a fragment of a multi-part update, the app asks the server to read
the language for it. The local result is still the fallback if the request
fails, times out, or the user is signed out.

It does not change any number. The model returns facts — food, amount, unit,
activity, duration, intensity — and the same deterministic engine produces
every calorie. The model is prohibited from generating calories or macros.

### What it costs

A parse request is small: one sentence in, a short JSON object out. On a
flash-lite class model that is a fraction of a cent per call. Several things
keep the total bounded:

- The remote parser is only consulted after the local one fails, which for
  everyday logging is rare.
- Results are encrypted and cached server-side for 30 days. A repeated
  identical request costs nothing.
- Per-account, per-UTC-day ceilings are enforced before the model is called:
  `CALORIE_LENS_AI_TEXT_DAILY_LIMIT` (4 by default) and
  `CALORIE_LENS_AI_DAILY_LIMIT` (8 by default). Lower them if you want a harder
  ceiling.

With the shipped caps, the worst case is 4 text parses per account per day.
Ten active users who all hit the cap every day is 1,200 small requests a month
— single-digit dollars at flash-lite pricing, and in practice far less because
most of it is cached or never reaches the model. Google AI Studio also has a
free tier with rate limits that may cover a small user base outright. Check
current pricing before relying on any of these figures.

The `/v1/parse-command/audio` and `/v1/coach` endpoints still exist on the
server, but the current app calls neither: voice is transcribed on the device
and the Coach tab is computed on the device. If nothing else uses them, they
cost nothing.

---

## What actually costs money

| Item | Cost | Needed for |
| --- | ---: | --- |
| The app itself | $0 | — |
| Voice transcription | $0 | Nothing. It is the OS's recogniser, on the device. |
| Food and exercise estimates | $0 | Nothing. Bundled catalog, on-device engine. |
| Coach tab insights | $0 | Nothing. Arithmetic on the device. |
| Server compute, always-on | ~$2–5/month | Tier 1 sync |
| Server compute, free tier | $0 | Tier 1 sync, with the sleep and disk caveats above |
| Persistent volume (1 GB) | ~$0.15–1/month | Tier 1 sync; often included in a VPS |
| Domain name | ~$10–15/year | Optional; platform subdomains are free |
| TLS certificate | $0 | Let's Encrypt, or handled by the platform |
| Gemini API calls | fractions of a cent per call, capped | Tier 2 only |
| **Apple Developer Program** | **$99/year** | Any iOS distribution — TestFlight or the App Store |
| Google Play Developer | $25 one-time | Android distribution through Play |

The Apple Developer Program fee is $99/year and is unavoidable if you want the
app on an iPhone that is not your own development device. It is charged whether
or not you run a server, and at Tier 0 it is the only recurring cost of the
entire project. A free Apple ID can sideload a build onto your own device, but
the provisioning profile expires after seven days.

---

## Recommendation

Launch at Tier 0. It costs $99/year for Apple and nothing else, and the only
feature the user gives up is cross-device sync — which the JSON backup export
covers well enough for a single-device user.

Add Tier 1 when someone actually asks for sync, and put it on a persistent
volume from the first deploy rather than migrating to one later. Leave Tier 2
off unless unrecognized foods turn out to be a real, measured complaint;
the clarifying question is a reasonable answer to that problem and costs
nothing.
