# dash.lost.plus

The Grimoire dashboard, as its own site.

It used to be a page inside the chat webui (`webui/src/routes/dashboard/`). That
made every dashboard change a change to a fork of upstream llama.cpp's UI, which
has to be merged forward every time upstream moves. Nothing about the dashboard
needed to live there — it shares no components with chat.

## What it is

A static site plus a thin nginx proxy. No database, no state of its own. Every
number comes from the Grimoire gateway's `/stats` endpoints:

| Endpoint | Purpose |
| --- | --- |
| `GET /stats/dashboard?window=…` | Everything drawn on the page |
| `PUT /stats/card-order` | Saves the card arrangement you drag into place |

That is the entire contract with Grimoire. The telemetry itself cannot move
here: it comes from `nvidia-smi`, `/sys/class/hwmon` and the powercap mount, all
of which only exist inside the gateway container on the grimoire host.

## Authentication

There is none, by design — you do not sign in.

nginx proxies `/stats/` through to the gateway and attaches the API key on the
way, from `GRIMOIRE_API_KEY` in the container's environment. So the page makes
ordinary same-origin requests, the browser never holds a credential, and there is
nothing to type. Neither the served HTML nor the JS bundle contains the key.

**The consequence is that the site itself is the only gate.** Anyone who can
reach it can read your token counts, your spend, and your GPU telemetry. There is
no second check behind it. Whatever fronts this hostname — a Cloudflare Access
policy, a private network, or nothing at all — is exactly the protection the data
has.

Because the requests are now same-origin, `GRIMOIRE_CORS_ORIGINS` is no longer
needed for the dashboard. The gateway still supports it for any other browser
app, but it can be left empty.

## Running it

```sh
npm install
npm run dev                 # http://localhost:5173
npm run build               # static output in dist/
```

In the fleet it runs as the `dash` service in the repo's `docker-compose.yml`,
nginx serving `dist/` on host port 9002 and proxying `/stats/` to
`http://grimoire:9001` over the shared compose network.

`npm run dev` serves only the static page; `/stats` requests will 404 without the
nginx in front, so use `docker compose up dash` to exercise the real thing.

## Polling

The page refetches no faster than a quarter of one bin, capped at 30 seconds, and
stops entirely while the tab is in the background. This matters more than it
looks: the gateway process that answers `/stats` is the same one that serves
chat. An earlier version polled every second regardless of window, which on the
"All" view meant a permanent queue of multi-second database scans in front of
every model request. The server caches payloads on the same schedule.
