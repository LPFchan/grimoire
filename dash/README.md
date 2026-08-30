# dash.lost.plus

The Grimoire dashboard, as its own site.

It used to be a page inside the chat webui (`webui/src/routes/dashboard/`). That
made every dashboard change a change to a fork of upstream llama.cpp's UI, which
has to be merged forward every time upstream moves. Nothing about the dashboard
needed to live there — it shares no components with chat.

## What it is

A static site. No server, no database, no state of its own. Every number comes
from the Grimoire gateway's `/stats` endpoints, fetched from the browser:

| Endpoint | Purpose |
| --- | --- |
| `GET /stats/dashboard?window=…` | Everything drawn on the page |
| `PUT /stats/card-order` | Saves the card arrangement you drag into place |

That is the entire contract with Grimoire. The telemetry itself cannot move
here: it comes from `nvidia-smi`, `/sys/class/hwmon` and the powercap mount, all
of which only exist inside the gateway container on the grimoire host.

## Signing in

Because the site is on a different host from the API, the gateway's `gw_session`
cookie does not apply — it is `SameSite=Lax` and scoped to the chat host. The
sign-in screen takes your Grimoire API key, keeps it in `localStorage`, and sends
it as a bearer token.

Two things have to line up for that to work:

1. The gateway must name this origin in `GRIMOIRE_CORS_ORIGINS`, or the browser
   refuses to hand over the response. Set it to `https://dash.lost.plus`; never
   to `*`, since these endpoints serve private usage and cost figures.
2. `VITE_API_BASE` must point at the gateway. It is baked in at build time
   (default `https://chat.lost.plus`) and can be overridden per browser from the
   "Using a different gateway?" field on the sign-in screen.

## Running it

```sh
npm install
npm run dev                 # http://localhost:5173
npm run build               # static output in dist/
```

In the fleet it runs as the `dash` service in the repo's `docker-compose.yml`,
nginx serving `dist/` on host port 9002.

## Polling

The page refetches no faster than a quarter of one bin, capped at 30 seconds, and
stops entirely while the tab is in the background. This matters more than it
looks: the gateway process that answers `/stats` is the same one that serves
chat. An earlier version polled every second regardless of window, which on the
"All" view meant a permanent queue of multi-second database scans in front of
every model request. The server caches payloads on the same schedule.
