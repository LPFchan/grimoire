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

## Chart scales

Sparklines use absolute axes, supplied per series by the gateway as a
`scale: {min, max}` alongside the data. A `null` max means "pin the floor, let
the top follow the data".

Nothing is hard-coded for one machine. Ceilings come from the hardware wherever
it will state one — `nvidia-smi` for GPU power limit, VRAM and max operating
temperature, `/proc/meminfo` for installed RAM, the cgroup for this container's
own memory ceiling. Where no limit is discoverable (CPU temperature and package
power, fan speed) the ceiling is derived from the highest value this host has
actually been observed reaching, rounded up to a legible step, with a minimum
span so an idle machine does not get an absurdly tight axis. See
`src/grimoire/limits.py`.

Two things keep a floating ceiling on purpose: GPU throughput, which depends
entirely on which model is loaded, and the token and cost series, which
accumulate across the window. Both still get a floor of zero — without it, an
idle window has no range at all and used to be drawn as a filled block at half
height, which reads as data rather than as nothing.

Temperature axes start at 20 °C rather than zero, since nothing here ever
approaches freezing and a zero-based axis would waste half of every card.

## Polling

The page refetches no faster than a quarter of one bin, capped at 30 seconds, and
stops entirely while the tab is in the background. This matters more than it
looks: the gateway process that answers `/stats` is the same one that serves
chat. An earlier version polled every second regardless of window, which on the
"All" view meant a permanent queue of multi-second database scans in front of
every model request. The server caches payloads on the same schedule.

Historical chart ceilings are maintained alongside incoming telemetry. A page
load reads those small records directly; it never scans raw history to rediscover
the same all-time peaks. The browser also issues only one initial refresh, then
refreshes again when a hidden tab becomes visible.
