#!/usr/bin/env python3
"""Load-test the grimoire reranker to find the --parallel throughput plateau.

Drives /v1/rerank at a fixed workload + offered concurrency and reports
throughput (pairs/s) and latency percentiles. Run once per --parallel setting
(the caller restarts the gateway between settings); this script only measures
whatever reranker is currently live.
"""

import argparse
import asyncio
import random
import statistics
import time

import httpx

# A pool of ~60-90 token documents so each scored pair does real work.
_TOPICS = [
    "the history of the Roman aqueduct system and its engineering",
    "photosynthesis in C4 plants under drought stress conditions",
    "monetary policy transmission through the bank lending channel",
    "the migratory patterns of arctic terns across hemispheres",
    "distributed consensus algorithms and the CAP theorem tradeoffs",
    "renaissance fresco pigment chemistry and lime plaster curing",
    "CRISPR-Cas9 off-target effects in mammalian cell lines",
    "tidal energy capture using oscillating water column turbines",
    "the linguistics of tone sandhi in Mandarin Chinese dialects",
    "supersymmetry constraints from the Large Hadron Collider runs",
]


def _make_docs(n: int, rng: random.Random) -> list[str]:
    docs = []
    for _ in range(n):
        topic = rng.choice(_TOPICS)
        filler = " ".join(w for _ in range(4) for w in rng.choice(_TOPICS).split()[:6])
        docs.append(f"A detailed discussion of {topic}. Further notes: {filler}.")
    return docs


async def _one_request(client, url, key, model, query, docs):
    t0 = time.perf_counter()
    r = await client.post(
        url,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": model, "query": query, "documents": docs},
    )
    r.raise_for_status()
    r.json()
    return time.perf_counter() - t0


async def _run_phase(client, args, n_requests, rng):
    sem = asyncio.Semaphore(args.concurrency)
    query = "Which document best explains the underlying mechanism in detail?"
    latencies = []

    async def worker():
        async with sem:
            docs = _make_docs(args.docs, rng)
            lat = await _one_request(client, args.url, args.key, args.model, query, docs)
            latencies.append(lat)

    t0 = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(n_requests)))
    wall = time.perf_counter() - t0
    return wall, latencies


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:9001/v1/rerank")
    ap.add_argument("--key", required=True)
    ap.add_argument("--model", default="eastself-reranker-0.6B")
    ap.add_argument("--concurrency", type=int, default=32, help="max in-flight requests")
    ap.add_argument("--docs", type=int, default=16, help="documents (pairs) per request")
    ap.add_argument("--requests", type=int, default=192, help="timed requests")
    ap.add_argument("--warmup", type=int, default=24, help="warmup requests (untimed)")
    ap.add_argument("--label", default="", help="tag printed with results (e.g. parallel=4)")
    args = ap.parse_args()

    rng = random.Random(1234)
    async with httpx.AsyncClient(timeout=120.0, limits=httpx.Limits(max_connections=args.concurrency + 8)) as client:
        await _run_phase(client, args, args.warmup, rng)          # warmup
        wall, lats = await _run_phase(client, args, args.requests, rng)

    pairs = args.requests * args.docs
    lats_ms = sorted(l * 1000 for l in lats)

    def pct(p):
        return lats_ms[min(len(lats_ms) - 1, int(p / 100 * len(lats_ms)))]

    print(f"{args.label:14} | concurrency={args.concurrency} docs/req={args.docs} reqs={args.requests} "
          f"| {pairs/wall:8.1f} pairs/s | {args.requests/wall:6.2f} req/s "
          f"| lat ms p50={pct(50):7.1f} p95={pct(95):7.1f} p99={pct(99):7.1f} "
          f"mean={statistics.mean(lats_ms):7.1f} | wall={wall:5.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
