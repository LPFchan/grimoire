#!/usr/bin/env python3
"""Load-test an OpenAI-compatible /v1/embeddings endpoint.

Mirrors bench_rerank_parallel.py for the embedder: fixed workload + offered
concurrency, reports throughput (texts/s, req/s) and latency percentiles.
Point --url at the gateway or directly at a llama-server.
"""

import argparse
import asyncio
import random
import statistics
import time

import httpx

_WORDS = ("aqueduct photosynthesis monetary migratory consensus fresco crispr tidal "
          "linguistics supersymmetry quantization throughput latency embedding reranker "
          "allocator gateway turboquant speculative").split()


def _make_input(n_words, rng):
    return " ".join(rng.choice(_WORDS) for _ in range(n_words))


async def _one(client, url, key, model, inputs):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    t0 = time.perf_counter()
    r = await client.post(url, headers=headers, json={"model": model, "input": inputs})
    r.raise_for_status()
    r.json()
    return time.perf_counter() - t0


async def _phase(client, args, n, rng):
    sem = asyncio.Semaphore(args.concurrency)
    lats = []

    async def w():
        async with sem:
            inputs = [_make_input(args.words, rng) for _ in range(args.batch)]
            if args.batch == 1:
                inputs = inputs[0]
            lats.append(await _one(client, args.url, args.key, args.model, inputs))

    t0 = time.perf_counter()
    await asyncio.gather(*(w() for _ in range(n)))
    return time.perf_counter() - t0, lats


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:9001/v1/embeddings")
    ap.add_argument("--key", default="")
    ap.add_argument("--model", default="eastself-embedder-0.6B")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--batch", type=int, default=1, help="inputs (texts) per request")
    ap.add_argument("--words", type=int, default=32, help="words per text")
    ap.add_argument("--requests", type=int, default=600)
    ap.add_argument("--warmup", type=int, default=32)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    rng = random.Random(7)
    async with httpx.AsyncClient(timeout=120.0, limits=httpx.Limits(max_connections=args.concurrency + 8)) as client:
        await _phase(client, args, args.warmup, rng)
        wall, lats = await _phase(client, args, args.requests, rng)

    texts = args.requests * args.batch
    ms = sorted(l * 1000 for l in lats)
    pct = lambda p: ms[min(len(ms) - 1, int(p / 100 * len(ms)))]
    print(f"{args.label:14} | conc={args.concurrency} batch={args.batch} reqs={args.requests} "
          f"| {texts/wall:8.1f} texts/s | {args.requests/wall:6.2f} req/s "
          f"| lat ms p50={pct(50):7.1f} p95={pct(95):7.1f} p99={pct(99):7.1f} | wall={wall:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
