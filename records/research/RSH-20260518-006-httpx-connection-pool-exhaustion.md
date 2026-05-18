# Host httpx Connection Pool Exhaustion After ~90 Requests (2026-05-18)

Opened: 2026-05-18 18-40-00 KST
Recorded by agent: opencode

## Observation

During the 100-iteration PFlash soak test, running the `httpx` client from the host machine caused the test to hang at ~92 iterations while the gateway remained healthy. The same test run from inside the Docker container completed all 100 iterations without issue.

## Root Cause

The host-side `httpx` client maintains a connection pool of HTTP keep-alive connections to the gateway (port 9001). After ~90 requests, the connection pool state degrades:
- Stale connections enter CLOSE_WAIT state (server closed, client hasn't closed)
- The connection pool tries to reuse these half-closed connections
- New requests block waiting for a usable connection
- Eventually httpx times out

This is NOT a gateway bug. The gateway processes all requests fine. It's an `httpx`/`httpcore` connection pool issue on the host.

## Diagnosis

Checking `/proc/net/tcp` during the hang showed `CLOSE_WAIT` state on the gateway's side (connection to llama-server). But there was NO connection from the gateway to llama-server at all when the hang occurred — the gateway was still in the PFlash compression step, not proxying to llama-server yet. The actual block was in the Python thread pool reading from the daemon pipe (a separate, now-fixed issue).

The host httpx hang was a separate symptom from the stale-thread deadlock (DEC-20260518-001).

## Lessons

- When stress-testing a gateway from a host machine, prefer running the test *inside* the container to avoid client-side connection pool issues
- Container-to-container requests use the Docker network, which doesn't have the same keep-alive connection degradation
- For long soak tests (>100 iterations), consider using `docker exec` to run the test client in-container

## Related

- `soak.py`: the soak script that exposed this
- DEC-20260518-001: the actual deadlock that the host hang was confused with
