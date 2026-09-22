#!/usr/bin/env python3
"""NeuroScope trace-dump client — capture JSON traces programmatically.

How it works (DOC-06 fix):
    1. Opens a WebSocket to ``WS /ws/generate`` and sends a generation request
       with ``record_trace: true``.  The server accumulates all token frames.
    2. After the stream ends, fetches the saved trace via ``GET /trace``.
    3. Writes the trace JSON to ``--output`` (or stdout).

Usage:
    python scripts/trace_client.py --prompt "The capital of France is" --output trace.json

    # Use a custom server address
    python scripts/trace_client.py --url http://localhost:8000 --prompt "Once upon a" --output trace.json

    # Record a pair for CI diffing
    python scripts/trace_client.py --prompt "Hello world" --output baseline.json
    python scripts/trace_client.py --prompt "Hello world" --output head.json
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import urllib.request
import urllib.parse

try:
    import websocket  # websocket-client package
    _WS_AVAILABLE = True
except ImportError:
    _WS_AVAILABLE = False


def _http_base_to_ws(base_url: str) -> str:
    """Convert http(s)://host to ws(s)://host."""
    return base_url.rstrip("/").replace("https://", "wss://").replace("http://", "ws://")


def capture_trace(
    base_url: str,
    prompt: str,
    max_new_tokens: int = 10,
    top_k: int = 10,
    timeout: int = 60,
) -> dict:
    """Stream a generation via ``WS /ws/generate`` then fetch the trace via ``GET /trace``.

    The ``record_trace`` flag tells the backend to accumulate all token frames
    in memory; after the WebSocket stream closes ``GET /trace`` retrieves them
    as a single JSON blob.

    Requires the ``websocket-client`` package (``pip install websocket-client``).
    """
    if not _WS_AVAILABLE:
        print(
            "Error: websocket-client is not installed.\n"
            "Install it with:  pip install websocket-client",
            file=sys.stderr,
        )
        sys.exit(1)

    ws_url = _http_base_to_ws(base_url) + "/ws/generate"
    payload = json.dumps({
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "top_k": top_k,
        "record_trace": True,   # was "trace": True — correct key is record_trace
    })

    done_event = threading.Event()
    ws_error: list[str] = []

    def on_open(ws):
        ws.send(payload)

    def on_message(ws, message):
        frame = json.loads(message)
        if frame.get("type") in ("done", "error"):
            if frame.get("type") == "error":
                ws_error.append(frame.get("message", "unknown error"))
            ws.close()

    def on_error(ws, error):
        ws_error.append(str(error))

    def on_close(ws, close_status_code, close_msg):
        done_event.set()

    ws = websocket.WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    thread = threading.Thread(target=ws.run_forever)
    thread.daemon = True
    thread.start()

    done_event.wait(timeout=timeout)
    if not done_event.is_set():
        print("Timed out waiting for generation to complete.", file=sys.stderr)
        sys.exit(1)
    if ws_error:
        print(f"WebSocket error: {ws_error[0]}", file=sys.stderr)
        sys.exit(1)

    # Fetch the trace that the server recorded during the generation.
    trace_url = base_url.rstrip("/") + "/trace"
    req = urllib.request.Request(trace_url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"GET /trace failed — HTTP {e.code}: {e.read().decode()}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error fetching trace: {e}", file=sys.stderr)
        sys.exit(1)


def trace_diff(baseline_path: str, head_path: str, tolerance: float = 0.05) -> list[str]:
    """Compare two trace files and report structural differences.

    Trace frames have the shape produced by ``TraceRecorder.build()``:
        frames[].type        — always "token"
        frames[].token       — the emitted token string
        frames[].token_id    — integer token id
        frames[].topk[]      — list of {id, token, prob} dicts
        frames[].logit_stats — optional per-layer float stats
    """
    with open(baseline_path) as f:
        base = json.load(f)
    with open(head_path) as f:
        head = json.load(f)

    diffs: list[str] = []

    frames_b = base.get("frames", [])
    frames_h = head.get("frames", [])

    if len(frames_b) != len(frames_h):
        diffs.append(f"Frame count: baseline={len(frames_b)} head={len(frames_h)}")

    for i, (fb, fh) in enumerate(zip(frames_b, frames_h)):
        # Compare chosen token id.
        tid_b = fb.get("token_id")
        tid_h = fh.get("token_id")
        if tid_b != tid_h:
            diffs.append(
                f"Token {i}: baseline id={tid_b} ({fb.get('token')!r}) "
                f"head id={tid_h} ({fh.get('token')!r})"
            )

        # Compare top-K probability distributions.
        tk_b = {c["id"]: c["prob"] for c in fb.get("topk", [])}
        tk_h = {c["id"]: c["prob"] for c in fh.get("topk", [])}
        for tok_id in set(tk_b) | set(tk_h):
            pb = tk_b.get(tok_id, 0.0)
            ph = tk_h.get(tok_id, 0.0)
            if abs(pb - ph) > tolerance:
                diffs.append(
                    f"Token {i}, top-k id {tok_id}: "
                    f"baseline={pb:.3f} head={ph:.3f}"
                )

        # Compare optional per-layer logit stats.
        ls_b = fb.get("logit_stats", [])
        ls_h = fh.get("logit_stats", [])
        for li, (lb, lh) in enumerate(zip(ls_b, ls_h)):
            if abs(lb - lh) > tolerance:
                diffs.append(
                    f"logit_stats layer {li} @ token {i}: "
                    f"baseline={lb:.4f} head={lh:.4f}"
                )

    return diffs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TokenPrint trace client — capture a generation trace over WebSocket",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--url", default="http://localhost:8000", help="Backend server URL")
    parser.add_argument("--prompt", default="The capital of France is", help="Prompt text")
    parser.add_argument("--max-tokens", type=int, default=10, help="Max generated tokens")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K candidates")
    parser.add_argument("--output", default=None, help="Save trace JSON to this file")
    parser.add_argument("--timeout", type=int, default=60, help="Request timeout in seconds")
    args = parser.parse_args()

    print(f"Capturing trace from {args.url} ...", file=sys.stderr)
    trace = capture_trace(args.url, args.prompt, args.max_tokens, args.top_k, args.timeout)

    if args.output:
        with open(args.output, "w") as fh:
            json.dump(trace, fh, indent=2)
        print(f"Trace saved to {args.output}", file=sys.stderr)
    else:
        print(json.dumps(trace, indent=2))


if __name__ == "__main__":
    main()
