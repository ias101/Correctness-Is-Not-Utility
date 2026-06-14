#!/usr/bin/env python
"""
Reusable cross-model reviewer call: gemini-3.1-pro-preview via REST.
Fallback reviewer for auto-review-loop after the codex/OpenAI account hit
quota. Gemini is a different model family from Claude AND from the prior GPT
reviewer, so this is a genuine cross-model adversarial review.

Usage:
  python gemini_review.py --prompt-file PROMPT.txt --out RESPONSE.md \
      [--model gemini-3.1-pro-preview] [--max-tokens 40000] [--temperature 0.25]

Reads GEMINI_API_KEY from the environment. Writes the response text to --out
and the raw JSON (incl. usage) to <out>.raw.json. Prints finishReason + usage.
Retries transient errors. Exits non-zero on hard failure so callers can detect it.
"""
import sys, os, json, argparse, time, urllib.request, urllib.error

ap = argparse.ArgumentParser()
ap.add_argument("--prompt-file", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="gemini-3.1-pro-preview")
ap.add_argument("--max-tokens", type=int, default=40000)
ap.add_argument("--temperature", type=float, default=0.25)
ap.add_argument("--system", default=None, help="optional system instruction file")
args = ap.parse_args()

key = os.environ.get("GEMINI_API_KEY")
if not key:
    print("ERROR: GEMINI_API_KEY not set", file=sys.stderr); sys.exit(2)

prompt = open(args.prompt_file, encoding="utf-8").read()
payload = {
    "contents": [{"parts": [{"text": prompt}]}],
    "generationConfig": {"maxOutputTokens": args.max_tokens,
                          "temperature": args.temperature},
}
if args.system:
    sys_txt = open(args.system, encoding="utf-8").read()
    payload["systemInstruction"] = {"parts": [{"text": sys_txt}]}

url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
       f"{args.model}:generateContent?key={key}")
data = json.dumps(payload).encode("utf-8")

last_err = None
for attempt in range(4):
    try:
        req = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            resp = json.loads(r.read().decode("utf-8"))
        cand = (resp.get("candidates") or [{}])[0]
        parts = cand.get("content", {}).get("parts", [])
        txt = "".join(p.get("text", "") for p in parts)
        finish = cand.get("finishReason")
        usage = resp.get("usageMetadata", {})
        if not txt.strip():
            last_err = f"empty text (finish={finish}, usage={usage})"
            # MAX_TOKENS with only thinking tokens -> retry with more budget
            time.sleep(3 + attempt * 3); continue
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(txt)
        with open(args.out + ".raw.json", "w", encoding="utf-8") as f:
            json.dump(resp, f, indent=2)
        print(f"OK model={args.model} finishReason={finish}")
        print(f"usage: prompt={usage.get('promptTokenCount')} "
              f"thoughts={usage.get('thoughtsTokenCount')} "
              f"out={usage.get('candidatesTokenCount')} "
              f"total={usage.get('totalTokenCount')}")
        print(f"wrote {len(txt)} chars -> {args.out}")
        sys.exit(0)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:500]
        last_err = f"HTTP {e.code}: {body}"
        if e.code in (429, 500, 502, 503, 504):
            time.sleep(5 + attempt * 5); continue
        break
    except Exception as e:
        last_err = repr(e); time.sleep(5 + attempt * 5); continue

print(f"ERROR after retries: {last_err}", file=sys.stderr)
sys.exit(1)
