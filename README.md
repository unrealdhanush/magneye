# Magneye

See inside anything you can photograph. Give Magneye one photo and a short prompt. Meta's models work out what the object is, what's inside it and how it works. The result is an interactive 3D exploded view: you can play how the object works, or describe a fault and watch the likely culprit light up red.

Built for the Meta Global AI Developer Hackathon (Oct 26 – Nov 4, 2026).

## How it works

| Stage | Model | What it does |
|---|---|---|
| Identify & infer internals | **Muse Wonder** (`muse-wonder-1.0`) | Looks at the photo and returns structured anatomy as JSON: the object, 12–28 parts with sizes and positions in millimetres, connections, how-it-works scenarios and failure modes |
| Ground in the photo | **SAM 3.1** (`sam-3.1`) | Segments the whole object and each part Wonder says is visible. Visible parts snap to where SAM actually found them, and the object mask becomes the 3D shell's silhouette |
| Reconstruct the shell | **SAM 3D Objects** (open weights, on Modal) | Optional. Turns the photo and object mask into a 3D mesh that replaces the extruded silhouette |
| Answer prompts | **Muse Wonder** | Answers each prompt using only the anatomy above. It picks a scenario to play or a failure to diagnose, and can write a new one using the existing parts |
| Voice | **Muse Voice Transcribe** | Push-to-talk questions |

The viewer (React + three.js) starts with the photo extruded into 3D. Pulling it apart reveals the inferred internals. Scenarios animate the chain of parts, with a signal pulse running between them. Failures lift off the parts in the way, outermost first, then pulse the culprit.

Hidden parts are inferred, not measured. The UI marks each part as either "seen in photo · SAM 3.1" or "hidden · inferred", and lists Wonder's caveats.

## Run it

```bash
cp .env.example .env            # add MODEL_API_KEY (Meta Model API)
uv sync
(cd web && npm install && npm run build)
uv run magneye                  # http://localhost:8787
```

For frontend development, run `npm run dev` in `web/`. It proxies `/api` to port 8787.

The live camera and microphone need a secure context, so use `localhost` or HTTPS.

## SAM 3D Objects (optional)

`gpu/sam3d.py` deploys SAM 3D Objects on a Modal A100. It needs gated Hugging Face access to `facebook/sam-3d-objects`. Deploy it, then set `SAM3D_URL`. Without it, Magneye uses the shell extruded from SAM 3.1's mask. **Status: written, not yet run.**

## Tests

```bash
uv run pytest
uv run python tests/fixtures/make_ui_session.py   # stand-in session at /#/s/uifixture0000001 for UI work without API calls
```

## Layout

```
src/magneye/   FastAPI app, Meta Model API client, anatomy prompts/schemas, SAM grounding
web/           React + three.js viewer
gpu/           SAM 3D Objects endpoint for Modal
tests/         Backend tests and a UI fixture session
```
