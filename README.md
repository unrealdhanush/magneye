# Magneye

See inside anything you can photograph. Give Magneye one photo and a short prompt. Meta's models work out what the object is, what's inside it and how it works. The result is an interactive 3D exploded view: you can play how the object works, or describe a fault and watch the likely culprit light up red.

Built for the Meta Global AI Developer Hackathon (Oct 26 – Nov 4, 2026).

## How it works

Two agents, both Muse Wonder (`muse-wonder-1.0`) with tools, stream every step to the UI's agent trace.

**Builder agent:** turns the photo into a verified anatomy.

1. Drafts the anatomy as JSON: the object, 12–28 parts with sizes and positions in millimetres, connections, how-it-works scenarios and failure modes.
2. Verifies the draft with its tools:
   - `find_in_photo` runs **SAM 3.1** on a noun phrase. It confirms each part the draft calls visible, reports where the part really is in the object's millimetre frame, and catches parts the draft imagined.
   - `check_geometry` flags parts outside the object, internal parts that share the same space, and removal orders that dig outward.
   - `view_layout` draws the agent's own layout over the photo so it can see what's off.
3. Fixes problems with `update_part`, `add_part` and `remove_part`, then calls `finish`. A final SAM 3.1 pass grounds the visible parts. The object mask becomes the 3D shell, extruded with the photo projected onto it.

**Guide agent:** answers prompts by driving the 3D viewer.

- `highlight`, `play_scenario` / `create_scenario`, `show_failure` / `create_failure` and `set_explode` control the viewer.
- `look_closer` crops the photo around a part.
- `ask_user` asks a single question when a symptom fits more than one fault. The question appears in the conversation with tap-to-answer options, and the conversation resumes on the user's reply.

| Model | Used for |
|---|---|
| **Muse Wonder** | Both agents |
| **SAM 3.1** | Builder verification, grounding, the guide's close-ups |
| **SAM 3D Objects** (open weights, on Modal) | Optional 3D mesh that replaces the extruded silhouette |
| **Muse Voice Transcribe** | Push-to-talk questions |

The viewer (React + three.js) shows scenarios as a chain of parts with a signal pulse running between them. For failures, it lifts off the parts in the way, outermost first, then pulses the culprit red. Hidden parts are inferred, not measured. Each part is marked either "seen in photo · SAM 3.1" or "hidden · inferred".

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
src/magneye/   FastAPI app, Meta Model API client, agent loop, builder + guide agents, checks, SAM grounding
web/           React + three.js viewer
gpu/           SAM 3D Objects endpoint for Modal
tests/         Backend + agent tests (scripted fake model) and a UI fixture session
```
