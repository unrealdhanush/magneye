# Magneye

See inside anything you can photograph. Give Magneye a photo (plus, optionally, up to five more from other sides or close up) and a short prompt. Meta's models work out what the object is, what's inside it and how it works. The result is an interactive 3D exploded view: you can play how the object works, or describe a fault and watch the likely culprit light up red.

Built for the Meta Global AI Developer Hackathon (Oct 26 – Nov 4, 2026).

## How it works

Two agents, both Muse Wonder (`muse-wonder-1.0`) with tools, stream every step to the UI's agent trace.

**Builder agent:** turns the photo into a verified anatomy.

1. Researches the object with Muse Wonder's web search: identifies the exact product, then looks for teardowns, specs and common failures (at most 6 searches) and writes a cited blueprint brief. Only URLs the search actually returned are kept as sources.
2. Finds other angles of the same product: photos on Wikimedia Commons (shown with licence and author) and on the pages the research cited (used as model input and linked, not shown). The user's own extra photos come first, each one kept. Muse Wonder vets each one (same body shape? which side?) and keeps up to three.
3. Drafts the anatomy as JSON from the photo, the brief and those views: the object, 12–28 parts with sizes and positions in millimetres, connections, how-it-works scenarios and failure modes. Each part is marked seen, documented (by a cited source) or inferred. Straps, bracelets and cables are accessories: the viewer frames the main body and moves them aside.
4. Verifies the draft with its tools:
   - `find_in_photo` runs **SAM 3.1** on a noun phrase. It confirms each part the draft calls visible, reports where the part really is in the object's millimetre frame, and catches parts the draft imagined.
   - `check_geometry` flags parts outside the object, internal parts that share the same space, and removal orders that dig outward.
   - `view_layout` draws the agent's own layout over the photo so it can see what's off.
5. Fixes problems with `move_parts` (shift the whole body at once), `update_parts`, `add_part` and `remove_part`, then calls `finish`. A final SAM 3.1 pass grounds the visible parts. The object mask becomes the 3D shell, extruded with the photo projected onto it. When a clean, upright side view is available, SAM 3.1 traces its outline (while verification runs) and the shell is bent to that side profile instead of a straight slab.

**Guide agent:** answers prompts by driving the 3D viewer.

- `highlight`, `play_scenario` / `create_scenario`, `show_failure` / `create_failure` and `set_explode` control the viewer.
- `look_closer` crops the photo around a part.
- `ask_user` asks a single question when a symptom fits more than one fault. The question appears in the conversation with tap-to-answer options, and the conversation resumes on the user's reply.

| Model | Used for |
|---|---|
| **Muse Wonder** | Both agents |
| **Muse Wonder + web search** | The builder's research and its vetting of reference photos |
| **SAM 3.1** | Builder verification, grounding, side profiles, the guide's close-ups |
| **SAM 3D Objects** (open weights, on Modal) | Optional 3D mesh that replaces the extruded silhouette |
| **Muse Voice Transcribe** | Push-to-talk questions |

The viewer (React + three.js) shows scenarios as a chain of parts with a signal pulse running between them. For failures, it lifts off the parts in the way, outermost first, then pulses the culprit red. Hidden parts are inferred or documented, not measured. Each part is marked "seen in photo · SAM 3.1", "documented · cited sources" or "hidden · inferred", and the object card lists the blueprint sources and the reference views.

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

## Benchmark: Muse Wonder vs Claude Opus 5.5

`bench/compare.py` runs the same photo through both agents with everything identical except the reasoning model: same prompts, SAM 3.1 calls, checks and agent code. Claude is reached through `bench/claude_adapter.py`, which translates the agents' messages to the Anthropic Messages API. The product itself only uses Meta models.

```bash
uv run --group bench python bench/compare.py photo.jpg --prompt "How does this work?"
```

It reports identification, part and SAM-grounding counts, leftover geometry issues, build time, tokens, and what the guide did for three prompts. Each run is saved as a session you can open in the viewer. Needs `MODEL_API_KEY` and `ANTHROPIC_API_KEY`.

## Layout

```
src/magneye/   FastAPI app, Meta Model API client, agent loop, builder + guide agents, checks, SAM grounding
web/           React + three.js viewer
gpu/           SAM 3D Objects endpoint for Modal
bench/         Muse Wonder vs Claude Opus 5.5 benchmark (not part of the product)
tests/         Backend + agent tests (scripted fake model) and a UI fixture session
```
