# Face Recognition: 20 cameras on a Mac mini M4

This system recognizes faces in real time from up to 20 RTSP cameras. SCRFD finds faces and ArcFace identifies them. Both run as Core ML models on the **Apple Neural Engine (ANE)**, and video is decoded by the Mac's hardware decoder, so CPU use stays low.

---

## How to run

### 1. Requirements

- Mac with Apple Silicon (M1–M4), macOS 14 or later
- [Homebrew](https://brew.sh)
- Your models: `scrfd.mlpackage` and `arcface.mlpackage`. If you only have the InsightFace ONNX files, see step 4b.

### 2. Install

```bash
# system packages: Python 3.12, GStreamer (includes the VideoToolbox hardware decoder) and build tools for PyGObject
brew install python@3.12 gstreamer gobject-introspection pkg-config cairo ffmpeg

git clone https://github.com/maverick2100/facereco.git
cd facereco

/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Check the install. It should print the GStreamer version and an empty "missing" list:

```bash
python -c "import gi; gi.require_version('Gst','1.0'); from gi.repository import Gst; Gst.init(None); print(Gst.version_string()); print('missing:', [e for e in ('rtspsrc','rtph264depay','h264parse','vtdec','videorate','videoconvert','appsink') if not Gst.ElementFactory.find(e)])"
```

> Use Python **3.12**, not the system Python 3.9. `requirements.txt` holds every Python dependency, pinned to the tested versions: the app, the GStreamer bindings (PyGObject, which needs the brew packages above), PostgreSQL, model conversion (torch, onnx) and tests. Run facereco natively, not in Docker: Core ML, the Neural Engine and VideoToolbox decoding only exist on macOS itself.

### 3. Configure

```bash
cp .env.example .env            # all settings, with comments
```

Edit **`config/cameras.yaml`** with one entry per camera:

```yaml
cameras:
  - camera_id: cam01
    name: Main Entrance
    url: rtsp://admin:password@192.168.1.101:554/Streaming/Channels/102   # use the SUB-stream
    fps: 4            # frames/s analysed per camera (the rest are dropped after decoding)
    width: 1280       # should match the stream resolution
    height: 720
    # codec: h265     # for HEVC cameras (default h264; "auto" also works)
    # roi: [200, 100, 1080, 700]   # only look for faces / people inside this box
    tasks: [face]     # face | person | [face, person]  (default: face)
```

**Choosing face recognition vs person counting per camera:** `tasks` decides what runs on each camera. Each stream is decoded once, even when a camera runs both tasks. To change it for every camera at once, set it under `defaults:`.

| `tasks` | Runs | Writes to |
|---|---|---|
| `[face]` | SCRFD + ArcFace recognition | offender search log (via `fn_search_offender`) |
| `[person]` | YOLOv7 person detection + counting | `person_counts` |
| `[face, person]` | both | both |

**Counting people IN and OUT across a line:** give a person camera a `roi` line `[x1, y1, x2, y2]` (its two ends). A person whose feet cross it in the camera's `in_direction` (`down`, `up`, `right` or `left`, default `down`) counts as one **IN**; the opposite direction is one **OUT**. Each tracked person is counted once, so feet bouncing back over the line at a door don't add an OUT (`FR_PERSON_COUNT_ONCE`). The new side must hold for `FR_PERSON_LINE_MIN_FRAMES` frames (default 2). Points can be pixels or 0–1 fractions of the frame.

```yaml
  - camera_id: cam05
    url: rtsp://...
    tasks: [person]
    roi: [0.0, 0.6, 1.0, 0.6]      # horizontal line across the frame, 60% down
    in_direction: down             # walking down across it = IN
```

| Table | What it holds |
|---|---|
| `person_counts` | one row per camera every `FR_PERSON_FLUSH_S`, now with `in_count`, `out_count`, `zone_count` (IN − OUT occupancy at the end of the interval) |
| `person_crossings` | one row per crossing: `camera_id`, `track_id`, `direction` (`in`/`out`), `ts`, `x`, `y`. Written in one batch every ~2 s |

`python main.py counts` shows IN/OUT per camera, and `run --show` draws the line with live IN/OUT totals.

Passwords may contain special characters (`@ # / ?`). They're handled automatically and are never written to the database.

### 3b. Database: SQLite or PostgreSQL

By default everything is stored in a local SQLite file (`data/face_recognition.db`), and nothing needs to be set up. To use PostgreSQL instead, put your server's address and credentials in **`.env`**. Use either one URL:

```bash
FR_DATABASE_URL=postgresql://facereco:secret@192.168.1.50:5432/facereco
```

or separate fields. Separate fields need no URL-encoding, so they're easier when the password has special characters:

```bash
FR_PG_HOST=192.168.1.50
FR_PG_PORT=5432
FR_PG_USER=facereco
FR_PG_PASSWORD=p@ss:w/rd#1
FR_PG_DATABASE=facereco
FR_PG_SSLMODE=prefer        # require / verify-full for a server outside your network
```

If both are set, the `FR_PG_*` fields override the matching part of the URL. The database must already exist, and the tables are created on first start. Then check the connection:

```bash
python main.py db-check                       # prints OK + row counts, or the connection error
python scripts/migrate_sqlite_to_postgres.py  # optional: copy existing SQLite data (people, person counts)
```

`.env` is in `.gitignore`, so your credentials stay out of git, and the password is masked in every log line.

### 3c. Optional: cameras and use cases from your stored procedure (frontend-driven)

Instead of `cameras.yaml`, the camera list can come from the database, so the frontend decides which cameras run face recognition and which run person counting. Set this in `.env`:

```bash
FR_CAMERA_SOURCE=db
FR_CAMERA_SQL=SELECT * FROM sp_get_cameras()   # or your own SP / query
```

`python main.py run` then calls the SP and starts the right use case on each camera. Face matches are logged by `fn_search_offender`; person counts go to `person_counts` / `person_crossings`. When the SP's answer changes (a camera is added, removed or switched between face and person, or its IN/OUT line is moved), the system saves what it has counted so far and restarts the pipelines with the new list. This takes a few seconds and needs no manual restart.

- **The SP's rows:** one row per camera. Columns `camera_id`, `url`, and either `tasks` (`face`, `person`, `face,person` or `both`) or boolean `face` / `person` columns. Optional columns: `name`, `location`, `enabled`, `roi`, `fps`, `width`, `height`, `codec`. Common alternative names such as `id`, `rtsp_url`, `use_case` and `is_active` also work, other columns are ignored, and NULLs fall back to `defaults:` in `cameras.yaml`. A bad row is logged and skipped, and the other cameras keep running.
- **Instant updates:** on PostgreSQL, anything that runs `NOTIFY camera_changed` triggers an immediate reload. Otherwise changes are picked up by a re-check every `FR_CAMERA_POLL_S` (30 s).
- **No SP yet?** [database/sql/camera_config_postgres.sql](database/sql/camera_config_postgres.sql) creates a `camera_config` table, `sp_get_cameras()`, `sp_upsert_camera(...)`, `sp_delete_camera(...)`, and a trigger that sends the NOTIFY:

  ```bash
  psql "$FR_DATABASE_URL" -f database/sql/camera_config_postgres.sql
  ```
  ```sql
  SELECT sp_upsert_camera(p_camera_id => 'cam07', p_name => 'Back door',
                          p_url => 'rtsp://user:pass@10.0.0.7/stream2', p_run_face => false, p_run_person => true,
                          p_roi => '[0.5, 0, 0.5, 1]');   -- person counting with an IN/OUT line
  ```
- **End-to-end test:** `python scripts/e2e_test.py` adds two temporary cameras through the SPs and checks the whole chain against your database: pipelines start, person counts and IN/OUT crossings arrive, a use-case change via the SP restarts within seconds, and a disabled camera stops. Then it removes them. Add `--fake-models` to test without `.mlpackage` files, or `--face-url` / `--person-url file:///clip.mp4` to use real video with the real models.
- **Check it:** `python main.py cameras` prints what the SP returns, including which cameras run face, which run person counting, and which have an IN/OUT line, without starting anything.

### 3d. Optional: HTTP API for the frontend

With PostgreSQL and `FR_CAMERA_SOURCE=db`, facereco serves an API, so a frontend can manage cameras and read the dashboard without a backend of its own:

```bash
python main.py run --api        # pipelines + API in one process (or: python main.py api, API only)
open http://127.0.0.1:8090/docs # every endpoint, with a "Try it out" button
```

On start-up, facereco creates `camera_config` and all SPs if they're missing (`FR_API_AUTO_SETUP`; `python main.py db-setup` does it by hand). The API and a backend calling the SPs directly both write the same table, so use either or both. A camera added with `sp_upsert_camera` shows up in `GET /cameras`, and one added through the API is returned by `sp_get_cameras()`. Either way, its use cases start within about a second.

| Endpoint | What it does |
|---|---|
| `POST /cameras` | Add a camera: `{"camera_id", "name", "url", "run_face", "run_person", "roi"?, ...}` |
| `PUT /cameras/{id}` | Add or replace (201 = created, 200 = updated) |
| `PATCH /cameras/{id}` | Change only the fields sent, e.g. `{"run_face": true}` or a new `roi` line |
| `POST /cameras/{id}/start` · `/stop` | Turn a camera on or off (settings kept) |
| `DELETE /cameras/{id}` | Remove a camera |
| `GET /cameras` · `/cameras/{id}` | Cameras with `use_cases`, `state` (`streaming` / `starting` / `stopped` / ...) and live fps |
| `GET /dashboard/summary` · `/hourly` · `/daily` · `/cameras` | Person-count numbers (`sp_dash_*`); filters `day`, `from`/`to`, `cameras=a,b`, `tz` |
| `GET /status` · `/health` | API, database and facereco status |

- **Passwords:** `GET` returns camera URLs as `rtsp://user:***@host/...`. Sending that URL back unchanged keeps the stored password.
- **Validation:** bad input (non-stream URL, a line whose two ends are the same point, unknown field) gets a 422 with the reason, before anything reaches facereco.
- **Security:** the API listens on `127.0.0.1` by default. To serve other machines, set `FR_API_HOST=0.0.0.0` and `FR_API_KEY` (clients send `X-API-Key`). For a browser app on another origin, list it in `FR_API_CORS_ORIGINS`.

### 4. Add the models

```
models/scrfd.mlpackage
models/arcface.mlpackage
models/yolov7.mlpackage     # only needed if any camera has tasks: [person]
```

The YOLOv7 engine ([person_detection/yolo_engine.py](person_detection/yolo_engine.py)) works out the output layout from the tensor shapes. It handles decoded `(1, N, 85)` output (`export.py --grid`), raw `(1, 3, H, W, 85)` heads, `--end2end` NMS output `(N, 7)`, and Core ML NMS pipelines (`coordinates` / `confidence`). It logs the layout it detected at start-up. For a model with array input, pixels are divided by `FR_PERSON_INPUT_STD` (255). For a model with image input, the `1/255` scale must be built into the model. If your model wasn't trained on COCO, set `FR_PERSON_NUM_CLASSES` and `FR_PERSON_CLASS_ID`.

**4b. Optional: convert from ONNX** (e.g. InsightFace `buffalo_l`: `det_10g.onnx` + `w600k_r50.onnx`, and a YOLOv7 export). The conversion packages come with `requirements.txt`:

```bash
python scripts/convert_models.py --scrfd det_10g.onnx --arcface w600k_r50.onnx --yolo yolov7.onnx --out models/
```

SCRFD must be a version **with landmarks** (`kps` outputs, like `det_10g.onnx`); the face pipeline needs them to align faces.

The converter applies the ANE optimizations described in [Making sure the models run on the ANE](#making-sure-the-models-run-on-the-ane).

### 5. Test each part

```bash
python scripts/test_coreml.py --image group_photo.jpg   # models load, faces detected, annotated image saved
python scripts/check_ane.py                             # per model: % of layers on ANE / CPU / GPU, and why
python scripts/benchmark_scrfd.py                       # ANE vs CPU speed and CPU cost
python scripts/benchmark_arcface.py
python scripts/test_gstreamer.py --cameras cam01,cam02  # cameras stream (no models needed)
python scripts/test_gstreamer.py --url rtsp://user:pass@ip/stream   # test a single URL
python scripts/init_database.py                         # creates the tables (SQLite file or PostgreSQL)
pytest -q                                               # unit tests (56; PostgreSQL ones run when FR_TEST_DATABASE_URL is set)
```

### 6. Enrol people

Put 3–10 photos per person in a folder named after them:

```
data/enrollment/
    Alice Smith/  1.jpg 2.jpg 3.jpg
    Bob/          bob1.png bob2.png
```

```bash
python main.py enroll          # enrols everyone in data/enrollment
python main.py persons         # list enrolled people
```

Photos taken by the real cameras give the best results. You can re-run `enroll` at any time, even while the system is running: new photos are added and the running system reloads within 30 s.

### 7. Run

```bash
python main.py run                      # all enabled cameras
python main.py run --cameras cam01,cam02
python main.py run --show               # also open a window with live boxes + names
python main.py run --show cam01,cam02   # window shows only these cameras
```

Stop with **Ctrl+C**, which shuts down cleanly. With `--show`, you can also press **q** or **Esc** in the window. The window shows every camera in a grid, with known faces in green (name + similarity), unknown faces in red, faces still being checked in grey, people in blue and the `roi` in yellow. It shows the frames that were actually analysed, so it updates at each camera's `fps`. The window needs `opencv-python`; the `opencv-python-headless` build can't open windows on some installs. While it runs:

| Where | What |
|---|---|
| console / `logs/face_recognition.log` | a status line every 30 s (cameras, fps, ms per frame, CPU, memory) |
| `logs/events.log` | one JSON line per recognition |
| `data/face_recognition.db` | SQLite: persons, person_counts |
| `http://127.0.0.1:8085/health` | short status (`/metrics` = everything, `/cameras` = per camera) |
| `data/exports/snapshots/` | face snapshots, if `FR_SAVE_SNAPSHOTS=true` |

### 8. Other commands

```bash
python main.py enroll --name "Alice" --images a.jpg b.jpg [--replace]   # rows in tbl_offenders
python main.py delete-person 3
python main.py counts --since 2026-11-13                       # people per camera: total distinct + peak
python main.py counts --camera cam05 --csv                     # every person_counts row -> data/exports/
python main.py info                                            # model + environment info
sudo .venv/bin/python main.py run    # also logs ANE power (mW) via powermetrics
```

**Person counting:** cameras with `tasks: [person]` append one row per camera to `person_counts` every `FR_PERSON_FLUSH_S` (60 s by default). Each row holds `current_count` (people in the last frame), `min_count`, `max_count` and `avg_count` (occupancy over the interval), `new_persons` (distinct people who appeared, from a light IoU tracker, so someone standing still is counted once) and `frames`. "Total persons detected" is `SUM(new_persons)`. Live totals are also available at `/metrics` under `person_counts`.

To run it as a service, start it with `launchd` or `tmux`. Running it as root is only needed for the ANE power readings.

---

## Tested results (Mac mini M4, 16 GB)

This test used InsightFace `buffalo_l` (SCRFD-10G + ArcFace R50) converted with `scripts/convert_models.py`, and 20 local H.264 RTSP streams (1280×720, 25 fps source, 4 fps analysed):

| Metric | Result |
|---|---|
| Cameras streaming | 20 / 20, 4.0 fps each |
| Detection | 80 frames/s, **4.3 ms/frame** on ANE (3× faster than CPU) |
| ArcFace | **2.2 ms/face** on ANE (5× faster, **30× less CPU** than CPU-only) |
| Faces skipped by the track cache | 85% (ArcFace only runs on new or re-checked faces) |
| Neural Engine busy | about 40%, so roughly 2× headroom |
| CPU (whole app, 20 cameras) | **about 65% of one core = 6.5% of the 10-core M4** |
| Hardware video decoding | about 11% of one core (VideoToolbox service) |
| Memory | about 600 MB RSS |
| Dropped or stale frames | 0 |
| Recognition | 6 / 6 people recognized on all 20 cameras, similarity 0.81–0.90, no false matches |

---

## How CPU use is kept low

```
cameras ─► GStreamer: rtspsrc ► vtdec (hardware) ► videorate (drop to N fps) ► BGR ► appsink
        ─► CameraQueue (newest frame only, per camera) ─► FrameScheduler (fair, drops stale frames)
        ─► [detector thread]   SCRFD on ANE, batch 4 ─► filter ─► track cache ─► align ─► quality gate
        ─► [recognizer thread] ArcFace on ANE, batch 16 ─► fn_search_offender (PostgreSQL) ─► de-duplicate ─► log
```

- **Decoding** runs on the media engine. Only the analysed frames are colour-converted.
- **Both models** run with `CPU_AND_NE`: Neural Engine plus CPU fallback, never the GPU.
- **Quality gate:** blurry, side-on or too-small faces never reach ArcFace.
- **Track cache:** a face that is still in view keeps its identity and is re-checked every 2 s.
- **Matching** happens in PostgreSQL (`fn_search_offender`) against the offender table.
- **Thread pools** for numpy and OpenCV are limited to 1 thread.
- **Compiled models** are cached in `models/.compiled/`, so startup takes under 0.1 s after the first run.

## Making sure the models run on the ANE

`scripts/benchmark_scrfd.py` and `benchmark_arcface.py` print **CPU ms/item** for each compute unit. On the ANE, `CPU_AND_NE` should be several times lower than `CPU_ONLY`. If the two are close, re-convert the model with these settings (`scripts/convert_models.py` does all of them):

1. **Fixed input shapes**, e.g. SCRFD `(1,3,640,640)` and ArcFace `(1,3,112,112)`.
2. **FP16 `mlprogram`**, `minimum_deployment_target=macOS14`.
3. **`ImageType` input**, so normalization runs inside the model:
   - SCRFD: `scale=1/128, bias=[-127.5/128]*3`, RGB
   - ArcFace: `scale=1/127.5, bias=[-1,-1,-1]`, RGB
4. **SCRFD heads cut before Transpose/Reshape.** Core ML runs those ops on the CPU. The engine decodes raw NCHW heads directly.

The code detects image vs. array inputs, output names and output layouts automatically. If an ArcFace model is converted with a fixed batch, e.g. `(16,3,112,112)`, it sends real batches.

## Tuning

| Variable (`.env`) | Default | Effect |
|---|---|---|
| `FR_MATCH_THRESHOLD` | 0.45 | Higher means fewer false matches but more "unknown". Calibrate on your own people (0.35–0.55). |
| `FR_MATCH_MARGIN` | 0.05 | Required gap between the best and second-best person |
| `FR_EVENT_COOLDOWN_S` | 10 | A person must be out of view this long before a new event is logged |
| `FR_TRACK_REVERIFY_S` | 2.0 | How often a tracked face is re-checked by ArcFace (0 = always) |
| `FR_MIN_FACE_SIZE` | 40 | Ignore faces smaller than this (pixels) |
| `FR_DET_BATCH` / `FR_REC_BATCH` | 4 / 16 | Larger means better ANE efficiency, slightly more latency |
| `FR_RECORD_UNKNOWN` | true | Log unknown faces too |
| `FR_SAVE_SNAPSHOTS` | false | Save a JPEG of each recognized face |
| `FR_COMPUTE_UNITS` | CPU_AND_NE | `ALL` lets Core ML also use the GPU |
| `fps`, `roi`, `priority` | per camera | Load control in `cameras.yaml` |

**Capacity:** load = cameras × fps (20 × 4 = 80 frames/s). If `/metrics` shows `inference_busy_pct` near 100% or `stale_dropped` rising, lower `fps`, set an `roi`, or raise `FR_MIN_FACE_SIZE`. Use camera sub-streams (720p), not 4K main streams.

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'gi'` | Install the brew packages from step 2, then `pip install -r requirements.txt` inside the venv. |
| Camera status `reconnecting`, error `Not found` | Wrong RTSP path. Test with `scripts/test_gstreamer.py --url ...`. |
| Camera never streams | Try `codec: h265` or `codec: auto`. Check that the URL works in VLC. |
| `Unrecognised SCRFD outputs` | Run `scripts/test_coreml.py` to see the output shapes. Set `FR_SCRFD_SIGMOID=true` if scores are logits. |
| Everyone is "unknown" | Lower `FR_MATCH_THRESHOLD`, enrol more and better photos, and check normalization (array-input models only: `FR_ARCFACE_MEAN/STD`). |
| `Address already in use` (health port) | Change `FR_HEALTH_PORT`. |
| High CPU with `FR_INGESTION_BACKEND=opencv` | The OpenCV fallback uses software decoding. Install GStreamer. |

## Project layout

```
main.py                 CLI + app wiring           constants.py   paths / model constants
config/                 settings (.env), cameras.yaml, logging.yaml
ingestion/              GStreamer pipelines, appsink, reconnects, camera health, OpenCV fallback
scheduler/              per-camera latest-frame queues, fair scheduler, micro-batching
inference/              Core ML loader + cache, ANE config, SCRFD / ArcFace engines, profiler
processing/             letterbox / normalization, alignment, quality gate, NMS
recognition/            embedding gallery, matcher, identity hot-reload, de-dup + track cache
person_detection/       YOLOv7 engine, person tracker, per-interval counter, person-count service
database/               SQLite (WAL) or PostgreSQL + repositories
services/               enrolment (writes the offender table), real-time recognition pipeline
monitoring/             CPU, memory, ANE, performance counters, health HTTP endpoint
scripts/                tests, benchmarks, DB init, ONNX -> Core ML converter
tests/                  unit tests (fake models, no hardware needed)
```
