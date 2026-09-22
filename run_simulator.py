#!/usr/bin/env python3
"""Run the unmodified v3.2 simulator with this workspace's data and logs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "Incident_simulator" / "emilyx_lte_end_to_end_simulator_v3_2.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog="Additional options are passed to v3.2. Relative paths use your current directory.")
    parser.add_argument("mode", choices=("terrestrial", "leo"))
    parser.add_argument("--duration", type=int, default=600)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--start-utc")
    parser.add_argument("--tle-file", type=Path, default=ROOT / "Incident_simulator" / "satellite.tle")
    parser.add_argument("--min-elevation-deg", type=float, default=30.0)
    args, extra = parser.parse_known_args()
    if args.duration < 2:
        parser.error("--duration must be at least 2 seconds")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = (args.output_dir or ROOT / "runs" / f"{args.mode}_{args.duration}s_{stamp}").resolve()
    if out.exists():
        parser.error(f"Output already exists; choose a new --output-dir: {out}")
    python = ROOT / ".venv" / "bin" / "python"
    if not python.exists():
        parser.error("Run bash setup_simulator.sh first")
    prefix = "srsran_d2c" if args.mode == "leo" else "srsran"
    command = [str(python), "-u", str(SOURCE), "--tx-type", args.mode,
               "--duration", str(args.duration), "--output-dir", str(out)]
    for state in ("idle", "light", "medium", "loaded"):
        path = ROOT / "EMILY-X" / f"{prefix}_{state}.cf32"
        if not path.is_file():
            parser.error(f"Missing IQ: {path}")
        command += [f"--iq-{state}", str(path)]
    start = args.start_utc or ("2026-09-11T15:35:00Z" if args.mode == "leo" else "2026-09-02T19:00:00Z")
    command += ["--start-utc", start]
    if args.mode == "leo":
        command += ["--tle-file", str(args.tle_file.resolve()),
                    "--min-elevation-deg", str(args.min_elevation_deg)]
    else:
        command += ["--srtm-dir", str(ROOT / "srtm_data")]
    command += extra
    env = os.environ.copy()
    env.update(MPLBACKEND="Agg", MPLCONFIGDIR=str(ROOT / ".cache" / "matplotlib"),
               XDG_CACHE_HOME=str(ROOT / ".cache"))
    # Bound scientific-library thread counts; small per-path tasks otherwise oversubscribe.
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env.setdefault(key, "2")
    out.mkdir(parents=True)
    started = time.monotonic()
    manifest = dict(command=command, cwd=str(Path.cwd()), started_utc=stamp,
                    source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                    note="LEO defaults reproduce the supplied author's historical 30-degree example.")
    manifest_path = out / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(shlex.join(command), flush=True)
    with (out / "run.log").open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            code = 130
    manifest.update(exit_code=code, elapsed_seconds=round(time.monotonic() - started, 2))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exit {code}; elapsed {manifest['elapsed_seconds']} s; outputs: {out}")
    return code


if __name__ == "__main__":
    sys.exit(main())
