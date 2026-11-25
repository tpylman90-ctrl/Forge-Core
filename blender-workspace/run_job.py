"""Run a registered, trusted headless Blender job; never execute user-provided code."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
JOBS = {"cargo_crate": ROOT / "blender-workspace/jobs/cargo_crate.json"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", choices=sorted(JOBS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    asset = out / "asset"
    request = JOBS[args.job]
    log = out / "blender.log"
    env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "USER", "LANG", "LC_ALL", "TMPDIR", "TEMP", "TMP", "SYSTEMROOT", "LD_LIBRARY_PATH")}
    env["PYTHONNOUSERSITE"] = "1"
    with log.open("w", encoding="utf-8") as stream:
        version = subprocess.check_output(["blender", "--version"], env=env, text=True, timeout=30)
        stream.write(version + "\n")
        stream.flush()
        subprocess.run([
            "blender", "--background", "--factory-startup", "--disable-autoexec",
            "--python-exit-code", "1", "--python", str(ROOT/"production/build_blender.py"),
            "--", "--request", str(request), "--out", str(asset)
        ], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=1200)
        stream.flush()
        subprocess.run([sys.executable, str(ROOT/"production/validate_glb.py"), str(asset)],
                       cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=60)
    expected = {"asset.blend", "asset.glb", "front.png", "side.png", "top.png", "three_quarter.png", "request.json", "validation.json"}
    if not expected <= {p.name for p in asset.iterdir() if p.is_file()}:
        raise RuntimeError("Incomplete asset package")
    report = json.loads((asset/"validation.json").read_text())
    if report.get("status") != "pass":
        raise RuntimeError("Validation did not pass")
    files = {}
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        files[str(path.relative_to(out))] = {"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = {"schema_version":1,"job":args.job,"blender_version":version.splitlines()[0],
                "source_commit":os.getenv("GITHUB_SHA"),"run_id":os.getenv("GITHUB_RUN_ID"),
                "validation_status":report["status"],"files":files}
    (out/"artifact_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest,indent=2))
if __name__=="__main__":
    main()
