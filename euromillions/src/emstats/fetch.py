"""Download the raw source files into data/raw/ and log what was fetched."""
import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone

import requests

from . import config as C


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_primary():
    r = requests.get(C.PRIMARY_URL, timeout=60)
    r.raise_for_status()
    C.PRIMARY_FILE.write_bytes(r.content)
    return {"url": C.PRIMARY_URL, "file": str(C.PRIMARY_FILE.relative_to(C.ROOT)),
            "sha256": _sha256(C.PRIMARY_FILE)}


def fetch_secondary():
    """The secondary source is frozen, so only download it once."""
    files = sorted(C.SECONDARY_DIR.glob("*.txt"))
    meta = requests.get(
        f"https://pypi.org/pypi/{C.SECONDARY_PACKAGE}/{C.SECONDARY_VERSION}/json", timeout=60
    ).json() if not files else None
    if not files:
        wheel = next(u for u in meta["urls"] if u["filename"].endswith(".whl"))
        blob = requests.get(wheel["url"], timeout=60).content
        C.SECONDARY_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            for name in z.namelist():
                if name.startswith("pyeuromil/data/") and name.endswith(".txt"):
                    (C.SECONDARY_DIR / name.rsplit("/", 1)[1]).write_bytes(z.read(name))
        files = sorted(C.SECONDARY_DIR.glob("*.txt"))
    return {"package": f"{C.SECONDARY_PACKAGE}=={C.SECONDARY_VERSION}",
            "files": [f.name for f in files]}


def fetch_all(offline=False):
    C.RAW.mkdir(parents=True, exist_ok=True)
    if offline:
        if not C.PRIMARY_FILE.exists():
            raise FileNotFoundError(f"--offline but {C.PRIMARY_FILE} is missing")
        print("offline: using existing raw files")
        return
    log = {"fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "primary": fetch_primary(), "secondary": fetch_secondary()}
    C.FETCH_LOG.write_text(json.dumps(log, indent=2))
    print(f"fetched primary ({log['primary']['sha256'][:12]}) and secondary")
