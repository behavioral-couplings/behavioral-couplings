from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / ".env"


def _parse_env_line(line: str) -> tuple[str, str] | None:
	line = line.strip()
	if not line or line.startswith("#"):
		return None
	if line.startswith("export "):
		line = line[len("export ") :].lstrip()
	if "=" not in line:
		return None
	key, value = line.split("=", 1)
	key = key.strip()
	value = value.strip()
	if not key:
		return None
	if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
		value = value[1:-1]
	return key, value


def _load_env_without_dependency(path: Path) -> None:
	if not path.exists():
		return
	for line in path.read_text(encoding="utf-8").splitlines():
		parsed = _parse_env_line(line)
		if parsed is None:
			continue
		key, value = parsed
		os.environ.setdefault(key, value)


def load_env() -> None:
	try:
		from dotenv import load_dotenv
	except ImportError:
		_load_env_without_dependency(ENV_PATH)
	else:
		load_dotenv(ENV_PATH, override=False)
	os.environ.setdefault("COUPLING_HOME", str(REPO_ROOT))


load_env()

coupling_home = Path(os.environ["COUPLING_HOME"]).expanduser().resolve()
os.environ["COUPLING_HOME"] = str(coupling_home)
DATASET_DIR = coupling_home / "dataset"
MODEL_DIR = coupling_home / "models"
OUTPUT_DIR = coupling_home / "outputs"
