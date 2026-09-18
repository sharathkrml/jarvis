import os
from pathlib import Path

_path = Path(__file__).with_name(".env")
if _path.exists():
    for _line in _path.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _key, _, _val = _line.partition("=")
        os.environ.setdefault(_key.strip(), _val.strip().strip("\"'"))


def env(key, default):
    return os.environ.get(key, default)


if __name__ == "__main__":
    assert env("JARVIS_TEST_MISSING", "fallback") == "fallback"
    os.environ["JARVIS_TEST_SET"] = "x"
    assert env("JARVIS_TEST_SET", "fallback") == "x"
    print("ok")
