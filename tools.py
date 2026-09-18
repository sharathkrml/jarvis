import json
import shlex
import subprocess

from config import env

TIMEOUT = 30
MAX_OUT = 2000

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": "Run a read-only shell command on the user's Mac and return its "
            "output. Use for checking the time, disk space, files, git status, system info. "
            "Writes, deletes, and redirects are not permitted.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "The shell command to run."}
                },
                "required": ["cmd"],
            },
        },
    }
]

# Read-only commands. Anything that can write/delete/execute another program is out.
SAFE = {
    "ls", "cat", "date", "cal", "df", "du", "uptime", "whoami", "id", "pwd",
    "echo", "printf", "head", "tail", "wc", "grep", "find", "ps", "uname",
    "sw_vers", "which", "hostname", "printenv", "file", "stat", "sort", "cut",
    "tr", "tree", "realpath", "readlink", "dirname", "basename", "shasum", "md5",
    "jq", "system_profiler", "git",
}
# git subcommands that only read.
GIT_READ = {
    "status", "log", "diff", "show", "rev-parse", "ls-files", "ls-tree",
    "blame", "describe", "cat-file", "shortlog", "show-ref", "diff-tree", "grep",
}
# Flags that turn an otherwise-read command into a writer.
DENY_FLAGS = {
    "find": {"-exec", "-execdir", "-ok", "-okdir", "-delete", "-fprint", "-fprint0", "-fls"},
    "sort": {"-o", "--output"},
    "git": {"--output"},
}
SEPARATORS = set(";|&\n")

_CALL_TAG = "<tool_call>"


def parse_calls(text):
    """Extract (name, args) pairs from <tool_call> blocks, closed or not."""
    dec = json.JSONDecoder()
    calls, pos = [], 0
    while True:
        i = text.find(_CALL_TAG, pos)
        if i == -1:
            return calls
        i += len(_CALL_TAG)
        while i < len(text) and text[i].isspace():
            i += 1
        try:
            data, end = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            pos = i
            continue
        if isinstance(data, dict) and "name" in data:
            calls.append((data["name"], data.get("arguments") or {}))
        pos = end


def _segments(cmd):
    """Split on unquoted ; | & newline. Reject redirects, substitutions, bad quoting."""
    segs, cur, quote, i, n = [], [], None, 0, len(cmd)
    while i < n:
        c = cmd[i]
        if quote == "'":
            cur.append(c)
            if c == "'":
                quote = None
            i += 1
            continue
        if quote == '"':
            if c == '"':
                quote = None
                cur.append(c)
                i += 1
                continue
            if c == "\\" and i + 1 < n:
                cur.extend((c, cmd[i + 1]))
                i += 2
                continue
            if c == "`" or (c == "$" and i + 1 < n and cmd[i + 1] in "({"):
                return []
            cur.append(c)
            i += 1
            continue
        if c in "'\"":
            quote = c
            cur.append(c)
            i += 1
            continue
        if c == "\\" and i + 1 < n:
            cur.extend((c, cmd[i + 1]))
            i += 2
            continue
        if c == "`" or (c == "$" and i + 1 < n and cmd[i + 1] in "({"):
            return []
        if c == ">" or (c == "<" and i + 1 < n and cmd[i + 1] == "("):
            return []
        if c in SEPARATORS:
            segs.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(c)
        i += 1
    if quote:
        return []
    segs.append("".join(cur))
    return [s.strip() for s in segs if s.strip()]


def _allowed(cmd):
    if env("JARVIS_BASH", "safe") == "full":
        return True
    segs = _segments(cmd)
    if not segs:
        return False
    for seg in segs:
        try:
            argv = shlex.split(seg)
        except ValueError:
            return False
        if not argv or argv[0] not in SAFE:
            return False
        if argv[0] == "git":
            sub = next((a for a in argv[1:] if not a.startswith("-")), None)
            if sub not in GIT_READ:
                return False
        bad = DENY_FLAGS.get(argv[0], set())
        if any(t == f or t.startswith(f) for t in argv[1:] for f in bad):
            return False
    return True


def run_bash(cmd):
    if not _allowed(cmd):
        return "Blocked: not in the safe allowlist. Set JARVIS_BASH=full to allow any command."
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return f"Timed out after {TIMEOUT}s."
    out = ((p.stdout or "") + (p.stderr or "")).strip()
    return out[:MAX_OUT] or "(no output)"


def run(name, args):
    if name == "run_bash":
        return run_bash(args.get("cmd", ""))
    return f"Unknown tool: {name}"


if __name__ == "__main__":
    assert parse_calls('x <tool_call>{"name":"run_bash","arguments":{"cmd":"date"}}</tool_call> y') == [("run_bash", {"cmd": "date"})]
    assert parse_calls('x <tool_call>{"name":"run_bash","arguments":{"cmd":"date"}} y') == [("run_bash", {"cmd": "date"})]
    assert parse_calls("no tools here") == []
    assert parse_calls("<tool_call>not json</tool_call>") == []
    assert _allowed("date")
    assert _allowed("ls -la && grep -r 'def' . | grep -v __init__ | head -20")
    assert _allowed('grep "a|b" llm.py')
    assert _allowed("git status")
    assert _allowed("git log --oneline -5")
    assert not _allowed("rm -rf /")
    assert not _allowed("date; rm -rf /")
    assert not _allowed("ls > out.txt")
    assert not _allowed("cat $(whoami)")
    assert not _allowed("echo `whoami`")
    assert not _allowed("find . -delete")
    assert not _allowed("find . -exec rm {} ;")
    assert not _allowed("git clean -fd")
    assert not _allowed("sort -o out.txt f")
    assert not _allowed("sort --output=out.txt f")
    assert not _allowed("git log --output=out.txt")
    assert not _allowed("python -c 'import os'")
    assert not _allowed("ls 'unterminated")
    assert run_bash("echo ok") == "ok"
    print("ok")
