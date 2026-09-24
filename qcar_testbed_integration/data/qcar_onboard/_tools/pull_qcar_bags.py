#!/usr/bin/env python3
"""Mirror a QCar's raw Vicon capture bags into qcar_onboard/<IP>/bags/.

Written 2026-09-17 to replace a transfer script that was lost (mentioned in
the project's bug log as sync_198.sh / pull_all_bags.sh, used to pull
qcar-52775/52776's bags but never saved to any repo). Works against ANY
QCar on the network, not just those two -- pass --remote-dir if a given
car keeps its bags somewhere other than the default.

Read-only on the vehicle: probes + `tar -cf -`, nothing else. No install,
no move, no delete on the car side.

    python3 pull_qcar_bags.py 192.168.1.198
    python3 pull_qcar_bags.py 192.168.1.158 --password <password>       # if SSH keys aren't set up on this machine
    python3 pull_qcar_bags.py 192.168.1.198 --manifest-only          # re-describe what's already on disk, pull nothing
    python3 pull_qcar_bags.py 192.168.0.200 --remote-dir /home/nvidia/qcar_bags  # a car with a different layout

Needs `paramiko` (not in the heal38 env this project otherwise uses --
found already installed under the `qlab` pyenv env, 2026-09-17, or
`pip install paramiko` into whichever env you run this with).
"""
import argparse
import hashlib
import json
import os
import sys
import tarfile
import time

import paramiko

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # _tools/ -> qcar_onboard/ -> data/ -> qcar_testbed_integration/
DEFAULT_CONF = os.path.join(HERE, "conf.json")


class _Conf(dict):
    """conf.json contents. It is THE source of every default -- nothing is
    hardcoded in this script; a CLI flag only overrides a key for one run.
    A missing key is a clear error, never a silent fallback."""

    def __init__(self, path, data):
        dict.__init__(self, data)
        self.path = path

    def __missing__(self, key):
        raise SystemExit("%s has no %r key -- add it there" % (self.path, key))

    def path_of(self, key):
        """Path-valued key: relative paths resolve against the
        qcar_testbed_integration/ root (REPO_ROOT); null stays None."""
        v = self[key]
        return v if not v or os.path.isabs(v) else os.path.join(REPO_ROOT, v)


def _load_conf(path):
    if not os.path.exists(path):
        raise SystemExit("conf.json not found: %s" % path)
    with open(path) as f:
        return _Conf(path, json.load(f))




# Everything from this line down in a manifest is hand-written and preserved
# across a refresh.
MARKER = "<!-- HAND-WRITTEN NOTES -- everything below this line survives a refresh -->"

PROBES = [
    ("hostname",  "hostname"),
    ("kernel",    "uname -srm"),
    ("serial",    "cat /proc/device-tree/serial-number 2>/dev/null | tr -d '\\0'"),
    ("mac",       "cat /sys/class/net/eth0/address 2>/dev/null"),
    ("uptime",    "uptime -p"),
    ("clock",     "date -u +%Y-%m-%dT%H:%M:%SZ"),
    ("disk_home", "df -h /home/nvidia | tail -1"),
]


def connect(host, user, password):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    if password:
        c.connect(host, username=user, password=password, timeout=20,
                  banner_timeout=20, auth_timeout=20,
                  look_for_keys=False, allow_agent=False)
    else:
        c.connect(host, username=user, timeout=20,
                  banner_timeout=20, auth_timeout=20,
                  look_for_keys=True, allow_agent=True)
    return c


def run(client, cmd, timeout=60):
    _, out, err = client.exec_command(cmd, timeout=timeout)
    text = out.read().decode("utf-8", "replace")
    status = out.channel.recv_exit_status()
    return text.strip(), err.read().decode("utf-8", "replace").strip(), status


def pull_tree(client, remote, dest_dir, label):
    """Stream one remote directory down as a tar and unpack it."""
    _, _, status = run(client, "test -d %s" % remote)
    if status != 0:
        print("  skip     %-34s (does not exist on this car)" % remote)
        return None

    parent, leaf = remote.rsplit("/", 1)
    _, out, err = client.exec_command("tar -cf - -C %s %s" % (parent, leaf), timeout=None)
    out.channel.settimeout(600)

    tar_path = os.path.join(dest_dir, "_%s.tar" % label.replace("/", "_"))
    digest = hashlib.sha256()
    total = 0
    t0 = time.time()
    last_print_t = t0
    last_print_bytes = 0
    PRINT_EVERY_S = 1.0
    with open(tar_path, "wb") as fh:
        while True:
            chunk = out.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            digest.update(chunk)
            total += len(chunk)

            now = time.time()
            if now - last_print_t >= PRINT_EVERY_S:
                inst_speed = (total - last_print_bytes) / (now - last_print_t) / 1048576.0
                avg_speed = total / (now - t0) / 1048576.0
                sys.stdout.write(
                    "\r  pulling  %-34s    %8.1f MB   %6.2f MB/s (avg %6.2f MB/s)   %5.0f s"
                    % (remote, total / 1048576.0, inst_speed, avg_speed, now - t0))
                sys.stdout.flush()
                last_print_t = now
                last_print_bytes = total
    if last_print_bytes:  # only clear the progress line if we ever printed one
        sys.stdout.write("\r" + " " * 100 + "\r")
        sys.stdout.flush()
    code = out.channel.recv_exit_status()
    stderr = err.read().decode("utf-8", "replace").strip()
    if code != 0:
        os.remove(tar_path)
        raise RuntimeError("tar of %s failed (%d): %s" % (remote, code, stderr))

    target = os.path.join(dest_dir, label)
    if os.path.exists(target):
        os.rename(target, target + ".superseded_%s" % time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    os.makedirs(os.path.dirname(target) or dest_dir, exist_ok=True)
    with tarfile.open(tar_path) as tf:
        members = tf.getnames()
        tf.extractall(os.path.dirname(target) or dest_dir)
    os.rename(os.path.join(os.path.dirname(target) or dest_dir, leaf), target)
    os.remove(tar_path)

    print("  ok       %-34s -> %s  (%.1f MB, %d files, %.0f s)"
          % (remote, label, total / 1048576.0, len(members), time.time() - t0))
    return {"remote": remote, "local": label, "bytes": total,
            "files": len(members), "sha256": digest.hexdigest(),
            "seconds": round(time.time() - t0, 1)}


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--conf", default=DEFAULT_CONF)
    conf_path = pre.parse_known_args()[0].conf
    conf = _load_conf(conf_path)

    ap = argparse.ArgumentParser(
        description=__doc__ + "\n\nDefaults come from conf.json next to this script -- edit "
                    "that file and run with no args, or pass a flag/positional to override "
                    "it for one run.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("host", nargs="?", default=conf["host"],
                     help="any QCar IP reachable on the lab network")
    ap.add_argument("--conf", default=DEFAULT_CONF,
                     help="Path to the conf.json to read defaults from (default: %s)" % DEFAULT_CONF)
    ap.add_argument("--user", default=conf["user"])
    ap.add_argument("--password", default=conf["password"],
                    help="omit to use SSH keys instead of a password")
    ap.add_argument("--remote-dir", default=conf["remote_dir"],
                    help="remote bags directory (default: conf.json's remote_dir)")
    ap.add_argument("--local-root", default=conf.path_of("local_root"),
                    help="local folder holding one <host>/ per car (default: conf.json's local_root)")
    ap.add_argument("--manifest-only", action="store_true", default=conf["manifest_only"],
                    help="only probe and rewrite the manifest; pull nothing")
    args = ap.parse_args()

    if not args.host:
        ap.error("host not set: pass it positionally, or set \"host\" in %s" % conf_path)

    car_dir = os.path.join(args.local_root, args.host)
    LOCAL_LABEL = conf["local_label"]
    os.makedirs(car_dir, exist_ok=True)

    print("connecting to %s ..." % args.host)
    client = connect(args.host, args.user, args.password)

    facts = {}
    for key, cmd in PROBES:
        text, _, _ = run(client, cmd)
        facts[key] = text.splitlines()[0] if text else "(empty)"
    bags_count, _, _ = run(client, "ls -1 %s 2>/dev/null | wc -l" % args.remote_dir)
    facts["bags_count"] = bags_count or "0"
    print("  %s · %s bags on the car" % (facts["hostname"], facts["bags_count"]))

    pulled = []
    if args.manifest_only:
        target = os.path.join(car_dir, LOCAL_LABEL)
        if os.path.isdir(target):
            n = sum(len(f) for _, _, f in os.walk(target))
            b = sum(os.path.getsize(os.path.join(r, f))
                    for r, _, fs in os.walk(target) for f in fs)
            pulled.append({"remote": args.remote_dir, "local": LOCAL_LABEL,
                           "bytes": b, "files": n, "sha256": "(already on disk, not re-pulled)"})
            print("  on disk  %-34s -> %s  (%.1f MB, %d files)"
                  % (args.remote_dir, LOCAL_LABEL, b / 1048576.0, n))
    else:
        got = pull_tree(client, args.remote_dir, car_dir, LOCAL_LABEL)
        if got:
            pulled.append(got)
    client.close()

    stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime())
    lines = [
        "# QCar %s (bags)" % args.host,
        "",
        "`%s`. Mirror of `%s` pulled on **%s**." % (facts["hostname"], args.remote_dir, stamp),
        "Read-only operation on the vehicle: probes + `tar -cf -`, nothing else.",
        "Regenerate with `python3 _tools/pull_qcar_bags.py %s`." % args.host,
        "",
        "## The car",
        "", "| fact | value |", "|---|---|",
    ]
    lines += ["| `%s` | `%s` |" % (k, facts[k]) for k, _ in PROBES]
    lines += [
        "",
        "## What was pulled",
        "", "| local | remote | size | files | tar sha256 |", "|---|---|---|---|---|",
    ]
    for p in pulled:
        lines.append("| `%s/` | `%s` | %.1f MB | %d | `%s` |"
                     % (p["local"], p["remote"], p["bytes"] / 1048576.0,
                        p["files"], p["sha256"]))

    manifest = os.path.join(car_dir, "BAGS_MANIFEST.md")
    kept = ""
    if os.path.exists(manifest):
        prev = open(manifest).read()
        if MARKER in prev:
            kept = "\n" + prev[prev.index(MARKER):]
    with open(manifest, "w") as fh:
        fh.write("\n".join(lines) + "\n" + kept)
    print("\nBAGS_MANIFEST.md written to %s" % car_dir)


if __name__ == "__main__":
    sys.exit(main())
