"""Maintain private SSH reverse forwarding for remote training's W&B traffic."""

import argparse
import hashlib
import os
import shlex
import socket
import subprocess
import time
from pathlib import Path


def tunnel_command(config, host, local_port, remote_port):
    return [
        "ssh", "-F", str(config), "-NT", "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=15", "-o", "ControlMaster=no", "-o", "ControlPath=none",
        "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=3", "-R",
        f"127.0.0.1:{remote_port}:127.0.0.1:{local_port}", host,
    ]


def unit_name(config, host, local_port, remote_port):
    identity = f"{config.resolve()}:{host}:{local_port}:{remote_port}"
    return "lerobot-wandb-proxy-" + hashlib.sha256(identity.encode()).hexdigest()[:12]


def probe_command(config, host, remote_port):
    # Do not print bodies/credentials. This checks the API, not merely an open TCP port.
    curl = shlex.join([
        "curl", "--silent", "--show-error", "--fail", "--max-time", "15",
        "--noproxy", "", "--proxy", f"http://127.0.0.1:{remote_port}",
        "--header", "Content-Type: application/json", "--data", '{"query":"{ __typename }"}',
        "--output", "/dev/null", "https://api.wandb.ai/graphql",
    ])
    # A server configured with GatewayPorts=yes can override SSH's requested bind.
    listener = f"127.0.0.1:{remote_port}"
    remote = (
        f"listeners=$(ss -H -ltn 'sport = :{remote_port}' | awk '{{print $4}}'); "
        f'if test -n "$listeners" && test "$listeners" != {listener}; then exit 77; fi; '
        f'test "$listeners" = {listener} && {curl}'
    )
    return ["ssh", "-F", str(config), "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, remote]


def ensure_user_manager():
    user = str(os.getuid())
    result = subprocess.run(
        ["loginctl", "show-user", user, "--property=Linger", "--value"],
        capture_output=True, text=True, check=True,
    )
    if result.stdout.strip() != "yes":
        # The tunnel must outlive logout, not just the submitting terminal.
        subprocess.run(["loginctl", "enable-linger", user], check=True)


def ensure(config, host, local_port, remote_port):
    with socket.create_connection(("127.0.0.1", local_port), timeout=5):
        pass
    ensure_user_manager()
    unit = unit_name(config, host, local_port, remote_port)
    state = subprocess.run(
        ["systemctl", "--user", "show", "--property=LoadState", "--value", unit],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if state == "loaded":
        # An auto-restarting unit is not "active" yet; do not create a duplicate.
        subprocess.run(["systemctl", "--user", "start", unit], check=True)
    elif state == "not-found":
        subprocess.run([
            "systemd-run", "--user", f"--unit={unit}", "--collect",
            "--property=Restart=always", "--property=RestartSec=5",
            "--property=StartLimitIntervalSec=0",
            "--description=LeRobot W&B private reverse proxy",
            *tunnel_command(config, host, local_port, remote_port),
        ], check=True)
    else:
        raise RuntimeError(f"Unexpected systemd unit state: {state}")
    for _ in range(12):
        result = subprocess.run(probe_command(config, host, remote_port), capture_output=True)
        if result.returncode == 77:
            subprocess.run(["systemctl", "--user", "stop", unit], check=True)
            raise RuntimeError(f"{host}: reverse proxy was not loopback-only; stopped tunnel")
        if result.returncode == 0:
            print(f"{host}: W&B proxy ready (systemd user unit {unit}, remote loopback {remote_port})")
            return
        time.sleep(2)
    raise RuntimeError(f"{host}: proxy preflight failed; inspect journalctl --user -u {unit}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-config", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--local-port", type=int, required=True)
    parser.add_argument("--remote-port", type=int, required=True)
    args = parser.parse_args()
    if not args.ssh_config.is_file() or args.host.startswith("-"):
        parser.error("invalid SSH config or host")
    if not all(0 < port < 65536 for port in (args.local_port, args.remote_port)):
        parser.error("ports must be in 1..65535")
    ensure(args.ssh_config.resolve(), args.host, args.local_port, args.remote_port)


if __name__ == "__main__":
    main()
