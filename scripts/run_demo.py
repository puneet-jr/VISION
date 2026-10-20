"""One-command demo server.

Serves the same app twice from one process:
  * http://127.0.0.1:8000   - for the laptop/desktop (localhost is a secure context)
  * https://<LAN-IP>:8443   - for phones: browsers only allow camera access over HTTPS

A self-signed certificate is generated into data/certs/ (git-ignored). Phones show a one-time
warning ("not private"); accept it to continue. This is a local demo setup: there is no
authentication, so only run it on a network you trust.

    python scripts/run_demo.py            # starts both and opens the dashboard
    python scripts/run_demo.py --no-browser --no-https
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime as dt
import ipaddress
import os
import socket
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn  # noqa: E402

from app.api.routes.network import lan_addresses  # noqa: E402
from app.core.config import BASE_DIR  # noqa: E402

CERT_DIR = BASE_DIR / "data" / "certs"


def ensure_certificate(ips: list[str]) -> tuple[Path, Path]:
    """Create (or reuse) a self-signed certificate valid for localhost and the given LAN IPs."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    CERT_DIR.mkdir(parents=True, exist_ok=True)
    cert_path, key_path, ips_path = CERT_DIR / "demo.crt", CERT_DIR / "demo.key", CERT_DIR / "ips.txt"
    wanted = ",".join(sorted(set(ips)))
    if cert_path.is_file() and key_path.is_file() and ips_path.is_file() and ips_path.read_text() == wanted:
        return cert_path, key_path

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "VisionShield AI local demo")])
    san = [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
    san += [x509.IPAddress(ipaddress.ip_address(ip)) for ip in ips]
    now = dt.datetime.now(dt.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(minutes=5)).not_valid_after(now + dt.timedelta(days=365))
            .add_extension(x509.SubjectAlternativeName(san), critical=False)
            .sign(key, hashes.SHA256()))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM,
                                           serialization.PrivateFormat.TraditionalOpenSSL,
                                           serialization.NoEncryption()))
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    ips_path.write_text(wanted)
    return cert_path, key_path


async def self_check(ips: list[str], port: int) -> None:
    """Confirm the HTTPS server answers on the LAN address (a local test: it cannot see a firewall
    or Wi-Fi client isolation, which only a real phone can reveal)."""
    await asyncio.sleep(2.5)
    target = ips[0] if ips else "127.0.0.1"
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection(target, port), 3)
        writer.close()
        print(f"  [ok] HTTPS server is listening on {target}:{port}")
    except (OSError, asyncio.TimeoutError):
        print(f"  [!!] could not connect to {target}:{port} - check the port is free and not blocked")


async def serve(args: argparse.Namespace) -> None:
    ips = lan_addresses()
    https_on = not args.no_https
    if https_on:
        os.environ["PUBLIC_HTTPS_PORT"] = str(args.https_port)  # picked up by Settings

    from app.main import create_app
    app = create_app()

    http = uvicorn.Server(uvicorn.Config(app, host=args.host, port=args.port, log_level="info"))
    servers = [http]
    if https_on:
        cert, key = ensure_certificate(ips)
        https = uvicorn.Server(uvicorn.Config(
            app, host="0.0.0.0", port=args.https_port, ssl_certfile=str(cert), ssl_keyfile=str(key),
            lifespan="off",  # the HTTP server already runs startup/shutdown for the shared app
            log_level="warning"))
        https.capture_signals = contextlib.nullcontext  # type: ignore[method-assign]
        servers.append(https)

    async def follow_http() -> None:
        while not http.should_exit:
            await asyncio.sleep(0.2)
        for server in servers[1:]:
            server.should_exit = True

    print("\n  VisionShield AI demo")
    print(f"  Dashboard (this computer): http://{args.host}:{args.port}/")
    if https_on:
        for ip in ips:
            print(f"  Phone camera page:         https://{ip}:{args.https_port}/phone")
        if not ips:
            print("  (no LAN address found - connect to Wi-Fi to use a phone)")
        print("  Phone: same Wi-Fi, accept the one-time certificate warning, allow the camera.")
        print("  Can't connect from the phone? (1) both on the SAME Wi-Fi; (2) campus/guest Wi-Fi often")
        print("  isolates devices - use the phone's hotspot or Windows Mobile Hotspot instead; (3) allow the")
        print(f"  port in the firewall (admin PowerShell):  netsh advfirewall firewall add rule name=\"VisionShield\" dir=in action=allow protocol=TCP localport={args.https_port}")
    print("  Ctrl+C to stop.\n")

    if not args.no_browser:
        asyncio.get_running_loop().call_later(2.0, webbrowser.open, f"http://{args.host}:{args.port}/")
    extra = [self_check(ips, args.https_port)] if https_on else []
    await asyncio.gather(*(s.serve() for s in servers), follow_http(), *extra)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--https-port", type=int, default=8443)
    parser.add_argument("--no-https", action="store_true", help="skip the LAN HTTPS server (no phone camera)")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # show messages immediately, even when redirected
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(serve(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
