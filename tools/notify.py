"""Discord webhook notifier. Usage: python tools/notify.py "message" """
import json
import sys
import urllib.request

WEBHOOK = (
    "https://discord.com/api/webhooks/"
    "1547292221340913835/CfWoQAeE9JbIhb_RLvDgsMb9ifa7bQISsvZU_BWmCXq3MCumznEmidDhsGSIPcXZk2kE"
)


def main() -> None:
    msg = sys.argv[1] if len(sys.argv) > 1 else "update"
    payload = json.dumps({"content": msg}).encode("utf-8")
    try:
        req = urllib.request.Request(WEBHOOK, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            print(f"notified ({r.status})")
        return
    except Exception:  # noqa: BLE001 - fall through to curl fallback
        pass
    try:
        import subprocess
        subprocess.run(
            ["curl", "-s", "-X", "POST", WEBHOOK, "-H", "Content-Type: application/json",
             "--data-binary", json.dumps({"content": msg})],
            timeout=15, check=False,
        )
        print("notified (curl fallback)")
    except Exception as e:  # noqa: BLE001 - notification must never kill the pipeline
        print(f"notify failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
