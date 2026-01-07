#!/usr/bin/env python3
import sys

import requests

UA = "ArxivConjectureScraper/1.0 (For academic research)"


def check_eprint(arxiv_id: str) -> None:
    url = f"https://arxiv.org/e-print/{arxiv_id}"
    print(f"Requesting: {url}")

    resp = requests.get(
        url,
        headers={"User-Agent": UA},
        timeout=30,
        stream=False,
    )
    print(f"HTTP {resp.status_code}")

    if resp.status_code != 200:
        print("Non-200 response; cannot inspect body reliably.")
        return

    data = resp.content
    print(f"Downloaded {len(data)} bytes")

    head = data[:2048]
    try:
        text_head = head.decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"decode error: {e}")
        text_head = ""

    lower_head = text_head.lower()

    # Simple format heuristics
    is_gzip = head[:2] == b"\x1f\x8b"
    is_zip = head[:4] in (b"PK\x03\x04", b"PK\x05\x06")
    is_pdf = head[:4] == b"%PDF"

    print(f"gzip? {is_gzip}, zip? {is_zip}, pdf? {is_pdf}")

    if "<html" in lower_head and "recaptcha" in lower_head and "arxiv" in lower_head:
        print(
            "\nRESULT: Looks like an arXiv reCAPTCHA HTML challenge (NOT a source archive)."
        )
        # Optionally show a small snippet so you can confirm:
        print("--- HTML head snippet ---")
        print("\n".join(text_head.splitlines()[:15]))
        return

    if is_gzip or is_zip or is_pdf:
        print(
            "\nRESULT: Response looks like a binary archive (gzip/zip/pdf). Not reCAPTCHA."
        )
    else:
        print("\nRESULT: Response is not obvious archive magic and not reCAPTCHA HTML;")
        print("        could be plain text / HTML error / something else.")
        print("--- Head snippet ---")
        print("\n".join(text_head.splitlines()[:15]))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_recaptcha.py <arxiv_id>")
        print("Example: python check_recaptcha.py 2211.11689")
        sys.exit(1)

    arxiv_id = sys.argv[1]
    check_eprint(arxiv_id)
