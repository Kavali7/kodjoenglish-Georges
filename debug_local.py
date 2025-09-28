import os
import json
import pandas as pd
from datetime import date

from ghl_client import build_dashboard_dataframe, GHLSettings


class LocalJsonClient:
    def __init__(self, base_dir: str = "."):
        self.base_dir = base_dir

    def _load(self, filename: str, keys: list[str]):
        path = os.path.join(self.base_dir, filename)
        if not os.path.exists(path):
            return []
        with open(path, "rb") as f:
            raw = f.read()
        text = None
        for enc in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be"):
            try:
                text = raw.decode(enc)
                break
            except Exception:
                continue
        if text is None:
            text = raw.decode("latin-1", errors="ignore")
        data = json.loads(text)
        if isinstance(data, dict):
            for k in keys:
                if k in data and isinstance(data[k], list):
                    return data[k]
        if isinstance(data, list):
            return data
        return []

    def get_contacts(self, start, end):
        return self._load("contacts.json", ["contacts", "items"])

    def get_opportunities(self, start, end):
        return self._load("opportunities.json", ["opportunities", "items"])

    def get_forms(self, start, end):
        return self._load("forms.json", ["forms", "submissions", "items"])

    def get_messages(self, start, end):
        return self._load("messages.json", ["messages", "items"])

    def get_payments(self, start, end):
        return self._load("payments.json", ["transactions", "items"])

    def get_calls(self, start, end):
        return self._load("calls.json", ["calls", "items"])


def main():
    # Minimal settings to satisfy build; secrets are not used here
    settings = GHLSettings(
        api_key="dummy",
        timezone="Africa/Porto-Novo",
        user_mapping={},
        source_mapping={
            "facebook": "From_FB",
            "instagram": "From_Insta",
            "tiktok": "From_TikTok",
            "linkedin": "From_LinkedIn",
            "forms": "From_Forms",
            "direct": "From_Direct",
        },
        history_days=7,
    )
    client = LocalJsonClient()
    df = build_dashboard_dataframe(client, settings)
    print("rows:", len(df), "cols:", list(df.columns))
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
