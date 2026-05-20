from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import datetime, timezone
import time
import random
from typing import Any, Literal, Optional

import requests
from urllib.parse import urlsplit


def _iso_utc_now() -> str:
    # OKX expects RFC3339 / ISO8601 timestamp with Z, milliseconds precision.
    return datetime.now(tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass
class OKXCredentials:
    api_key: str
    secret_key: str
    passphrase: str


@dataclass
class OKXClient:
    creds: OKXCredentials
    base_url: str = "https://www.okx.com"
    simulated: bool = True
    timeout_sec: int = 20
    max_retries: int = 5

    def _sign(self, ts: str, method: str, request_path: str, body: str) -> str:
        prehash = f"{ts}{method.upper()}{request_path}{body}"
        mac = hmac.new(self.creds.secret_key.encode("utf-8"), prehash.encode("utf-8"), hashlib.sha256).digest()
        return base64.b64encode(mac).decode("utf-8")

    def _headers(self, ts: str, sign: str) -> dict[str, str]:
        h = {
            "OK-ACCESS-KEY": self.creds.api_key,
            "OK-ACCESS-SIGN": sign,
            "OK-ACCESS-TIMESTAMP": ts,
            "OK-ACCESS-PASSPHRASE": self.creds.passphrase,
            "Content-Type": "application/json",
        }
        if self.simulated:
            h["x-simulated-trading"] = "1"
        return h

    def _request(
        self,
        method: Literal["GET", "POST"],
        path: str,
        params: Optional[dict[str, Any]] = None,
        body: Optional[dict[str, Any]] = None,
        auth: bool = True,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        params = params or {}
        body = body or {}

        body_str = json.dumps(body, separators=(",", ":"), ensure_ascii=False) if method == "POST" else ""
        ts = _iso_utc_now()

        # Prepare request URL so signing uses the exact path/query encoding.
        s = requests.Session()
        req = requests.Request(method=method, url=url, params=(params if method == "GET" else None), data=(body_str if method == "POST" else None))
        prepped = s.prepare_request(req)
        sp = urlsplit(prepped.url)
        request_path = sp.path + (("?" + sp.query) if sp.query else "")

        headers = {}
        if auth:
            sign = self._sign(ts, method, request_path, body_str)
            headers = self._headers(ts, sign)

        try:
            last_err: Exception | None = None
            for attempt in range(self.max_retries):
                if method == "GET":
                    resp = requests.get(url, params=params, headers=headers, timeout=self.timeout_sec)
                else:
                    resp = requests.post(url, data=body_str.encode("utf-8"), headers=headers, timeout=self.timeout_sec)

                if resp.status_code // 100 == 2:
                    return resp.json()

                try:
                    err = resp.json()
                except Exception:
                    err = {"raw": resp.text}

                code = str(err.get("code", ""))
                # OKX rate limit: HTTP 429 + code 50011
                is_rl = (resp.status_code == 429) or (code == "50011")
                if is_rl and attempt < (self.max_retries - 1):
                    # Exponential backoff with jitter
                    base = 0.5 * (2**attempt)
                    sleep_s = min(8.0, base + random.uniform(0, 0.25))
                    time.sleep(sleep_s)
                    continue

                last_err = RuntimeError(f"OKX HTTP {resp.status_code} {request_path} => {err}")
                break

            raise last_err or RuntimeError(f"OKX HTTP error {request_path}")
        finally:
            s.close()

    def public_time(self) -> dict[str, Any]:
        return self._request("GET", "/api/v5/public/time", auth=False)

    def get_instruments(self, inst_type: str, inst_id: Optional[str] = None) -> dict[str, Any]:
        params: dict[str, Any] = {"instType": inst_type}
        if inst_id:
            params["instId"] = inst_id
        return self._request("GET", "/api/v5/public/instruments", params=params, auth=False)

    def get_positions(self, inst_type: str = "SWAP", inst_id: Optional[str] = None) -> dict[str, Any]:
        params: dict[str, Any] = {"instType": inst_type}
        if inst_id:
            params["instId"] = inst_id
        return self._request("GET", "/api/v5/account/positions", params=params, auth=True)

    def get_account_config(self) -> dict[str, Any]:
        return self._request("GET", "/api/v5/account/config", auth=True)

    def get_balance(self, ccy: str = "USDT") -> dict[str, Any]:
        return self._request("GET", "/api/v5/account/balance", params={"ccy": ccy}, auth=True)

    def set_leverage(
        self,
        inst_id: str,
        lever: int,
        mgn_mode: str = "isolated",
        pos_side: Optional[str] = None,
        ccy: Optional[str] = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"instId": inst_id, "lever": str(int(lever)), "mgnMode": mgn_mode}
        if pos_side:
            body["posSide"] = pos_side
        if ccy:
            body["ccy"] = ccy
        return self._request("POST", "/api/v5/account/set-leverage", body=body, auth=True)

    def place_order(
        self,
        inst_id: str,
        td_mode: str,
        side: Literal["buy", "sell"],
        ord_type: str,
        sz: str,
        pos_side: Optional[str] = None,
        reduce_only: Optional[bool] = None,
        lever: Optional[int] = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"instId": inst_id, "tdMode": td_mode, "side": side, "ordType": ord_type, "sz": sz}
        if pos_side:
            body["posSide"] = pos_side
        if reduce_only is not None:
            body["reduceOnly"] = "true" if reduce_only else "false"
        if lever is not None:
            body["lever"] = str(int(lever))
        return self._request("POST", "/api/v5/trade/order", body=body, auth=True)
