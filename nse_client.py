# nse_client.py
import time
import requests
from datetime import datetime
from typing import Optional, Dict, Any, List

NSE_BASE = "https://www.nseindia.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/"
}

def _to_iso(date_str: str) -> str:
    # NSE format: '18-Sep-2025' -> '2025-09-18'
    try:
        return datetime.strptime(date_str, "%d-%b-%Y").date().isoformat()
    except Exception:
        return date_str

def _expiries_to_iso(expiry_list: List[str]) -> List[str]:
    out = []
    for s in expiry_list or []:
        iso = _to_iso(s)
        if len(iso) == 10:
            out.append(iso)
    return sorted(out)

class NseClient:
    """
    Minimal NSE client to fetch real option premiums (LTP) for index options.
    Uses public option-chain endpoint with proper headers/cookies.
    """
    def __init__(self, index_symbol: str = "NIFTY"):
        self.index_symbol = index_symbol.upper()   # "NIFTY", "BANKNIFTY", "FINNIFTY", ...
        self.sess = requests.Session()
        self.sess.headers.update(HEADERS)
        self._last_cookie = 0.0

    def _prime(self):
        # refresh cookies ~every 60s to avoid 401s
        if time.time() - self._last_cookie < 60:
            return
        self.sess.get(NSE_BASE, timeout=6)
        self._last_cookie = time.time()

    def get_option_chain(self, underlying: Optional[str] = None) -> Dict[str, Any]:
        self._prime()
        u = (underlying or self.index_symbol).upper()
        url = f"{NSE_BASE}/api/option-chain-indices?symbol={u}"
        r = self.sess.get(url, timeout=8)
        r.raise_for_status()
        return r.json()

    def get_nearest_expiry_iso(self) -> Optional[str]:
        data = self.get_option_chain()
        expiries = (data.get("records") or {}).get("expiryDates") or []
        iso = _expiries_to_iso(expiries)
        today = datetime.now().date().isoformat()
        for d in iso:
            if d >= today:
                return d
        return iso[0] if iso else None

    def get_option_ltp(self, strike: int, is_call: bool, expiry_iso: str) -> Optional[float]:
        """
        Return last traded price (LTP) for the given strike/side/expiry.
        NEVER uses OI; we only read lastPrice (and closePrice fallback).
        """
        data = self.get_option_chain()
        rows = (data.get("records") or {}).get("data", []) or []
        side = "CE" if is_call else "PE"

        for row in rows:
            if int(row.get("strikePrice", 0)) != int(strike):
                continue
            if _to_iso(row.get("expiryDate", "")) != expiry_iso:
                continue
            opt = row.get(side, {}) or {}
            lp = opt.get("lastPrice") or opt.get("closePrice")
            return float(lp) if lp not in (None, 0) else None

        return None
