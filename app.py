import math
import os
import threading
import traceback
import webbrowser
from typing import Literal

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from scipy.stats import norm

# ==============================================================================
# 1. CONFIGURATION & CONSTANTS
# ==============================================================================

N_ROWS = 4
N_COLS = 5
YEARS = [2023, 2024, 2025]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
HTML_CANDIDATES = [
    os.path.join(BASE_DIR, "kumaraswamy_layout_dashboard_v5.html"),
    os.path.join(DATA_DIR, "kumaraswamy_layout_dashboard_v5.html"),
    "kumaraswamy_layout_dashboard_v5.html",
]

ZONE_CONSERVATION_VALUES = [
    3, 5, 4, 9, 3,
    3, 4, 3, 5, 2,
    4, 7, 5, 10, 6,
    5, 6, 5, 6, 7,
]

N_MONTHS = 36
FORECAST_HORIZON = 12
ZONE_CAP_LAKH = 30.0
LAKE_THRESHOLD_PCT = 2.4


def _init_zone_parameters() -> tuple[np.ndarray, np.ndarray]:
    seed = 7
    k_vals = []
    cst_vals = []
    for _ in range(20):
        seed = (seed * 16807) % 2147483647
        r1 = seed / 2147483647.0
        seed = (seed * 16807) % 2147483647
        r2 = seed / 2147483647.0
        k_vals.append(0.07 + r1 * 0.10)
        cst_vals.append(0.80 + r2 * 0.60)
    return np.array(k_vals), np.array(cst_vals)


RESPONSE_RATES, COST_FACTORS = _init_zone_parameters()


# ==============================================================================
# 2. FORECASTING, RISK MODELING & CONVEX OPTIMIZATION ENGINE
# ==============================================================================

def fit_trend_seasonality(series: np.ndarray, horizon: int = FORECAST_HORIZON) -> dict:
    n = len(series)
    t = np.arange(n, dtype=float)

    b, a = np.polyfit(t, series, 1)
    detrended = series - (a + b * t)

    seasonality = np.array([detrended[m::12].mean() for m in range(12)])
    residuals = detrended - np.array([seasonality[i % 12] for i in range(n)])
    sd = math.sqrt(np.sum(residuals ** 2) / max(1, n - 2)) + 0.004

    future_t = np.arange(n, n + horizon)
    fc = np.maximum(0.0, a + b * future_t + np.array([seasonality[k % 12] for k in future_t]))
    ci_90 = [1.645 * sd * math.sqrt(1.0 + (i + 1) / n) for i in range(horizon)]

    return {
        "forecast": fc.tolist(),
        "ci_90": ci_90,
        "annual_slope": float(12.0 * b),
        "sd": float(sd),
    }


class AdvisoryEngine:
    def __init__(self):
        self._zones_base = None
        self._csv_mtime: float | None = None
        self._csv_path: str | None = None

    def _resolve_csv_path(self) -> str:
        candidates = [
            os.path.join(DATA_DIR, "zone_timeseries.csv"),
            os.path.join(BASE_DIR, "zone_timeseries.csv"),
            "data/zone_timeseries.csv",
            "zone_timeseries.csv",
        ]
        for path in candidates:
            if os.path.exists(path):
                self._csv_path = path
                return path
        raise FileNotFoundError(f"zone_timeseries.csv not found. Checked: {candidates}")

    def refresh_if_needed(self) -> bool:
        """Reloads zone data if zone_timeseries.csv was modified on disk."""
        csv_path = self._resolve_csv_path()
        mtime = os.path.getmtime(csv_path)
        if self._zones_base is None or self._csv_mtime != mtime:
            self._zones_base = self._load_data(csv_path)
            self._csv_mtime = mtime
            return True
        return False

    @property
    def zones_base(self) -> list[dict]:
        self.refresh_if_needed()
        return self._zones_base

    @property
    def data_version(self) -> str:
        self.refresh_if_needed()
        return f"{self._csv_mtime or 0:.6f}"

    def _load_data(self, csv_path: str) -> list[dict]:
        df = pd.read_csv(csv_path)
        df.columns = [str(c).strip().lower() for c in df.columns]

        zone_col = next((c for c in ["zone_id", "zone", "id", "grid_id"] if c in df.columns), None)
        ndvi_col = next((c for c in ["ndvi", "ndvi_mean", "mean_ndvi"] if c in df.columns), None)
        water_col = next((c for c in ["water_pct", "water", "water_share"] if c in df.columns), None)

        if not zone_col or not ndvi_col or not water_col:
            raise ValueError(f"CSV columns found: {list(df.columns)} — need zone, ndvi, and water columns.")

        if df[zone_col].dtype == object:
            df[zone_col] = df[zone_col].astype(str).str.extract(r"(\d+)")[0].astype(int)

        if df[zone_col].min() == 0:
            df[zone_col] = df[zone_col] + 1

        sort_cols = [c for c in [zone_col, "year", "month", "date"] if c in df.columns]
        df = df.sort_values(sort_cols).reset_index(drop=True)

        zones = []
        for idx in range(20):
            zone_id = idx + 1
            z_df = df[df[zone_col] == zone_id].copy()
            if len(z_df) < 36:
                raise ValueError(f"Zone {zone_id} has {len(z_df)} rows (expected 36).")

            ndvi_s = pd.to_numeric(z_df[ndvi_col], errors="coerce").interpolate(limit_direction="both").fillna(0.2)
            water_s = pd.to_numeric(z_df[water_col], errors="coerce").interpolate(limit_direction="both").fillna(0.0)

            ndvi_hist = ndvi_s.to_numpy(dtype=float)[:36]
            water_hist = water_s.to_numpy(dtype=float)[:36]

            sorted_w = np.sort(water_hist)
            median_w = 0.5 * (sorted_w[17] + sorted_w[18])
            is_lake = bool(median_w >= LAKE_THRESHOLD_PCT)

            zones.append({
                "zone_id": zone_id,
                "name": f"Zone {zone_id}",
                "row": idx // 5,
                "col": idx % 5,
                "ndvi_history": ndvi_hist,
                "water_history": water_hist,
                "ndvi_fit": fit_trend_seasonality(ndvi_hist),
                "water_fit": fit_trend_seasonality(water_hist),
                "V": ZONE_CONSERVATION_VALUES[idx],
                "k": float(RESPONSE_RATES[idx]),
                "cost_factor": float(COST_FACTORS[idx]),
                "cap": ZONE_CAP_LAKH,
                "is_lake": is_lake,
            })
        return zones

    def simulate(
        self,
        budget: float = 120.0,
        uncertainty: float = 1.0,
        tau: float = 0.18,
        water_loss_pct: float = 15.0,
    ) -> dict:
        self.refresh_if_needed()
        scale_factor = math.sqrt(1.0 + FORECAST_HORIZON / N_MONTHS)
        results = []

        weights = np.zeros(20)
        k_arr = np.zeros(20)
        cst_arr = np.zeros(20)
        cap_arr = np.zeros(20)

        for i, z in enumerate(self._zones_base):
            ndvi_now = float(np.mean(z["ndvi_history"][-3:]))
            ndvi_mu = float(np.mean(z["ndvi_fit"]["forecast"]))
            ndvi_s = z["ndvi_fit"]["sd"] * uncertainty * scale_factor

            water_now = float(np.mean(z["water_history"][-3:]))
            water_mu = float(np.mean(z["water_fit"]["forecast"]))
            water_s = z["water_fit"]["sd"] * uncertainty * scale_factor + 0.05
            water_2023_mean = float(np.mean(z["water_history"][:12]))

            r_green = float(norm.cdf((tau - ndvi_mu) / ndvi_s))
            if z["is_lake"]:
                alert_level = (1.0 - water_loss_pct / 100.0) * water_2023_mean
                r_water = float(norm.cdf((alert_level - water_mu) / water_s))
            else:
                r_water = 0.0

            r_combined = 1.0 - (1.0 - r_green) * (1.0 - r_water)
            wt = z["V"] * (0.15 + r_combined)

            weights[i] = wt
            k_arr[i] = z["k"]
            cst_arr[i] = z["cost_factor"]
            cap_arr[i] = z["cap"]

            water_change_pct = ((water_mu - water_2023_mean) / water_2023_mean * 100.0) if water_2023_mean > 0 else 0.0

            results.append({
                "zone_id": z["zone_id"],
                "name": z["name"],
                "row": z["row"],
                "col": z["col"],
                "is_lake": z["is_lake"],
                "conservation_value_V": z["V"],
                "cost_factor": round(z["cost_factor"], 3),
                "response_rate_k": round(z["k"], 4),
                "ndvi_now": round(ndvi_now, 4),
                "ndvi_forecast_mean": round(ndvi_mu, 4),
                "ndvi_forecast_ci90": round(1.645 * ndvi_s, 4),
                "ndvi_annual_slope": round(z["ndvi_fit"]["annual_slope"], 4),
                "water_now_pct": round(water_now, 3),
                "water_2023_mean_pct": round(water_2023_mean, 3),
                "water_forecast_mean_pct": round(water_mu, 3),
                "water_change_pct": round(water_change_pct, 1),
                "risk_green": round(r_green, 4),
                "risk_water": round(r_water, 4),
                "risk_combined": round(r_combined, 4),
                "ndvi_history": z["ndvi_history"].tolist(),
                "ndvi_forecast": z["ndvi_fit"]["forecast"],
                "ndvi_ci90_band": z["ndvi_fit"]["ci_90"],
                "water_history": z["water_history"].tolist(),
                "water_forecast": z["water_fit"]["forecast"],
                "water_ci90_band": z["water_fit"]["ci_90"],
            })

        def allocation_for_lambda(lam: float) -> np.ndarray:
            raw_x = np.log(np.maximum(1e-12, (weights * k_arr) / (lam * cst_arr))) / k_arr
            return np.clip(raw_x, 0.0, cap_arr)

        def expected_benefit(x: np.ndarray) -> float:
            return float(np.sum(weights * (1.0 - np.exp(-k_arr * x))))

        lo, hi = 1e-6, 50.0
        for _ in range(60):
            mid = math.sqrt(lo * hi)
            if np.sum(allocation_for_lambda(mid) * cst_arr) > budget:
                lo = mid
            else:
                hi = mid

        opt_x = allocation_for_lambda(hi)
        opt_benefit = expected_benefit(opt_x)

        eq_x = np.minimum(cap_arr, budget / np.sum(cst_arr))
        eq_benefit = expected_benefit(eq_x)

        rv_weights = np.array([r["risk_combined"] * r["conservation_value_V"] for r in results])
        total_rv = np.sum(rv_weights)
        rp_x = np.minimum(cap_arr, (budget * rv_weights / total_rv) / cst_arr) if total_rv > 0 else np.zeros(20)
        rp_benefit = expected_benefit(rp_x)

        budget_curve = []
        for b_step in range(0, 405, 10):
            if b_step == 0:
                budget_curve.append({"budget": 0, "benefit": 0.0})
                continue
            b_lo, b_hi = 1e-6, 50.0
            for _ in range(50):
                b_mid = math.sqrt(b_lo * b_hi)
                if np.sum(allocation_for_lambda(b_mid) * cst_arr) > b_step:
                    b_lo = b_mid
                else:
                    b_hi = b_mid
            budget_curve.append({
                "budget": b_step,
                "benefit": round(expected_benefit(allocation_for_lambda(b_hi)), 3),
            })

        for i, r in enumerate(results):
            spend_i = float(opt_x[i] * cst_arr[i])
            r["allocation_lakh"] = round(float(opt_x[i]), 2)
            r["actual_spend_lakh"] = round(spend_i, 2)
            r["budget_share_pct"] = round((spend_i / budget * 100.0) if budget > 0 else 0.0, 1)

        ranked_zones = sorted(results, key=lambda z: (-z["allocation_lakh"], -z["risk_combined"]))

        kpis = {
            "budget_lakh": budget,
            "shadow_price_lambda": round(hi, 4),
            "expected_benefit_optimal": round(opt_benefit, 2),
            "expected_benefit_equal_split": round(eq_benefit, 2),
            "expected_benefit_risk_proportional": round(rp_benefit, 2),
            "gain_vs_equal_pct": round(((opt_benefit / eq_benefit) - 1.0) * 100.0, 1) if eq_benefit > 0 else 0.0,
            "gain_vs_risk_prop_pct": round(((opt_benefit / rp_benefit) - 1.0) * 100.0, 1) if rp_benefit > 0 else 0.0,
            "zones_funded_count": sum(1 for r in results if r["allocation_lakh"] > 0.5),
            "high_risk_zones_count": sum(1 for r in results if r["risk_combined"] > 0.5),
            "lake_zones_at_risk_count": sum(1 for r in results if r["is_lake"] and r["risk_water"] > 0.5),
        }

        return {
            "data_version": self.data_version,
            "kpis": kpis,
            "zones": results,
            "top_10_ranked": ranked_zones[:10],
            "budget_curve": budget_curve,
        }


# ==============================================================================
# 3. FASTAPI REST SERVER & ADVISORY GENERATOR
# ==============================================================================

app = FastAPI(
    title="Water and Green Cover Advisory API: Kumaraswamy Layout, Bengaluru",
    version="1.1.0",
    description="Backend optimization, Sentinel-2 time-series forecasting, and advisory service.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def debug_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error_type": type(exc).__name__,
            "error_detail": str(exc),
            "traceback": traceback.format_exc().splitlines()[-5:],
        },
    )


engine = AdvisoryEngine()


def _resolve_html_path() -> str:
    for path in HTML_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"kumaraswamy_layout_dashboard_v5.html not found. Checked: {HTML_CANDIDATES}")


class SimulationParams(BaseModel):
    budget: float = Field(default=120.0, ge=0.0, le=400.0)
    uncertainty: float = Field(default=1.0, ge=0.5, le=2.0)
    tau: float = Field(default=0.18, ge=0.10, le=0.35)
    water_loss_pct: float = Field(default=15.0, ge=5.0, le=40.0)


class AdvisoryRequest(SimulationParams):
    zone_id: int = Field(..., ge=1, le=20)
    use_llm: bool = Field(default=True)


def _rule_based_advisory(zone: dict, params: SimulationParams) -> list[str]:
    items = []
    if zone["risk_green"] > 0.5:
        items.append(
            f"Green cover is at risk: forecast NDVI {zone['ndvi_forecast_mean']:.2f} is near or below "
            f"tau = {params.tau:.2f} ({round(zone['risk_green'] * 100)}% risk). "
            "Plant native canopy trees and protect remaining tree patches."
        )
    else:
        items.append(
            f"Green cover is relatively stable ({round(zone['risk_green'] * 100)}% risk of falling below tau). "
            "Keep existing trees and re-check each season."
        )

    if zone["is_lake"]:
        if zone["risk_water"] > 0.3:
            direction = "below" if zone["water_change_pct"] < 0 else "above"
            items.append(
                f"Water area is projected to be {abs(round(zone['water_change_pct']))}% {direction} its 2023 level "
                f"({round(zone['risk_water'] * 100)}% risk). Restore the lake edge buffer, clear inlet drains "
                "before the monsoon, and add recharge pits in nearby layouts."
            )
        else:
            items.append(
                f"Water area looks stable ({round(zone['risk_water'] * 100)}% risk of crossing the "
                f"{round(params.water_loss_pct)}% loss alert). Keep drains and the lake edge clear."
            )
    else:
        items.append(
            "No significant surface water here. Rainwater harvesting and permeable ground in open plots "
            "will help local recharge."
        )

    items.append(
        f"Suggested budget share: Rs {zone['allocation_lakh']:.1f} lakh "
        f"({round(zone['budget_share_pct'])}% of the total)."
    )
    return items


@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    html_path = _resolve_html_path()
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/health")
def health_check():
    return {"status": "ok", "zones_loaded": len(engine.zones_base), "data_version": engine.data_version}


@app.get("/api/status")
def get_status():
    reloaded = engine.refresh_if_needed()
    return {
        "status": "ok",
        "reloaded": reloaded,
        "data_version": engine.data_version,
        "zones_count": len(engine.zones_base),
    }


@app.get("/api/simulate")
def run_simulation_get(
    budget: float = Query(120.0, ge=0.0, le=400.0),
    uncertainty: float = Query(1.0, ge=0.5, le=2.0),
    tau: float = Query(0.18, ge=0.10, le=0.35),
    water_loss_pct: float = Query(15.0, ge=5.0, le=40.0),
):
    return engine.simulate(budget=budget, uncertainty=uncertainty, tau=tau, water_loss_pct=water_loss_pct)


@app.post("/api/simulate")
def run_simulation_post(params: SimulationParams):
    return engine.simulate(budget=params.budget, uncertainty=params.uncertainty, tau=params.tau, water_loss_pct=params.water_loss_pct)


@app.get("/api/zones/{zone_id}")
def get_zone_detail(
    zone_id: int,
    budget: float = Query(120.0, ge=0.0, le=400.0),
    uncertainty: float = Query(1.0, ge=0.5, le=2.0),
    tau: float = Query(0.18, ge=0.10, le=0.35),
    water_loss_pct: float = Query(15.0, ge=5.0, le=40.0),
):
    if zone_id < 1 or zone_id > 20:
        raise HTTPException(status_code=404, detail="Zone ID must be between 1 and 20.")
    sim = engine.simulate(budget=budget, uncertainty=uncertainty, tau=tau, water_loss_pct=water_loss_pct)
    return sim["zones"][zone_id - 1]


@app.post("/api/advisory")
def generate_advisory(req: AdvisoryRequest):
    sim = engine.simulate(budget=req.budget, uncertainty=req.uncertainty, tau=req.tau, water_loss_pct=req.water_loss_pct)
    zone = sim["zones"][req.zone_id - 1]
    bullets = _rule_based_advisory(zone, req)
    return {
        "zone_id": zone["zone_id"],
        "zone_name": zone["name"],
        "metrics": {
            "risk_green": zone["risk_green"],
            "risk_water": zone["risk_water"],
            "risk_combined": zone["risk_combined"],
            "allocation_lakh": zone["allocation_lakh"],
            "budget_share_pct": zone["budget_share_pct"],
        },
        "structured_bullets": bullets,
    }


@app.get("/api/satellite/{year}/{band}")
def get_satellite_composite(
    year: Literal[2023, 2024, 2025],
    band: Literal["truecolor", "falsecolor", "ndvi", "water"],
):
    for folder in [DATA_DIR, BASE_DIR]:
        file_path = os.path.join(folder, f"{year}_{band}.png")
        if os.path.exists(file_path):
            return FileResponse(file_path, media_type="image/png")
    raise HTTPException(status_code=404, detail=f"Image {year}_{band}.png not found in {DATA_DIR} or {BASE_DIR}.")


def _open_browser_later(url: str, delay: float = 1.0):
    if os.getenv("NO_BROWSER") == "1":
        return
    timer = threading.Timer(delay, lambda: webbrowser.open(url))
    timer.daemon = True
    timer.start()


if __name__ == "__main__":
    _open_browser_later("http://localhost:8000", delay=1.0)
    uvicorn.run(app, host="127.0.0.1", port=8000)
