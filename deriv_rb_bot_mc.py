"""
+=========================================================================+
|  DERIV RANGE BOT  v11  —  Multi-Symbol  (Monte Carlo + Jump-Diffusion)  |
|                                                                         |
|  All intelligence is a dual Monte Carlo containment estimator:         |
|                                                                         |
|  1. Pure GBM paths  (baseline)                                         |
|       Each tick: dlog S = σ · Z,  Z ~ N(0,1)                          |
|       σ estimated via EWMA (α=0.06, λ=0.94)                           |
|                                                                         |
|  2. Merton Jump-Diffusion paths  (jump-aware)                          |
|       Each tick: dlog S = σ · Z + J · Y                               |
|         J ~ Poisson(λ_tick)      — does a jump occur this tick?       |
|         Y ~ N(μ_J, σ_J²)        — how big is the jump?               |
|       Parameters estimated live from recent tick history:              |
|         λ  = jump rate (jumps/tick), fitted from outlier frequency     |
|         μ_J = mean jump log-return (signed; usually ≈ 0)              |
|         σ_J = jump size std dev (always > diffusion σ)                |
|       Jump detection: returns whose |r| > jump_threshold·σ_ewma       |
|       are classified as jumps; the rest are pure diffusion residuals.  |
|                                                                         |
|  3. Blended containment probability                                     |
|       p_final = (1 - w) · p_gbm  +  w · p_jd                         |
|       w = mc_jd_weight (default 0.5).                                  |
|       Falls back to pure GBM when fewer than mc_jd_min_jumps           |
|       have been observed (not enough data to fit jump parameters).     |
|                                                                         |
|  4. p_final feeds EV gate → persistence → KellyStaker → SPRT          |
|                                                                         |
|  Requirements: pip install numpy websocket-client pandas               |
|  Env:          DERIV_API_TOKEN                                          |
+=========================================================================+
"""

import csv, enum, json, logging, math, os, random, sys, threading, time
import io
from collections import deque
from datetime import datetime

import numpy as np
import websocket
import pandas as pd


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_sh = logging.StreamHandler(
    io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace"))
_sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
_fh = logging.FileHandler("deriv_rb_bot_mc.log", encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[_sh, _fh])
log = logging.getLogger("DerivRB_MC")

DATA_DIR = os.environ.get("DATA_DIR", "rb_bot_data")
os.makedirs(DATA_DIR, exist_ok=True)


# ===========================================================================
# CONFIGURATION
# ===========================================================================
CONFIG = {
    # -- Deriv -----------------------------------------------------------------
    "app_id"    : 1089,
    "api_token" : os.environ.get("DERIV_API_TOKEN", ""),

    # -- Symbols ---------------------------------------------------------------
    # Bot watches ALL symbols simultaneously via independent workers.
    # Only the symbol with the highest p_blend trades at any given time.
    # One open trade at a time globally.
    "symbols"   : ["RDBEAR", "1HZ25V", "R_10", "R_25", "R_50", "R_75", "R_100"],
    "symbol"    : "RDBEAR",   # legacy key — kept for calibration logging compat

    # -- Tick collection -------------------------------------------------------
    "collect_hours" : 0.6,          # 30 min history
    "data_dir"      : os.path.join(DATA_DIR, "tick_data"),
    "min_ticks"     : 500,          # ~90s warmup on RDBEAR (was 500 = ~2.5min)

    # -- Expiry choices (minutes) ----------------------------------------------
    "hold_durations" : [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15],

    # -- Barrier fallback (used until calibration finishes) -------------------
    "expiryrange_barrier" : 2.97,
    "currency"            : "USD",

    # -- Monte Carlo -----------------------------------------------------------
    "mc_n_paths"          : 5000,    # raised: std error ≈ ±0.006 (was ±0.014 at 1000 paths)
    "mc_vol_window"       : 60,      # ticks used to compute EWMA σ
    "mc_ewma_alpha"       : 0.06,    # EWMA decay (RiskMetrics λ=0.94)
    "mc_ticks_per_min"    : None,    # None = auto-detect from tick timestamps
    # p_contain = fraction of paths staying inside barrier.
    # Minimum threshold before considering EV.
    "mc_p_floor"          : 0.670,   # slightly below break-even; ev_thr sanity check protects
    # Containment check mode:
    #   terminal (1.0) = final tick only  — matches Deriv EXPIRYRANGE settlement
    #   path     (0.0) = every tick       — strict first-passage check
    #   blend    (0.7) = 70% terminal + 30% path
    # RDBEAR EXPIRYRANGE settles on the LAST tick only. Pure terminal (1.0)
    # is the correct model. Path component (< 1.0) is an optional conservative
    # penalty — useful only if you want to avoid contracts where mid-breach risk
    # is elevated (e.g. during high-jump regimes). Start with 1.0.
    "mc_terminal_weight"  : 1.0,

    # -- Jump-Diffusion (Merton) -----------------------------------------------
    # Jump detection: any log-return whose |r| > jump_threshold * σ_ewma
    # is classified as a jump tick (not pure diffusion).
    "jd_jump_threshold" : 3.0,    # σ-multiples above which a return is a jump
    "jd_fit_window"     : 300,    # ticks of history used to fit jump params
    "jd_min_jumps"      : 5,      # minimum observed jumps needed to enable JD paths
    # Blend weight: p_final = (1-w)*p_gbm + w*p_jd
    # 0.0 = pure GBM,  1.0 = pure JD,  0.5 = equal blend
    "jd_weight"         : 0.5,

    # -- EV gate ---------------------------------------------------------------
    "ev_confidence_floor"  : 0.673,  # 1/1.486 — calibrated to actual 48.6% payout
    "min_ev_threshold"     : 0.001,  # sanity only — p_floor already ensures positive EV
    "min_payout_pct"       : 0.48,
    "ev_check_on_proposal" : True,

    # -- Signal persistence (consecutive ticks above p_floor before firing) ---
    "signal_persistence_ticks" : 2,   # lowered: 3 consecutive passes too rare on RDBEAR

    # -- Post-trade cooldown ---------------------------------------------------
    "min_ticks_between_trades" : 60,

    # -- Kelly staking ---------------------------------------------------------
    "kelly_fraction"  : 0.25,
    "kelly_max_pct"   : 0.10,
    "kelly_min_stake" : 0.35,
    "kelly_max_stake" : 5.0,
    "stake_pct"       : 0.35,

    # -- Risk limits -----------------------------------------------------------
    "max_daily_loss_pct"        : 0.80,
    "take_profit_pct"           : 9999.0,
    "max_drawdown_from_peak_pct": 0.80,

    # -- Consecutive-loss cooldown ---------------------------------------------
    "max_consec_losses"          : 5,
    "consec_loss_cooldown_ticks" : 60,

    # -- SPRT ------------------------------------------------------------------
    "sprt_p0"    : 0.50,
    "sprt_p1"    : 0.53,
    "sprt_alpha" : 0.10,
    "sprt_beta"  : 0.20,

    # -- Trade log -------------------------------------------------------------
    "trade_log" : os.path.join(DATA_DIR, "trade_log.csv"),
}

os.makedirs(CONFIG["data_dir"], exist_ok=True)


# ===========================================================================
# SYMBOL PROFILES
# Per-symbol calibration search bounds and tick rate hints.
# barrier_bounds: {dur: (lo, hi, payout_lo, payout_hi)}
# tpm_hint: approximate ticks/minute (None = auto-detect)
# min_ticks: warmup ticks before live evaluation starts
# ===========================================================================
SYMBOL_PROFILES = {
    "RDBEAR": {
        "tpm_hint"  : None,   # auto-detect (~200/min)
        "min_ticks" : 300,
        "barrier_bounds": {
            2:(0.50,5.0,0.46,0.50), 3:(0.80,6.0,0.46,0.50),
            4:(1.00,7.0,0.46,0.50), 5:(1.20,8.0,0.46,0.50),
            6:(1.40,9.0,0.46,0.50), 7:(1.60,10.5,0.46,0.50),
            8:(1.80,12.0,0.46,0.50), 9:(2.00,13.5,0.46,0.50),
            10:(2.20,15.0,0.46,0.50), 12:(2.60,17.5,0.46,0.50),
            15:(3.00,21.0,0.46,0.50),
        },
    },
    "1HZ25V": {
        "tpm_hint"  : 60,     # 1 tick/sec
        "min_ticks" : 150,
        "barrier_bounds": {
            2:(0.20,3.0,0.47,0.495), 3:(0.30,4.0,0.46,0.50),
            4:(0.40,5.0,0.46,0.50), 5:(0.50,6.0,0.46,0.50),
            6:(0.60,7.0,0.46,0.50), 7:(0.70,8.5,0.46,0.50),
            8:(0.80,10.0,0.46,0.50), 9:(0.90,11.5,0.46,0.50),
            10:(1.00,13.0,0.46,0.50), 12:(1.20,15.0,0.46,0.50),
            15:(1.50,18.0,0.46,0.50),
        },
    },
    "R_10": {
        "tpm_hint"  : None,
        "min_ticks" : 300,
        "barrier_bounds": {
            2:(0.03,0.40,0.46,0.50), 3:(0.05,0.55,0.46,0.50),
            4:(0.06,0.70,0.46,0.50), 5:(0.08,0.85,0.46,0.50),
            6:(0.09,1.00,0.46,0.50), 7:(0.10,1.15,0.46,0.50),
            8:(0.12,1.30,0.46,0.50), 9:(0.13,1.45,0.46,0.50),
            10:(0.15,1.60,0.46,0.50), 12:(0.17,1.90,0.46,0.50),
            15:(0.20,2.30,0.46,0.50),
        },
    },
    "R_25": {
        "tpm_hint"  : None,
        "min_ticks" : 300,
        "barrier_bounds": {
            2:(0.08,1.0,0.46,0.50), 3:(0.12,1.4,0.46,0.50),
            4:(0.15,1.8,0.46,0.50), 5:(0.18,2.2,0.46,0.50),
            6:(0.22,2.6,0.46,0.50), 7:(0.25,3.0,0.46,0.50),
            8:(0.28,3.4,0.46,0.50), 9:(0.32,3.8,0.46,0.50),
            10:(0.35,4.2,0.46,0.50), 12:(0.42,5.0,0.46,0.50),
            15:(0.50,6.0,0.46,0.50),
        },
    },
    "R_50": {
        "tpm_hint"  : None,
        "min_ticks" : 300,
        "barrier_bounds": {
            2:(0.15,2.0,0.46,0.50), 3:(0.22,2.8,0.46,0.50),
            4:(0.28,3.5,0.46,0.50), 5:(0.35,4.3,0.46,0.50),
            6:(0.42,5.0,0.46,0.50), 7:(0.48,5.8,0.46,0.50),
            8:(0.55,6.6,0.46,0.50), 9:(0.62,7.4,0.46,0.50),
            10:(0.68,8.0,0.46,0.50), 12:(0.80,9.5,0.46,0.50),
            15:(1.00,11.5,0.46,0.50),
        },
    },
    "R_75": {
        "tpm_hint"  : None,
        "min_ticks" : 300,
        "barrier_bounds": {
            2:(0.25,3.0,0.46,0.50), 3:(0.35,4.2,0.46,0.50),
            4:(0.45,5.5,0.46,0.50), 5:(0.55,6.5,0.46,0.50),
            6:(0.65,7.8,0.46,0.50), 7:(0.75,9.0,0.46,0.50),
            8:(0.85,10.2,0.46,0.50), 9:(0.95,11.5,0.46,0.50),
            10:(1.05,12.5,0.46,0.50), 12:(1.25,15.0,0.46,0.50),
            15:(1.50,18.0,0.46,0.50),
        },
    },
    "R_100": {
        "tpm_hint"  : None,
        "min_ticks" : 300,
        "barrier_bounds": {
            2:(0.35,4.5,0.46,0.50), 3:(0.50,6.0,0.46,0.50),
            4:(0.65,8.0,0.46,0.50), 5:(0.80,9.5,0.46,0.50),
            6:(0.95,11.0,0.46,0.50), 7:(1.10,13.0,0.46,0.50),
            8:(1.25,14.5,0.46,0.50), 9:(1.40,16.5,0.46,0.50),
            10:(1.55,18.0,0.46,0.50), 12:(1.80,21.0,0.46,0.50),
            15:(2.20,25.0,0.46,0.50),
        },
    },
}
# Fallback profile for any symbol not listed
_DEFAULT_PROFILE = SYMBOL_PROFILES["RDBEAR"]
def get_profile(symbol: str) -> dict:
    return SYMBOL_PROFILES.get(symbol, _DEFAULT_PROFILE)



# ===========================================================================
# UTILITIES
# ===========================================================================

def wilson_ci(wins, n, z=1.96):
    if n == 0: return 0.0, 1.0
    p   = wins / n
    denom = 1 + z**2/n
    centre = (p + z**2/(2*n)) / denom
    margin = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


# ===========================================================================
# SPRT MONITOR
# ===========================================================================

class SPRTMonitor:
    """Sequential Probability Ratio Test — passive edge tracker."""
    def __init__(self, p0=0.50, p1=0.53, alpha=0.10, beta=0.20):
        self.A   = math.log((1-beta)/alpha)
        self.B   = math.log(beta/(1-alpha))
        self.p0  = p0; self.p1 = p1
        self.llr = 0.0; self.n = 0; self.wins = 0
        self.status = "CONTINUE"

    def update(self, win: bool) -> str:
        self.n += 1
        if win:
            self.wins += 1
            self.llr  += math.log(self.p1/self.p0)
        else:
            self.llr  += math.log((1-self.p1)/(1-self.p0))
        if   self.llr >= self.A: self.status = "ACCEPT_H1"; self.llr = 0.0
        elif self.llr <= self.B: self.status = "ACCEPT_H0"; self.llr = 0.0
        else:                    self.status = "CONTINUE"
        return self.status

    def summary(self):
        wr     = self.wins/self.n if self.n else 0.0
        lo, hi = wilson_ci(self.wins, self.n)
        return f"{self.status}  n={self.n}  WR={wr:.3f}  CI=[{lo:.3f},{hi:.3f}]"


# ===========================================================================
# MONTE CARLO PRICER  (GBM  +  Merton Jump-Diffusion)
# ===========================================================================

class JumpParams:
    """Container for fitted Merton jump-diffusion parameters."""
    __slots__ = ("lam", "mu_j", "sigma_j", "n_jumps", "n_obs")

    def __init__(self, lam=0.0, mu_j=0.0, sigma_j=0.001,
                 n_jumps=0, n_obs=0):
        self.lam     = lam      # jump arrival rate (jumps per tick)
        self.mu_j    = mu_j     # mean log-return of a jump
        self.sigma_j = sigma_j  # std dev of jump log-return
        self.n_jumps = n_jumps  # observed jump count (used for fallback check)
        self.n_obs   = n_obs    # total observations used for fit

    def __repr__(self):
        return (f"JumpParams(λ={self.lam:.5f}/tick  "
                f"μ_J={self.mu_j:+.5f}  σ_J={self.sigma_j:.5f}  "
                f"n_jumps={self.n_jumps}/{self.n_obs})")


class MonteCarloPricer:
    """
    Estimates P(price stays inside ±barrier_offset for all T ticks) using
    two simulation models that are blended into a single containment probability.

    ── Model 1: Pure GBM ────────────────────────────────────────────────────
    Each tick step:
        dlog S_t = σ · Z_t,   Z_t ~ N(0,1)
    σ is the per-tick EWMA volatility (α=0.06).

    ── Model 2: Merton Jump-Diffusion ───────────────────────────────────────
    Each tick step:
        dlog S_t = σ_d · Z_t  +  Σ_{k=1}^{N_t} Y_k
    where:
        σ_d  = diffusion-only σ (EWMA σ computed on non-jump returns)
        N_t  ~ Poisson(λ)       per-tick jump count
        Y_k  ~ N(μ_J, σ_J²)    individual jump size

    Parameters are estimated from the tick buffer:
        - Any log-return with |r| > jump_threshold · σ_ewma is flagged as a jump.
        - λ    = n_jumps / n_obs                (empirical jump rate)
        - μ_J  = mean(jump_returns)             (signed; ≈ 0 for symmetric noise)
        - σ_J  = std(jump_returns)              (always >> σ_d)
        - σ_d  = EWMA σ recomputed on diffusion-only (non-jump) returns

    Vectorised implementation — no Python loops over paths:
        diffusion : (N, T) standard normals  ×  σ_d
        jumps     : (N, T) Poisson counts   ×  N(μ_J, σ_J²) per jump
                    drawn as Poisson(λ·T)   split uniformly across T ticks
                    (thinning approximation, exact for small λ)

    ── Blending ─────────────────────────────────────────────────────────────
        p_final = (1 - w) · p_gbm  +  w · p_jd
    where w = jd_weight (default 0.5).
    Falls back to pure GBM (w=0) when fewer than jd_min_jumps have been
    observed — not enough data to trust the jump parameter estimates.

    ── Logging ──────────────────────────────────────────────────────────────
    The last fitted JumpParams are stored in self.last_jump_params for
    inclusion in trade logs and diagnostics.
    """

    def __init__(self, cfg):
        self.n_paths         = cfg.get("mc_n_paths",          1000)
        self.vol_win         = cfg.get("mc_vol_window",        60)
        self.alpha           = cfg.get("mc_ewma_alpha",        0.06)
        self.p_floor         = cfg.get("mc_p_floor",           0.670)
        self._tpm_cfg        = cfg.get("mc_ticks_per_min",     None)
        self.terminal_weight = cfg.get("mc_terminal_weight",   0.7)

        # Jump-diffusion settings
        self.jump_threshold  = cfg.get("jd_jump_threshold",   3.0)
        self.jd_fit_window   = cfg.get("jd_fit_window",       300)
        self.jd_min_jumps    = cfg.get("jd_min_jumps",        5)
        self.jd_weight       = cfg.get("jd_weight",           0.5)

        self._rng            = np.random.default_rng()   # PCG64, thread-safe per instance

        # Public: last fitted jump params (for logging)
        self.last_jump_params: JumpParams = JumpParams()

    # ── σ estimation ──────────────────────────────────────────────────────

    def ewma_sigma(self, tick_buf) -> float:
        """Full-sample EWMA σ (diffusion + jumps). Per-tick σ."""
        buf = list(tick_buf)[-(self.vol_win + 1):]
        if len(buf) < 3:
            return 0.001
        prices = np.array([t["price"] for t in buf], dtype=float)
        lr     = np.diff(np.log(np.maximum(prices, 1e-8)))
        if len(lr) < 2:
            return max(float(np.std(lr)), 1e-8)
        var = float(lr[0] ** 2)
        for r in lr[1:]:
            var = self.alpha * float(r ** 2) + (1.0 - self.alpha) * var
        return max(math.sqrt(var), 1e-8)

    # ── tpm detection ────────────────────────────────────────────────────

    @staticmethod
    def detect_tpm(tick_buf, window=30) -> float:
        """Estimate ticks-per-minute from tick timestamps."""
        buf = list(tick_buf)[-window:]
        if len(buf) < 2:
            return 60.0
        dt = buf[-1]["timestamp"] - buf[0]["timestamp"]
        return (len(buf) - 1) / dt * 60.0 if dt > 0 else 60.0

    # ── Jump parameter estimation ─────────────────────────────────────────

    def fit_jumps(self, tick_buf, sigma_ewma: float) -> JumpParams:
        """
        Estimate Merton jump parameters from recent tick history.

        Classification rule
        -------------------
        A log-return r is classified as a jump if:
            |r| > jump_threshold · σ_ewma

        This separates the heavy-tailed outliers (jumps) from the
        Gaussian diffusion core.  The threshold is intentionally wide
        (default 3σ) so that normal variance is not misclassified as jumps.

        Returns a JumpParams with fitted λ, μ_J, σ_J (and σ_d implicitly
        available as the EWMA of the non-jump returns, recomputed in
        simulate_with_sigma).
        """
        buf = list(tick_buf)[-(self.jd_fit_window + 1):]
        if len(buf) < 10:
            return JumpParams()

        prices   = np.array([t["price"] for t in buf], dtype=float)
        lr       = np.diff(np.log(np.maximum(prices, 1e-8)))
        n_obs    = len(lr)

        threshold  = self.jump_threshold * sigma_ewma
        jump_mask  = np.abs(lr) > threshold
        n_jumps    = int(jump_mask.sum())

        if n_jumps == 0:
            return JumpParams(lam=0.0, mu_j=0.0,
                              sigma_j=sigma_ewma * self.jump_threshold,
                              n_jumps=0, n_obs=n_obs)

        jump_returns = lr[jump_mask]
        lam          = n_jumps / n_obs
        mu_j         = float(np.mean(jump_returns))
        sigma_j      = float(np.std(jump_returns, ddof=1)) if n_jumps > 1 \
                       else abs(mu_j) * 0.5 + 1e-8

        return JumpParams(lam=lam, mu_j=mu_j, sigma_j=sigma_j,
                          n_jumps=n_jumps, n_obs=n_obs)

    # ── Core simulation ───────────────────────────────────────────────────

    def simulate(self, tick_buf, barrier_offset: float, duration_mins: float
                 ) -> tuple[float, float, int]:
        """
        Convenience entry-point: compute σ and tpm, then delegate to
        simulate_with_sigma.
        """
        if not tick_buf:
            return 0.5, 0.001, 1
        sigma = self.ewma_sigma(tick_buf)
        tpm   = self._tpm_cfg or self.detect_tpm(tick_buf)
        return self.simulate_with_sigma(tick_buf, barrier_offset,
                                        duration_mins, sigma, tpm)

    def simulate_with_sigma(self, tick_buf, barrier_offset: float,
                            duration_mins: float, sigma: float, tpm: float
                            ) -> tuple[float, float, int]:
        """
        Run both GBM and Jump-Diffusion simulations and return the
        blended containment probability.

        Called from _evaluate_signal (with shared σ/tpm across durations).

        Returns (p_blended, sigma, T)
        """
        if not tick_buf:
            return 0.5, sigma, 1

        S  = max(list(tick_buf)[-1]["price"], 1e-8)
        T  = max(1, int(round(tpm * duration_mins)))

        # Log-barrier bounds (symmetric around current log-price)
        log_hi = math.log((S + barrier_offset) / S)
        log_lo = math.log(max(S - barrier_offset, 1e-8) / S)

        # ── GBM paths ─────────────────────────────────────────────────
        gbm_shocks = self._rng.standard_normal((self.n_paths, T)) * sigma
        gbm_cum    = np.cumsum(gbm_shocks, axis=1)   # (N, T)

        # Terminal containment: final log-return inside band
        # (matches Deriv EXPIRYRANGE settlement — last tick only)
        gbm_terminal = (
            (gbm_cum[:, -1] <= log_hi) &
            (gbm_cum[:, -1] >= log_lo)
        )
        # Path containment: never breaches at any tick (strict first-passage)
        gbm_path = (
            (gbm_cum <= log_hi).all(axis=1) &
            (gbm_cum >= log_lo).all(axis=1)
        )
        # Blend: terminal_weight controls the mix
        tw    = self.terminal_weight
        p_gbm = float(tw * gbm_terminal.mean() + (1.0 - tw) * gbm_path.mean())

        # ── Jump-Diffusion paths ───────────────────────────────────────
        jp = self.fit_jumps(tick_buf, sigma)
        self.last_jump_params = jp   # store for logging

        sufficient_jumps = jp.n_jumps >= self.jd_min_jumps
        effective_weight = self.jd_weight if sufficient_jumps else 0.0

        if effective_weight > 0.0:
            # Diffusion-only σ: recompute EWMA excluding jump ticks
            buf      = list(tick_buf)[-(self.jd_fit_window + 1):]
            prices   = np.array([t["price"] for t in buf], dtype=float)
            lr_all   = np.diff(np.log(np.maximum(prices, 1e-8)))
            thr      = self.jump_threshold * sigma
            diff_lr  = lr_all[np.abs(lr_all) <= thr]
            if len(diff_lr) >= 2:
                var_d = float(diff_lr[0] ** 2)
                for r in diff_lr[1:]:
                    var_d = self.alpha * float(r ** 2) + (1.0 - self.alpha) * var_d
                sigma_d = max(math.sqrt(var_d), 1e-8)
            else:
                sigma_d = sigma   # fall back if too few diffusion ticks

            # Diffusion component: (N, T)
            diff_shocks = self._rng.standard_normal((self.n_paths, T)) * sigma_d

            # Jump component — thinning approximation:
            #   Expected total jumps per path = λ · T
            #   Draw Poisson(λ·T) total jumps, place them uniformly on [0,T)
            lam_total = jp.lam * T
            # Vectorised: draw total jump counts per path, then add
            # their summed sizes directly to the log-return matrix.
            # For small λ (typical: < 0.02/tick) this is exact in distribution.
            n_jumps_per_path = self._rng.poisson(lam_total, size=self.n_paths)
            # jump_impact[i] = sum of Y_k for path i, where each Y_k ~ N(μ_J, σ_J²)
            # Using the identity: sum of m iid N(μ,σ²) = N(m·μ, m·σ²)
            jump_impact = np.where(
                n_jumps_per_path > 0,
                self._rng.normal(
                    n_jumps_per_path * jp.mu_j,
                    np.sqrt(np.maximum(n_jumps_per_path, 0)) * jp.sigma_j
                ),
                0.0,
            )   # shape (N,)

            # Distribute jump impact uniformly across a random tick in [0, T)
            # (for barrier purposes, the worst case is a jump at any tick)
            # We add total impact at a random step index so the path shape
            # is realistic — the cumsum barrier check will capture it.
            jump_step = self._rng.integers(0, T, size=self.n_paths)
            jump_matrix = np.zeros((self.n_paths, T), dtype=float)
            np.add.at(jump_matrix, (np.arange(self.n_paths), jump_step),
                      jump_impact)

            jd_cum  = np.cumsum(diff_shocks + jump_matrix, axis=1)  # (N, T)

            # Same dual terminal+path check as GBM
            jd_terminal = (
                (jd_cum[:, -1] <= log_hi) &
                (jd_cum[:, -1] >= log_lo)
            )
            jd_path = (
                (jd_cum <= log_hi).all(axis=1) &
                (jd_cum >= log_lo).all(axis=1)
            )
            p_jd = float(tw * jd_terminal.mean() + (1.0 - tw) * jd_path.mean())
        else:
            p_jd = p_gbm   # not enough jump history — JD = GBM

        # ── Blend ─────────────────────────────────────────────────────
        p_blend = (1.0 - effective_weight) * p_gbm + effective_weight * p_jd

        return float(np.clip(p_blend, 0.01, 0.99)), sigma, T

    # ── EV helper ─────────────────────────────────────────────────────────

    @staticmethod
    def ev(p_win: float, payout_ratio: float) -> float:
        return float(p_win * payout_ratio - (1.0 - p_win))


# ===========================================================================
# KELLY STAKER
# ===========================================================================

class KellyStaker:
    """
    Tiered stake sizing calibrated for $1 account:
      $1.00–$1.99  → 35%  (ensures $0.35 Deriv minimum is met)
      $2.00–$4.99  → 12%  (tapering off)
      $5.00–$14.99 →  7%  (conservative growth)
      $15.00+      →  5%  (steady compounding)
    Hard cap: 10% of balance per trade, $5.00 absolute maximum.
    """
    def __init__(self, cfg):
        self.fraction  = cfg["kelly_fraction"]
        self.max_pct   = cfg["kelly_max_pct"]
        self.min_stake = cfg["kelly_min_stake"]
        self.max_stake = cfg["kelly_max_stake"]
        self.wins = self.n = 0

    def _tiered_pct(self, balance):
        if balance < 2.00:  return 0.35
        if balance < 5.00:  return 0.12
        if balance < 15.00: return 0.07
        return 0.05

    def next_stake(self, p_win, balance, payout_ratio=0.48):
        if balance <= 0: return self.min_stake
        prop_stake  = balance * self._tiered_pct(balance)
        b           = payout_ratio; q = 1.0 - p_win
        f_star      = (p_win * b - q) / b
        kelly_stake = min(f_star * self.fraction, self.max_pct) * balance if f_star > 0 else 0.0
        stake = max(prop_stake, kelly_stake)
        stake = min(stake, self.max_stake, balance * self.max_pct)
        stake = max(stake, self.min_stake)
        stake = min(stake, balance)
        return round(stake, 2)

    def record(self, win: bool):
        if win: self.wins += 1
        self.n += 1
        log.info("[Kelly] %s  WR=%.1f%%",
                 "WIN" if win else "LOSS",
                 self.wins / self.n * 100 if self.n else 0)


# ===========================================================================
# TRADE LOGGER  (CSV)
# ===========================================================================

class TradeLogger:
    FIELDS = [
        "timestamp", "symbol", "duration_mins",
        "barrier_offset", "price_at_entry",
        "p_contain", "ev", "payout",
        "sigma_ewma", "mc_paths", "mc_ticks",
        "jd_lambda", "jd_mu_j", "jd_sigma_j", "jd_n_jumps", "jd_weight_used",
        "stake", "balance_before", "outcome", "profit", "balance_after",
        "sprt_status", "session_wr",
    ]

    def __init__(self, path):
        self.path    = path
        self._exists = os.path.isfile(path)

    def log(self, row: dict):
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS, extrasaction="ignore")
            if not self._exists:
                w.writeheader()
                self._exists = True
            w.writerow(row)


# ===========================================================================
# CONNECTION LAYER  (ConnState + DerivWSManager)
# ===========================================================================

class ConnState(enum.IntEnum):
    """Connection lifecycle states, in ascending order of readiness."""
    DISCONNECTED  = 0
    CONNECTING    = 1
    CONNECTED     = 2   # TCP + TLS up
    AUTHENTICATED = 3   # authorize response confirmed
    SUBSCRIBED    = 4   # tick stream flowing


class DerivWSManager:
    """
    Manages a persistent, self-healing WebSocket connection to Deriv.

    Improvements over the naive run_forever + sleep(3) loop
    --------------------------------------------------------
    1.  NEW WebSocketApp every reconnect cycle.
        run_forever() leaves the object in a dead state; recycling it
        causes silent send failures or hangs.

    2.  Exponential back-off with +/-1s jitter.
        First connect is immediate. Attempt n waits:
            min(2 * 2^(n-1), 120) + uniform(-1, 1) seconds.
        Avoids thundering-herd and Deriv 429 rate-limit after outages.

    3.  Single heartbeat daemon bound per WS object.
        The thread exits cleanly when its ws instance dies.
        Old code spawned a new Heartbeat thread on every _on_open while
        the previous thread was still running on a dead socket.

    4.  Thread-safe safe_send().
        Captures both ws pointer and state check atomically under _lock.
        Never raises; callers fire-and-forget.

    5.  on_disconnect_cb fires before every reconnect.
        Lets the trader reset waiting_proposal / waiting_result so the
        bot cannot hang indefinitely after a mid-trade connection drop.

    6.  stop() is idempotent; safe from any thread.
    """

    WS_URL             = "wss://ws.binaryws.com/websockets/v3"
    HEARTBEAT_INTERVAL = 20      # JSON pings (beats Railway ~55s idle timeout)
    PING_INTERVAL      = 25      # websocket-client WS-frame ping
    PING_TIMEOUT       = 15
    RECONNECT_BASE     = 2.0     # initial back-off (seconds)
    RECONNECT_CAP      = 120.0   # maximum back-off (seconds)

    def __init__(self, app_id: int,
                 on_open_cb,
                 on_message_cb,
                 on_disconnect_cb=None,
                 name: str = "DerivWS"):
        self.app_id            = app_id
        self._on_open_cb       = on_open_cb
        self._on_message_cb    = on_message_cb
        self._on_disconnect_cb = on_disconnect_cb
        self.name              = name

        self._lock    = threading.Lock()
        self.state    = ConnState.DISCONNECTED
        self._running = False
        self._ws      = None   # replaced every cycle; NEVER reused
        self._attempt = 0      # drives back-off; reset to 0 on successful open

    # ── Public ────────────────────────────────────────────────────────────

    def safe_send(self, payload: dict) -> bool:
        """
        Thread-safe JSON send.
        Returns True if delivered to the socket, False if not connected.
        Never raises; callers can fire-and-forget.
        """
        with self._lock:
            ws   = self._ws
            live = (self.state >= ConnState.CONNECTED and ws is not None)
        if not live:
            return False
        try:
            ws.send(json.dumps(payload))
            return True
        except Exception as e:
            log.warning("[%s] safe_send failed: %s", self.name, e)
            return False

    def start(self):
        """Enter the reconnect loop (blocks the calling thread until stop())."""
        self._running = True
        self._loop()

    def stop(self):
        """Signal stop and close the active connection. Idempotent."""
        self._running = False
        self.state    = ConnState.DISCONNECTED
        with self._lock:
            ws = self._ws
        if ws:
            try: ws.close()
            except Exception: pass

    # ── Reconnect loop ────────────────────────────────────────────────────

    def _loop(self):
        while self._running:
            # Back-off (skipped on very first connect)
            if self._attempt > 0:
                delay = min(
                    self.RECONNECT_BASE * (2 ** (self._attempt - 1)),
                    self.RECONNECT_CAP,
                ) + random.uniform(-1.0, 1.0)
                delay = max(1.0, delay)
                log.info("[%s] Reconnect #%d in %.1fs ...",
                         self.name, self._attempt, delay)
                time.sleep(delay)

            if not self._running:
                break

            # Fresh WebSocketApp -- NEVER reuse the previous object
            self.state = ConnState.CONNECTING
            ws = websocket.WebSocketApp(
                f"{self.WS_URL}?app_id={self.app_id}",
                on_open    = self._cb_open,
                on_message = self._cb_message,
                on_error   = self._cb_error,
                on_close   = self._cb_close,
            )
            with self._lock:
                self._ws = ws

            try:
                ws.run_forever(
                    ping_interval = self.PING_INTERVAL,
                    ping_timeout  = self.PING_TIMEOUT,
                    sslopt        = {"check_hostname": True},
                )
            except Exception as e:
                log.error("[%s] run_forever raised: %s", self.name, e)

            # Connection is dead
            self.state = ConnState.DISCONNECTED
            with self._lock:
                self._ws = None

            if not self._running:
                break

            # Fire hook so trader can reset stuck flags before the sleep
            if self._on_disconnect_cb:
                try: self._on_disconnect_cb()
                except Exception as e:
                    log.error("[%s] on_disconnect_cb raised: %s", self.name, e)

            self._attempt += 1

        log.info("[%s] Connection loop exited cleanly.", self.name)

    # ── WebSocketApp callbacks ────────────────────────────────────────────

    def _cb_open(self, ws):
        self._attempt = 0            # successful connect: reset back-off
        self.state    = ConnState.CONNECTED
        log.info("[%s] Connected (back-off reset).", self.name)
        self._spawn_heartbeat(ws)    # heartbeat bound to THIS ws object
        try:
            self._on_open_cb(ws)
        except Exception as e:
            log.error("[%s] on_open_cb raised: %s", self.name, e, exc_info=True)

    def _cb_message(self, ws, raw):
        try:
            self._on_message_cb(ws, raw)
        except Exception as e:
            log.error("[%s] on_message_cb raised: %s",
                      self.name, e, exc_info=True)

    def _cb_error(self, ws, error):
        log.warning("[%s] WS error: %s", self.name, error)

    def _cb_close(self, ws, code, msg):
        log.info("[%s] WS closed  code=%s  msg=%s", self.name, code, msg)

    # ── Heartbeat ─────────────────────────────────────────────────────────

    def _spawn_heartbeat(self, ws):
        """
        Starts a daemon thread that sends a JSON ping every HEARTBEAT_INTERVAL.
        The thread is bound to this specific ws object and exits cleanly when
        that ws dies, leaving reconnect and any new heartbeat unaffected.
        """
        def _beat():
            while self._running and self.state >= ConnState.CONNECTED:
                try:
                    ws.send(json.dumps({"ping": 1}))
                except Exception:
                    break   # ws is dead; _loop handles reconnect
                time.sleep(self.HEARTBEAT_INTERVAL)
            log.debug("[%s] Heartbeat exiting.", self.name)

        threading.Thread(
            target = _beat,
            daemon = True,
            name   = f"{self.name}-HB",
        ).start()


# ===========================================================================
# HISTORICAL COLLECTOR
# ===========================================================================

class HistoricalCollector:
    WS_URL       = "wss://ws.binaryws.com/websockets/v3"
    MAX_PER_CALL = 5000

    def __init__(self, symbol, cfg, done, existing_df=None):
        self.symbol    = symbol
        self.cfg       = cfg
        self.done      = done
        self._existing = existing_df

    def _fetch_page_once(self, end_epoch):
        """Single attempt to fetch one page of historical ticks."""
        import queue as _q
        q = _q.Queue()

        def _on_open(ws):
            ws.send(json.dumps({
                "ticks_history": self.symbol, "end": end_epoch,
                "count": self.MAX_PER_CALL, "style": "ticks",
                "adjust_start_time": 1,
            }))

        def _on_msg(ws, raw):
            try:
                msg = json.loads(raw)
                if msg.get("msg_type") == "history":
                    h = msg.get("history", {})
                    q.put(sorted(
                        [{"timestamp": float(t), "price": float(p)}
                         for t, p in zip(h.get("times", []), h.get("prices", []))],
                        key=lambda x: x["timestamp"]
                    ))
                elif "error" in msg:
                    log.error("[Collector] Error: %s", msg["error"])
                    q.put([])
                else:
                    return
                ws.close()
            except Exception as e:
                log.warning("[Collector] Parse error: %s", e)
                q.put([]); ws.close()

        def _on_err(ws, e):
            log.warning("[Collector] WS error: %s", e); q.put([])

        ws = websocket.WebSocketApp(
            f"{self.WS_URL}?app_id={self.cfg['app_id']}",
            on_open=_on_open, on_message=_on_msg,
            on_error=_on_err, on_close=lambda *_: None,
        )
        t = threading.Thread(target=ws.run_forever, daemon=True)
        t.start()
        try:
            return q.get(timeout=25)
        except Exception:
            return []
        finally:
            try: ws.close()
            except Exception: pass
            t.join(timeout=3)

    def _fetch_page(self, end_epoch, max_retries=3):
        """Fetch one page of tick history with exponential back-off retry."""
        for attempt in range(max_retries):
            result = self._fetch_page_once(end_epoch)
            if result:
                return result
            if attempt < max_retries - 1:
                delay = (2 ** attempt) + random.uniform(0.0, 1.0)
                log.warning("[Collector/%s] Page fetch empty (attempt %d/%d)"
                            " -- retrying in %.1fs ...",
                            self.symbol, attempt + 1, max_retries, delay)
                time.sleep(delay)
        return []

    def _collect(self):
        cfg          = self.cfg
        collect_secs = int(cfg["collect_hours"] * 3600)
        now_epoch    = int(time.time())
        cutoff_epoch = now_epoch - collect_secs

        log.info("[Collector/%s] Fetching %.1fh of ticks ...",
                 self.symbol, cfg["collect_hours"])

        all_ticks = []
        if self._existing is not None and not self._existing.empty:
            in_win = self._existing[self._existing["timestamp"] >= cutoff_epoch]
            if not in_win.empty:
                all_ticks = in_win.to_dict("records")
                if min(t["timestamp"] for t in all_ticks) <= cutoff_epoch:
                    log.info("[Collector/%s] Existing CSV covers window — reusing.",
                             self.symbol)
                    self._save(all_ticks); return

        fetch_end = now_epoch
        for page_num in range(1, 20):
            page = self._fetch_page(fetch_end)
            if not page:
                log.warning("[Collector/%s] Empty page %d.", self.symbol, page_num)
                break
            all_ticks.extend([tk for tk in page if tk["timestamp"] >= cutoff_epoch])
            if page[0]["timestamp"] <= cutoff_epoch:
                break
            fetch_end = int(page[0]["timestamp"]) - 1
            time.sleep(0.3)

        log.info("[Collector/%s] Collected %d ticks.", self.symbol, len(all_ticks))
        self._save(all_ticks)

    def _save(self, ticks):
        os.makedirs(self.cfg["data_dir"], exist_ok=True)
        path = os.path.join(self.cfg["data_dir"], f"ticks_{self.symbol}.csv")
        if ticks:
            df = (pd.DataFrame(ticks)
                  .drop_duplicates(subset=["timestamp"])
                  .sort_values("timestamp")
                  .reset_index(drop=True))
            df.to_csv(path, index=False)
            log.info("[Collector/%s] Saved %d ticks → %s",
                     self.symbol, len(df), path)
        self.done.set()

    def start(self):
        def _run():
            try:
                self._collect()
            except Exception as e:
                log.error("[Collector/%s] Fatal: %s", self.symbol, e, exc_info=True)
                self.done.set()
        threading.Thread(target=_run, daemon=True,
                         name=f"HistCol-{self.symbol}").start()


# ===========================================================================
# LIVE TRADER
# ===========================================================================

class LiveTrader:
    WS_URL = "wss://ws.binaryws.com/websockets/v3"

    def __init__(self, cfg, initial_ticks, staker, sprt, trade_logger):
        self.cfg       = cfg
        self.staker    = staker
        self.sprt      = sprt
        self.logger    = trade_logger
        self.pricer    = MonteCarloPricer(cfg)

        # Tick buffer — seeded with historical ticks, grows with live ticks
        self.tick_buf  = deque(initial_ticks, maxlen=5000)

        # Per-duration calibrated barriers and payout cache
        self.barrier_table = {}   # {duration_mins: barrier_offset}
        self.payout_table  = {}   # {duration_mins: payout_ratio}

        # State
        self.balance        = None
        self.start_balance  = None
        self.peak_balance   = None
        self.running        = False

        # Trade state
        self.waiting_result   = False
        self.waiting_proposal = False
        self.pending_pid      = None
        self.pending_stake    = 0.0
        self.pending_dur      = None
        self.pending_p_stat   = 0.0
        self.pending_barrier  = 0.0
        self.pending_payout   = 0.0

        # Counters
        self.live_tick_count  = 0
        self.post_trade_tick  = 0
        self.consec_losses    = 0
        self.cooldown_until   = 0
        self.wins = self.total = 0
        self.session_pnl      = 0.0

        # Signal persistence
        self._persist_count   = 0
        self._persist_best_p  = 0.0
        self._persist_dur     = None

        # Cache for CSV logging
        self._last_p_stat     = 0.0
        self._last_ev         = 0.0
        self._last_sigma      = 0.0
        self._last_mc_T       = 0

        # Barrier calibration flag
        self._calibrating     = False

        # Connection manager (replaces raw WebSocketApp management)
        self._conn = DerivWSManager(
            app_id           = cfg["app_id"],
            on_open_cb       = self._on_ws_open,
            on_message_cb    = self._on_message,
            on_disconnect_cb = self._on_disconnect,
            name             = "LiveTrader",
        )

    # ── WebSocket lifecycle ───────────────────────────────────────────────

    def run(self):
        self.running = True
        log.info("[Trader] Starting connection manager ...")
        self._conn.start()   # blocks until stop() is called

    def stop(self):
        self.running = False
        self._conn.stop()

    def _on_ws_open(self, ws):
        """Called by DerivWSManager on every (re)connect."""
        log.info("[Trader] (Re)connected -- authorising ...")
        # Heartbeat is owned by DerivWSManager._spawn_heartbeat; none needed here.
        self._conn.safe_send({"authorize": self.cfg["api_token"]})

    def _on_disconnect(self):
        """
        Called by DerivWSManager before every reconnect attempt.
        Resets flags that could leave the bot permanently hung when a
        connection drop occurs mid-trade or mid-proposal.
        """
        if self.waiting_proposal:
            log.warning("[Trader] Connection lost during pending proposal"
                        " -- resetting flag.")
            self.waiting_proposal = False
        if self.waiting_result:
            log.warning("[Trader] Connection lost while awaiting contract"
                        " settlement -- resetting flag (outcome unknown).")
            self.waiting_result = False

    def _on_message(self, ws, raw):
        try:
            msg = json.loads(raw)
        except Exception:
            return
        mt = msg.get("msg_type", "")
        try:
            if   mt == "authorize":      self._on_auth(msg)
            elif mt == "balance":        self._on_balance(msg)
            elif mt == "tick":           self._on_tick(msg)
            elif mt == "proposal":       self._on_proposal(msg)
            elif mt == "buy":            self._on_buy(msg)
            elif mt == "proposal_open_contract": self._on_poc(msg)
            elif "error" in msg:
                log.warning("[Trader] API error: %s",
                            msg["error"].get("message", str(msg["error"])))
        except Exception as e:
            log.error("[Trader] Message handler error: %s", e, exc_info=True)

    # ── Auth ─────────────────────────────────────────────────────────────

    def _on_auth(self, msg):
        info           = msg.get("authorize", {})
        self.balance   = float(info.get("balance", 0))
        self.start_balance = self.balance
        self.peak_balance  = self.balance
        log.info("[Trader] Authorised: %s | %s %.2f",
                 info.get("loginid", "?"),
                 info.get("currency", "USD"),
                 self.balance)
        self._conn.state = ConnState.AUTHENTICATED
        self._conn.safe_send({"balance": 1, "subscribe": 1})
        self._conn.safe_send({"ticks": self.cfg["symbol"], "subscribe": 1})
        self._conn.state = ConnState.SUBSCRIBED
        threading.Thread(target=self._calibrate_barriers,
                         daemon=True, name="BarrierCal").start()
        log.info("[Trader] Waiting for tick buffer to fill "
                 "(need %d ticks) ...", self.cfg["min_ticks"])

    # ── Balance ───────────────────────────────────────────────────────────

    def _on_balance(self, msg):
        b = msg.get("balance", {})
        if "balance" in b:
            self.balance = float(b["balance"])
            if self.peak_balance is None or self.balance > self.peak_balance:
                self.peak_balance = self.balance

    # ── Tick ──────────────────────────────────────────────────────────────

    def _on_tick(self, msg):
        t = msg.get("tick", {})
        self.tick_buf.append({
            "timestamp": float(t.get("epoch", time.time())),
            "price"    : float(t.get("quote", 0)),
        })
        self.live_tick_count += 1

        if not self.running or self.waiting_result or self.waiting_proposal:
            return

        # Not enough ticks yet
        if len(self.tick_buf) < self.cfg["min_ticks"]:
            if self.live_tick_count % 100 == 0:
                log.info("[Trader] Buffer %d/%d ticks ...",
                         len(self.tick_buf), self.cfg["min_ticks"])
            return

        # Cooldown
        if self.live_tick_count < self.cooldown_until:
            return

        # Post-trade cooldown
        min_gap = self.cfg.get("min_ticks_between_trades", 60)
        if self.live_tick_count - self.post_trade_tick < min_gap:
            return

        self._evaluate_signal()

    # ── Monte Carlo signal evaluation ─────────────────────────────────────

    @staticmethod
    def _dur_crosses_midnight(duration_mins: int) -> bool:
        """True if a trade opened now would expire past 00:00 UTC."""
        now = datetime.utcnow()
        secs_to_midnight = (23 - now.hour)*3600 + (59 - now.minute)*60 + (59 - now.second)
        return duration_mins * 60 > secs_to_midnight

    def _evaluate_signal(self):
        cfg      = self.cfg
        ev_floor = cfg.get("ev_confidence_floor", 0.673)
        ev_thr   = cfg.get("min_ev_threshold",    0.001)
        fallback = cfg.get("expiryrange_barrier",  2.97)
        durations = cfg.get("hold_durations",      [2, 3, 4, 5, 6])

        best_p   = 0.0
        best_dur = None
        best_sigma = 0.0
        best_T   = 0

        # Compute σ once for the current tick buffer — shared across all durations.
        # Only the tick horizon T differs per duration; σ is buffer-wide.
        shared_sigma = self.pricer.ewma_sigma(self.tick_buf)
        tpm          = self.pricer._tpm_cfg or MonteCarloPricer.detect_tpm(self.tick_buf)

        # ── Run MC simulation for every candidate duration, pick best ──────
        for dur in durations:
            if self._dur_crosses_midnight(dur):
                continue   # would cross UTC midnight — Deriv rejects
            barrier = self.barrier_table.get(dur, fallback)
            # Pass pre-computed sigma directly to avoid recomputing per duration
            p, sigma, T = self.pricer.simulate_with_sigma(
                self.tick_buf, barrier, dur, shared_sigma, tpm)
            if p > best_p:
                best_p    = p
                best_dur  = dur
                best_sigma = sigma
                best_T    = T

        # Cache for logging
        self._last_p_stat = best_p
        self._last_sigma  = best_sigma
        self._last_mc_T   = best_T

        # ── EV gate (prelim — uses cached payout) ─────────────────────────
        cached_payout = self.payout_table.get(best_dur,
                        cfg.get("min_payout_pct", 0.48))
        ev = MonteCarloPricer.ev(best_p, cached_payout)
        self._last_ev = ev

        if best_p < ev_floor or ev < ev_thr:
            # Soft decay instead of hard reset — one below-floor tick no longer
            # wipes accumulated signal. Count drops by 1 each miss (floor 0).
            self._persist_count = max(0, self._persist_count - 1)
            if self.live_tick_count % 30 == 0:
                log.info("[MC] p=%.4f ev=%+.4f dur=%sm  SKIP (below floor)  "
                         "σ=%.6f λ=%.5f  persist_decay=%d",
                         best_p, ev,
                         str(best_dur) if best_dur else "?",
                         best_sigma,
                         self.pricer.last_jump_params.lam,
                         self._persist_count)
            return

        # ── Signal persistence ─────────────────────────────────────────────
        required = cfg.get("signal_persistence_ticks", 2)
        if best_dur == self._persist_dur:
            self._persist_count += 1
            if best_p > self._persist_best_p:
                self._persist_best_p = best_p
        else:
            self._persist_count  = 1
            self._persist_best_p = best_p
            self._persist_dur    = best_dur

        if self._persist_count < required:
            if self.live_tick_count % 10 == 0:
                log.info("[MC] Persistence %d/%d  p=%.4f  dur=%dm",
                         self._persist_count, required, best_p, best_dur)
            return

        # ── All checks passed — send proposal ─────────────────────────────
        jp = self.pricer.last_jump_params
        log.info("[MC] *** SIGNAL *** p_blend=%.4f ev=%+.4f "
                 "dur=%dm σ=%.6f T=%d persist=%d | "
                 "JD: λ=%.5f μ_J=%+.5f σ_J=%.5f n_jumps=%d/%d weight=%.1f",
                 best_p, ev, best_dur, best_sigma, best_T,
                 self._persist_count,
                 jp.lam, jp.mu_j, jp.sigma_j, jp.n_jumps, jp.n_obs,
                 self.cfg.get("jd_weight", 0.5))

        self._persist_count = 0   # reset after firing

        if not self.balance or self.balance <= 0:
            log.warning("[Trader] Balance not confirmed — skipping.")
            return

        stake = self.staker.next_stake(best_p, self.balance, cached_payout)

        self.pending_stake    = stake
        self.pending_dur      = best_dur
        self.pending_p_stat   = best_p
        self.pending_barrier  = self.barrier_table.get(best_dur, fallback)
        self.pending_payout   = cached_payout
        self.waiting_proposal = True

        barrier_offset = self.barrier_table.get(best_dur, fallback)
        self._conn.safe_send({
            "proposal"      : 1,
            "amount"        : stake,
            "basis"         : "stake",
            "contract_type" : "EXPIRYRANGE",
            "currency"      : cfg.get("currency", "USD"),
            "duration"      : best_dur,
            "duration_unit" : "m",
            "symbol"        : cfg["symbol"],
            "barrier"       : f"+{barrier_offset:.2f}",
            "barrier2"      : f"-{barrier_offset:.2f}",
        })

    # ── Proposal response ─────────────────────────────────────────────────

    def _on_proposal(self, msg):
        if "error" in msg:
            log.warning("[Trader] Proposal error: %s",
                        msg["error"].get("message", str(msg["error"])))
            self.waiting_proposal = False
            return

        prop          = msg.get("proposal", {})
        pid           = msg.get("id") or prop.get("id")
        ask_price     = float(prop.get("ask_price",
                              prop.get("cost", self.pending_stake)))
        payout_amount = float(prop.get("payout", ask_price * 1.486))
        payout_ratio  = round((payout_amount - ask_price) / max(ask_price, 1e-8), 4)

        cfg      = self.cfg
        ev_floor = cfg.get("ev_confidence_floor", 0.673)
        ev_thr   = cfg.get("min_ev_threshold",    0.001)
        min_pay  = cfg.get("min_payout_pct",       0.48)

        # Payout minimum check
        if payout_ratio < min_pay:
            log.info("[EV] SKIP — payout=%.1f%% < min=%.0f%%",
                     payout_ratio * 100, min_pay * 100)
            self.waiting_proposal = False
            return

        # Final EV check with real payout
        p  = self.pending_p_stat
        ev = MonteCarloPricer.ev(p, payout_ratio)
        self._last_ev = ev

        if cfg.get("ev_check_on_proposal", True):
            if p < ev_floor or ev < ev_thr:
                log.info("[EV] SKIP — p=%.4f < floor=%.3f or EV=%+.4f < thr",
                         p, ev_floor, ev)
                self.waiting_proposal = False
                return

        self.pending_pid    = pid
        self.pending_payout = payout_ratio
        self.waiting_proposal = False
        self.waiting_result   = True

        log.info("[Trader] Proposal OK  payout=%.1f%%  p=%.4f  "
                 "EV=%+.4f  stake=$%.2f  pid=%s",
                 payout_ratio * 100, p, ev, self.pending_stake, pid)

        self._conn.safe_send({
            "buy"  : pid,
            "price": self.pending_stake,
        })

    # ── Buy confirmation ──────────────────────────────────────────────────

    def _on_buy(self, msg):
        if "error" in msg:
            log.warning("[Trader] Buy error: %s",
                        msg["error"].get("message", str(msg["error"])))
            self.waiting_result = False
            return
        buy = msg.get("buy", {})
        cid = str(buy.get("contract_id", "?"))
        log.info("[Trader] Contract opened  cid=%s  paid=$%.2f",
                 cid, float(buy.get("buy_price", self.pending_stake)))
        self._conn.safe_send({
            "proposal_open_contract": 1,
            "contract_id"           : int(cid),
            "subscribe"             : 1,
        })

    # ── Settlement ────────────────────────────────────────────────────────

    def _on_poc(self, msg):
        poc = msg.get("proposal_open_contract", {})
        if not poc.get("is_sold", 0):
            return  # not settled yet

        profit  = float(poc.get("profit", 0))
        win     = profit > 0
        new_bal = float(poc.get("balance_after", self.balance or 0))
        old_bal = self.balance or new_bal

        self.balance = new_bal
        if self.peak_balance is None or new_bal > self.peak_balance:
            self.peak_balance = new_bal

        self.session_pnl += profit
        self.total += 1
        if win:
            self.wins += 1
            self.consec_losses = 0
        else:
            self.consec_losses += 1

        wr = self.wins / self.total * 100

        log.info("[Trade] %s #%d | P&L=%+.2f | bal=$%.2f | "
                 "W/L=%d/%d (%.1f%%) | SPRT:%s",
                 "WIN " if win else "LOSS",
                 self.total, profit, new_bal,
                 self.wins, self.total - self.wins, wr,
                 self.sprt.update(win))

        self.staker.record(win)
        self.post_trade_tick = self.live_tick_count
        self.waiting_result  = False

        # Consecutive loss cooldown
        if (not win and
                self.consec_losses >= self.cfg.get("max_consec_losses", 5)):
            cd = self.cfg.get("consec_loss_cooldown_ticks", 60)
            self.cooldown_until = self.live_tick_count + cd
            log.warning("[Risk] %d consecutive losses — cooling down %d ticks.",
                        self.consec_losses, cd)
            self.consec_losses = 0

        # CSV log
        jp = self.pricer.last_jump_params
        eff_w = self.cfg.get("jd_weight", 0.5) if jp.n_jumps >= self.cfg.get("jd_min_jumps", 5) else 0.0
        self.logger.log({
            "timestamp"      : datetime.utcnow().isoformat(),
            "symbol"         : self.cfg["symbol"],
            "duration_mins"  : self.pending_dur,
            "barrier_offset" : round(self.pending_barrier, 4),
            "price_at_entry" : round(list(self.tick_buf)[-1]["price"], 5)
                               if self.tick_buf else "",
            "p_contain"      : round(self.pending_p_stat, 5),
            "ev"             : round(self._last_ev, 5),
            "payout"         : round(self.pending_payout, 4),
            "sigma_ewma"     : round(self._last_sigma, 8),
            "mc_paths"       : self.cfg.get("mc_n_paths", 1000),
            "mc_ticks"       : self._last_mc_T,
            "jd_lambda"      : round(jp.lam, 6),
            "jd_mu_j"        : round(jp.mu_j, 6),
            "jd_sigma_j"     : round(jp.sigma_j, 6),
            "jd_n_jumps"     : jp.n_jumps,
            "jd_weight_used" : round(eff_w, 2),
            "stake"          : round(self.pending_stake, 2),
            "balance_before" : round(old_bal, 2),
            "outcome"        : "WIN" if win else "LOSS",
            "profit"         : round(profit, 2),
            "balance_after"  : round(new_bal, 2),
            "sprt_status"    : self.sprt.status,
            "session_wr"     : round(wr, 2),
        })

        # Recalibrate barriers in background after every trade
        threading.Thread(target=self._calibrate_barriers,
                         daemon=True, name="BarrierRecal").start()

        # Risk checks
        if not self._risk_ok():
            log.warning("[Risk] Risk limit hit — stopping.")
            self.stop()

    # ── Risk ──────────────────────────────────────────────────────────────

    def _risk_ok(self):
        cfg = self.cfg
        sb  = self.start_balance or 1.0
        pb  = self.peak_balance  or self.balance or 1.0
        bl  = self.balance       or 0.0

        if self.session_pnl <= -(sb * cfg["max_daily_loss_pct"]):
            log.warning("[Risk] Daily loss limit: P&L=%.2f", self.session_pnl)
            return False
        if bl > 0 and (pb - bl) >= sb * cfg["max_drawdown_from_peak_pct"]:
            log.warning("[Risk] Drawdown limit: peak=%.2f current=%.2f", pb, bl)
            return False
        if self.session_pnl >= sb * cfg.get("take_profit_pct", 9999):
            log.info("[Risk] Take-profit reached: P&L=%.2f", self.session_pnl)
            return False
        return True

    # ── Barrier calibration ───────────────────────────────────────────────

    def _calibrate_barriers(self):
        if self._calibrating: return
        self._calibrating = True

        cfg      = self.cfg
        durations = cfg.get("hold_durations", [2, 3, 4, 5, 6])
        fallback = cfg.get("expiryrange_barrier", 2.97)
        symbol   = cfg["symbol"]

        log.info("[BarrierCal] Calibrating %s for durations %s ...",
                 symbol, durations)

        import queue as _q

        def _probe_once(duration, barrier):
            """Single attempt to probe the payout for a given duration+barrier."""
            q = _q.Queue()

            def _oo(ws):
                ws.send(json.dumps({
                    "proposal"      : 1,
                    "amount"        : cfg["kelly_min_stake"],
                    "basis"         : "stake",
                    "contract_type" : "EXPIRYRANGE",
                    "currency"      : cfg["currency"],
                    "duration"      : duration,
                    "duration_unit" : "m",
                    "symbol"        : symbol,
                    "barrier"       : f"+{barrier:.2f}",
                    "barrier2"      : f"-{barrier:.2f}",
                }))

            def _om(ws, raw):
                try:
                    m = json.loads(raw)
                    if m.get("msg_type") == "proposal":
                        p   = m.get("proposal", {})
                        ask = float(p.get("ask_price", cfg["kelly_min_stake"]))
                        pay = float(p.get("payout",   ask * 1.486))
                        q.put((pay - ask) / max(ask, 1e-8))
                    elif "error" in m:
                        q.put(None)
                    else:
                        return
                    ws.close()
                except Exception:
                    q.put(None); ws.close()

            ws2 = websocket.WebSocketApp(
                f"{self.WS_URL}?app_id={cfg['app_id']}",
                on_open=_oo, on_message=_om,
                on_error=lambda *_: q.put(None),
                on_close=lambda *_: None,
            )
            t = threading.Thread(target=ws2.run_forever, daemon=True)
            t.start()
            try:
                return q.get(timeout=15)
            except Exception:
                return None
            finally:
                try: ws2.close()
                except Exception: pass
                t.join(timeout=3)

        def _probe(duration, barrier, max_retries=3):
            """Probe with exponential back-off retry. Returns payout ratio or None."""
            for attempt in range(max_retries):
                result = _probe_once(duration, barrier)
                if result is not None:
                    return result
                if attempt < max_retries - 1:
                    delay = 2 ** attempt
                    log.warning("[BarrierCal] Probe failed  dur=%dm  barrier=+/-%.2f"
                                "  (attempt %d/%d) retrying in %ds ...",
                                duration, barrier, attempt + 1, max_retries, delay)
                    time.sleep(delay)
            return None

        # Per-duration barrier search bounds (lo, hi, payout_lo, payout_hi)
        dur_bounds = {
            2 : (0.50,  5.0,  0.46, 0.50),
            3 : (0.80,  6.0,  0.46, 0.50),
            4 : (1.00,  7.0,  0.46, 0.50),
            5 : (1.20,  8.0,  0.46, 0.50),
            6 : (1.40,  9.0,  0.46, 0.50),
            7 : (1.60, 10.5,  0.46, 0.50),
            8 : (1.80, 12.0,  0.46, 0.50),
            9 : (2.00, 13.5,  0.46, 0.50),
            10: (2.20, 15.0,  0.46, 0.50),
            12: (2.60, 17.5,  0.46, 0.50),
            15: (3.00, 21.0,  0.46, 0.50),
        }
        for dur in durations:
            lo, hi, p_lo, p_hi = dur_bounds.get(dur, (0.5, 8.0, 0.46, 0.50))
            best = fallback; best_pr = None
            for _ in range(10):
                mid = round((lo + hi) / 2, 2)
                pr  = _probe(dur, mid)
                if pr is None: break
                if best_pr is None or abs(pr - 0.485) < abs(best_pr - 0.485):
                    best = mid; best_pr = pr
                if p_lo <= pr <= p_hi: break
                if pr < p_lo: hi = mid
                else:         lo = mid
                time.sleep(0.3)

            self.barrier_table[dur] = best
            if best_pr is not None:
                self.payout_table[dur] = float(best_pr)
            log.info("[BarrierCal] %s %dm → barrier=±%.2f  payout=%.1f%%",
                     symbol, dur, best,
                     (best_pr or 0) * 100)

        log.info("[BarrierCal] Done: %s",
                 {f"{k}m": f"±{v:.2f}" for k, v in self.barrier_table.items()})
        self._calibrating = False



# ===========================================================================
# SYMBOL WORKER
# Wraps a LiveTrader for a single symbol. Evaluates MC every tick and
# reports the best (p_blend, duration, sigma) to the arbiter for selection.
# ===========================================================================

class SymbolWorker:
    """
    One SymbolWorker per symbol. Maintains:
      - its own tick buffer (seeded with historical ticks)
      - its own MonteCarloPricer and barrier calibration table
      - reports candidate signals to the SymbolArbiter

    The worker does NOT fire trades itself. It calls arbiter.try_trade(...)
    which grants entry only to the best-p symbol when no trade is live.
    """

    def __init__(self, symbol: str, cfg: dict, arbiter, staker, sprt, tlog):
        self.symbol   = symbol
        self.cfg      = cfg
        self.arbiter  = arbiter
        self.profile  = get_profile(symbol)
        self.pricer   = MonteCarloPricer(cfg)
        self.staker   = staker
        self.sprt     = sprt
        self.logger   = tlog

        # Override tpm if profile specifies it
        if self.profile.get("tpm_hint"):
            self.pricer._tpm_cfg = float(self.profile["tpm_hint"])

        # Tick buffer
        self.tick_buf  = deque(maxlen=5000)

        # Barrier / payout tables
        self.barrier_table : dict = {}
        self.payout_table  : dict = {}

        # Signal persistence
        self._persist_count = 0
        self._persist_best_p = 0.0
        self._persist_dur   = None

        # Live state
        self.live_tick_count = 0
        self.post_trade_tick = 0
        self._calibrating    = False
        self._last_p         = 0.0
        self._last_ev        = 0.0
        self._last_dur       = None
        self._last_sigma     = 0.0

        # Connection
        self._conn   : DerivWSManager | None = None
        self._running = False

    # ── Startup ────────────────────────────────────────────────────────────

    def seed(self, ticks: list):
        """Seed tick buffer from historical data."""
        self.tick_buf.extend(ticks)
        log.info("[%s] Seeded with %d historical ticks.", self.symbol, len(ticks))

    def start(self):
        self._running = True
        self._conn = DerivWSManager(
            app_id           = self.cfg["app_id"],
            on_open_cb       = self._on_open,
            on_message_cb    = self._on_message,
            on_disconnect_cb = self._on_disconnect,
            name             = self.symbol,
        )
        self._conn.start()

    def stop(self):
        self._running = False
        if self._conn:
            self._conn.stop()

    # ── WebSocket callbacks ────────────────────────────────────────────────

    def _on_open(self, ws):
        self._conn.safe_send({"authorize": self.cfg["api_token"]})

    def _on_disconnect(self):
        pass   # DerivWSManager handles reconnect automatically

    def _on_message(self, ws, raw):
        msg = json.loads(raw)
        mt  = msg.get("msg_type", "")

        if mt == "ping":
            return
        elif mt == "authorize":
            log.info("[%s] Authorised. Starting tick stream + calibration.", self.symbol)
            ws.send(json.dumps({
                "ticks": self.symbol,
                "subscribe": 1,
            }))
            threading.Thread(
                target=self._calibrate_barriers,
                args=(ws,),
                daemon=True,
                name=f"BarCal-{self.symbol}"
            ).start()
        elif mt == "tick":
            self._on_tick(msg)

    # ── Tick processing ────────────────────────────────────────────────────

    def _on_tick(self, msg):
        t = msg.get("tick", {})
        self.tick_buf.append({
            "timestamp": float(t.get("epoch", time.time())),
            "price"    : float(t.get("quote", 0)),
        })
        self.live_tick_count += 1

        # Block evaluation while trade is live or calibrating
        if self.arbiter.trade_live or self._calibrating:
            return

        min_ticks = self.profile.get("min_ticks", self.cfg.get("min_ticks", 300))
        if len(self.tick_buf) < min_ticks:
            return

        # Post-trade cooldown (per-symbol)
        min_gap = self.cfg.get("min_ticks_between_trades", 60)
        if self.live_tick_count - self.post_trade_tick < min_gap:
            return

        self._evaluate()

    def _evaluate(self):
        cfg      = self.cfg
        ev_floor = cfg.get("ev_confidence_floor", 0.673)
        ev_thr   = cfg.get("min_ev_threshold",    0.001)
        durations = cfg.get("hold_durations",      [2,3,4,5,6])
        fallback  = cfg.get("expiryrange_barrier", 2.97)
        required  = cfg.get("signal_persistence_ticks", 2)

        if not self.barrier_table:
            return  # calibration hasn't completed yet

        shared_sigma = self.pricer.ewma_sigma(self.tick_buf)
        tpm = self.pricer._tpm_cfg or MonteCarloPricer.detect_tpm(self.tick_buf)

        best_p = 0.0; best_dur = None; best_sigma = 0.0; best_T = 0
        for dur in durations:
            if LiveTrader._dur_crosses_midnight(dur):
                continue
            barrier = self.barrier_table.get(dur, fallback)
            p, sigma, T = self.pricer.simulate_with_sigma(
                self.tick_buf, barrier, dur, shared_sigma, tpm)
            if p > best_p:
                best_p = p; best_dur = dur
                best_sigma = sigma; best_T = T

        self._last_sigma = best_sigma
        cached_payout = self.payout_table.get(best_dur, cfg.get("min_payout_pct", 0.48))
        ev = MonteCarloPricer.ev(best_p, cached_payout)
        self._last_p = best_p; self._last_ev = ev; self._last_dur = best_dur

        if best_p < ev_floor or ev < ev_thr:
            self._persist_count = max(0, self._persist_count - 1)
            return

        # Persistence
        if best_dur == self._persist_dur:
            self._persist_count += 1
            if best_p > self._persist_best_p:
                self._persist_best_p = best_p
        else:
            self._persist_count  = 1
            self._persist_best_p = best_p
            self._persist_dur    = best_dur

        if self._persist_count >= required:
            # Submit to arbiter — only fires if we have the best p across all symbols
            self.arbiter.try_trade(
                worker       = self,
                symbol       = self.symbol,
                p_blend      = self._persist_best_p,
                best_dur     = best_dur,
                best_sigma   = best_sigma,
                cached_payout= cached_payout,
                ev           = ev,
            )

    # ── Barrier calibration ────────────────────────────────────────────────

    def _calibrate_barriers(self, ws):
        if self._calibrating:
            return
        self._calibrating = True
        cfg      = self.cfg
        symbol   = self.symbol
        durations = cfg.get("hold_durations", [2,3,4,5,6])
        fallback  = cfg.get("expiryrange_barrier", 2.97)
        bounds    = self.profile.get("barrier_bounds", {})

        log.info("[BarrierCal] Calibrating %s for %d durations ...",
                 symbol, len(durations))

        import queue as _queue

        def _probe_once(duration, barrier):
            q = _queue.Queue()
            def _oo(w): 
                w.send(json.dumps({"authorize": cfg["api_token"]}))
            def _om(w, raw):
                m = json.loads(raw)
                mt = m.get("msg_type","")
                if mt == "authorize":
                    w.send(json.dumps({
                        "proposal": 1, "amount": 1, "basis": "stake",
                        "contract_type": "EXPIRYRANGE",
                        "currency": cfg.get("currency","USD"),
                        "duration": duration, "duration_unit": "m",
                        "symbol": symbol,
                        "barrier": f"+{barrier:.2f}",
                        "barrier2": f"-{barrier:.2f}",
                    }))
                elif mt == "proposal":
                    if "error" in m:
                        q.put(None)
                    else:
                        ask   = float(m["proposal"].get("ask_price", 0))
                        payout= float(m["proposal"].get("payout", ask * 1.486))
                        q.put(round(payout / ask - 1, 4) if ask > 0 else None)
                    ws2.close()
            ws2 = websocket.WebSocketApp(
                f"{self.WS_URL}?app_id={cfg['app_id']}",
                on_open=_oo, on_message=_om,
                on_error=lambda *_: q.put(None),
                on_close=lambda *_: None,
            )
            t = threading.Thread(target=ws2.run_forever, daemon=True)
            t.start()
            try:
                return q.get(timeout=15)
            except Exception:
                return None
            finally:
                try: ws2.close()
                except: pass
                t.join(timeout=3)

        def _probe(duration, barrier, retries=3):
            for attempt in range(retries):
                r = _probe_once(duration, barrier)
                if r is not None: return r
                time.sleep(2 ** attempt)
            return None

        for dur in durations:
            lo, hi, p_lo, p_hi = bounds.get(dur, (0.5, 8.0, 0.46, 0.50))
            best = fallback; best_pr = None
            for _ in range(10):
                mid = round((lo + hi) / 2, 2)
                pr  = _probe(dur, mid)
                if pr is None: break
                if best_pr is None or abs(pr - 0.485) < abs(best_pr - 0.485):
                    best = mid; best_pr = pr
                if p_lo <= pr <= p_hi: break
                if pr < p_lo: hi = mid
                else:         lo = mid
                time.sleep(0.3)

            self.barrier_table[dur] = best
            if best_pr is not None:
                self.payout_table[dur] = float(best_pr)
            log.info("[BarrierCal] %s %dm → barrier=±%.2f  payout=%.1f%%",
                     symbol, dur, best, (best_pr or 0)*100)

        log.info("[BarrierCal] %s Done: %s", symbol,
                 {f"{k}m": f"±{v:.2f}" for k,v in self.barrier_table.items()})
        self._calibrating = False

    WS_URL = "wss://ws.binaryws.com/websockets/v3"

    def on_trade_complete(self):
        """Called by arbiter after a trade settles — reset cooldown."""
        self.post_trade_tick = self.live_tick_count
        self._persist_count  = 0
        # Recalibrate barriers in background after every trade
        if self._conn and self._conn._ws:
            threading.Thread(
                target=self._calibrate_barriers,
                args=(self._conn._ws,),
                daemon=True,
                name=f"BarCal-{self.symbol}"
            ).start()


# ===========================================================================
# SYMBOL ARBITER
# Receives candidate signals from all SymbolWorkers.
# Picks the best p_blend, enforces one-trade-at-a-time globally,
# and executes the trade through the LiveTrader.
# ===========================================================================

class SymbolArbiter:
    """
    Receives try_trade() calls from any SymbolWorker.
    Fires the trade only if:
      1. No trade is currently live (trade_live == False)
      2. This symbol has the highest p_blend of all pending candidates
         in the current evaluation window (SELECTION_WINDOW_SECS)

    The arbiter uses the LiveTrader's trade execution machinery
    (proposal → buy → result) for whichever symbol wins.
    """

    SELECTION_WINDOW_SECS = 2.0  # window to collect competing signals before picking best

    def __init__(self, cfg, staker, sprt, tlog):
        self.cfg     = cfg
        self.staker  = staker
        self.sprt    = sprt
        self.tlog    = tlog
        self._lock   = threading.Lock()
        self.trade_live = False
        self._balance   : float = 0.0   # updated by TradeExecutor and auth responses

        # pending candidates in current window: {symbol: (p_blend, dur, sigma, payout, ev, worker)}
        self._candidates   : dict = {}
        self._window_timer : threading.Timer | None = None

    def try_trade(self, worker, symbol, p_blend, best_dur,
                  best_sigma, cached_payout, ev):
        """Called by a SymbolWorker when it has a qualifying signal."""
        with self._lock:
            if self.trade_live:
                return   # trade already live — ignore

            # Register this symbol as a candidate
            self._candidates[symbol] = (p_blend, best_dur, best_sigma,
                                        cached_payout, ev, worker)

            # Start/reset the selection window timer
            if self._window_timer:
                self._window_timer.cancel()
            self._window_timer = threading.Timer(
                self.SELECTION_WINDOW_SECS,
                self._select_and_fire
            )
            self._window_timer.daemon = True
            self._window_timer.start()

    def _select_and_fire(self):
        """Pick the best candidate and fire the trade."""
        with self._lock:
            if self.trade_live or not self._candidates:
                self._candidates.clear()
                return

            # Pick highest p_blend
            best_sym = max(self._candidates,
                           key=lambda s: self._candidates[s][0])
            p_blend, best_dur, best_sigma, cached_payout, ev, worker =                 self._candidates[best_sym]
            self._candidates.clear()
            self.trade_live = True

        log.info("[Arbiter] Selected %s  p=%.4f  ev=%+.4f  dur=%dm  "
                 "(from %d candidates)",
                 best_sym, p_blend, ev, best_dur,
                 len(self._candidates) + 1)

        # Execute trade via a dedicated TradeExecutor
        executor = TradeExecutor(
            symbol       = best_sym,
            p_blend      = p_blend,
            best_dur     = best_dur,
            best_sigma   = best_sigma,
            cached_payout= cached_payout,
            ev           = ev,
            worker       = worker,
            arbiter      = self,
            cfg          = self.cfg,
            staker       = self.staker,
            sprt         = self.sprt,
            tlog         = self.tlog,
        )
        threading.Thread(
            target=executor.execute,
            daemon=True,
            name=f"Trade-{best_sym}"
        ).start()

    def on_trade_done(self, worker, win: bool, profit: float, new_balance: float):
        """Called by TradeExecutor when a trade settles."""
        self.trade_live = False
        worker.on_trade_complete()
        log.info("[Arbiter] Trade settled — %s  profit=%+.2f  balance=%.2f",
                 "WIN" if win else "LOSS", profit, new_balance)


# ===========================================================================
# TRADE EXECUTOR
# Handles the proposal → buy → result flow for a single selected trade.
# Runs in its own thread. Reports result back to the SymbolArbiter.
# ===========================================================================

class TradeExecutor:
    WS_URL = "wss://ws.binaryws.com/websockets/v3"

    def __init__(self, symbol, p_blend, best_dur, best_sigma,
                 cached_payout, ev, worker, arbiter, cfg,
                 staker, sprt, tlog):
        self.symbol        = symbol
        self.p_blend       = p_blend
        self.best_dur      = best_dur
        self.best_sigma    = best_sigma
        self.cached_payout = cached_payout
        self.ev            = ev
        self.worker        = worker
        self.arbiter       = arbiter
        self.cfg           = cfg
        self.staker        = staker
        self.sprt          = sprt
        self.tlog          = tlog

    def execute(self):
        cfg = self.cfg
        import queue as _queue

        balance = self.arbiter._balance
        if not balance or balance <= 0:
            log.warning("[TradeExec/%s] No balance — aborting.", self.symbol)
            self.arbiter.on_trade_done(self.worker, False, 0.0, 0.0)
            return

        stake   = self.staker.next_stake(self.p_blend, balance, self.cached_payout)
        barrier = self.worker.barrier_table.get(
            self.best_dur, cfg.get("expiryrange_barrier", 2.97))

        log.info("[TradeExec/%s] Stake=%.2f  dur=%dm  barrier=±%.2f  p=%.4f",
                 self.symbol, stake, self.best_dur, barrier, self.p_blend)

        q      = _queue.Queue()
        result = {}

        def _oo(ws):
            ws.send(json.dumps({"authorize": cfg["api_token"]}))

        def _om(ws, raw):
            m  = json.loads(raw)
            mt = m.get("msg_type", "")
            if mt == "authorize":
                ws.send(json.dumps({
                    "proposal": 1, "amount": stake, "basis": "stake",
                    "contract_type": "EXPIRYRANGE",
                    "currency": cfg.get("currency","USD"),
                    "duration": self.best_dur, "duration_unit": "m",
                    "symbol": self.symbol,
                    "barrier" : f"+{barrier:.2f}",
                    "barrier2": f"-{barrier:.2f}",
                }))
            elif mt == "proposal":
                if "error" in m:
                    log.warning("[TradeExec/%s] Proposal error: %s",
                                self.symbol, m["error"].get("message","?"))
                    q.put({"error": True})
                    ws.close(); return
                pid    = m["proposal"]["id"]
                payout = float(m["proposal"].get("payout", stake * 1.486))
                ev_real = MonteCarloPricer.ev(self.p_blend, payout / stake - 1
                                              if stake > 0 else self.cached_payout)
                log.info("[TradeExec/%s] Proposal OK  payout=%.1f%%  EV=%+.4f",
                         self.symbol, (payout/stake-1)*100 if stake else 0, ev_real)
                ws.send(json.dumps({"buy": pid, "price": stake}))
            elif mt == "buy":
                if "error" in m:
                    log.warning("[TradeExec/%s] Buy error: %s",
                                self.symbol, m["error"].get("message","?"))
                    q.put({"error": True})
                    ws.close(); return
                cid = m["buy"]["contract_id"]
                result["stake"]  = stake
                result["cid"]    = cid
                result["payout"] = float(m["buy"].get("payout", stake * 1.486))
                log.info("[TradeExec/%s] Opened  cid=%s  paid=%.2f",
                         self.symbol, cid, stake)
                ws.send(json.dumps({"proposal_open_contract": 1,
                                    "contract_id": cid, "subscribe": 1}))
            elif mt == "proposal_open_contract":
                poc = m.get("proposal_open_contract", {})
                if poc.get("is_sold") or poc.get("status") in ("won","lost"):
                    win    = poc.get("profit", 0) > 0
                    profit = float(poc.get("profit", 0))
                    new_bal= float(poc.get("balance_after",
                                           balance + profit))
                    result.update({"win": win, "profit": profit,
                                   "balance_after": new_bal})
                    # Update shared balance in arbiter
                    self.arbiter._balance = new_bal
                    q.put(result)
                    ws.close()

        ex_ws = websocket.WebSocketApp(
            f"{self.WS_URL}?app_id={cfg['app_id']}",
            on_open=_oo, on_message=_om,
            on_error=lambda *_: q.put({"error": True}),
            on_close=lambda *_: None,
        )
        t = threading.Thread(target=ex_ws.run_forever, daemon=True)
        t.start()

        try:
            r = q.get(timeout=max(self.best_dur * 60 + 30, 120))
        except Exception:
            log.warning("[TradeExec/%s] Timed out waiting for result.", self.symbol)
            r = {"error": True}
        finally:
            try: ex_ws.close()
            except: pass
            t.join(timeout=3)

        win    = r.get("win", False)
        profit = r.get("profit", 0.0)
        new_b  = r.get("balance_after", balance)

        if not r.get("error"):
            self.sprt.update(win)
            self.tlog.log({
                "timestamp"  : datetime.utcnow().isoformat(),
                "symbol"     : self.symbol,
                "duration"   : self.best_dur,
                "p_blend"    : round(self.p_blend, 4),
                "ev"         : round(self.ev, 4),
                "stake"      : round(stake, 2),
                "outcome"    : "WIN" if win else "LOSS",
                "profit"     : round(profit, 2),
                "balance_after": round(new_b, 2),
            })

        self.arbiter.on_trade_done(self.worker, win, profit, new_b)

# ===========================================================================
# MAIN
# ===========================================================================

def main():
    cfg     = CONFIG
    symbols = cfg.get("symbols", ["RDBEAR"])

    log.info("=" * 65)
    log.info("  DERIV RANGE BOT  v11  —  Multi-Symbol  MC + Jump-Diffusion")
    log.info("  Symbols  : %s", ", ".join(symbols))
    log.info("  MC paths : %d   vol_window : %d ticks   α=%.2f   terminal_w=%.1f",
             cfg["mc_n_paths"], cfg["mc_vol_window"], cfg["mc_ewma_alpha"],
             cfg["mc_terminal_weight"])
    log.info("  JD       : threshold=%.1fσ  fit_window=%d  min_jumps=%d  weight=%.2f",
             cfg["jd_jump_threshold"], cfg["jd_fit_window"],
             cfg["jd_min_jumps"], cfg["jd_weight"])
    log.info("  p_floor  : %.3f  min_ev: %.3f",
             cfg["ev_confidence_floor"], cfg["min_ev_threshold"])
    log.info("  Persist  : %d ticks  |  One trade at a time globally",
             cfg["signal_persistence_ticks"])
    log.info("=" * 65)

    if not cfg.get("api_token"):
        log.error("DERIV_API_TOKEN not set. Exiting.")
        sys.exit(1)

    # ── Phase 1: Collect historical ticks for all symbols ─────────────────
    log.info("\n>> PHASE 1 — Historical tick collection (%d symbols)", len(symbols))
    all_initial_ticks = {}
    for sym in symbols:
        data_path = os.path.join(cfg["data_dir"], f"ticks_{sym}.csv")
        existing  = pd.read_csv(data_path) if os.path.isfile(data_path) else pd.DataFrame()
        done      = threading.Event()
        HistoricalCollector(sym, cfg, done, existing).start()
        done.wait()
        df = pd.read_csv(data_path) if os.path.isfile(data_path) else pd.DataFrame()
        ticks = df.tail(5000).to_dict("records") if not df.empty else []
        all_initial_ticks[sym] = ticks
        log.info("[Main/%s] Loaded %d historical ticks.", sym, len(ticks))

    # ── Phase 2: Fetch initial balance ────────────────────────────────────
    log.info("\n>> PHASE 2 — Fetching account balance")
    import queue as _bq
    bq = _bq.Queue()
    def _b_oo(ws):
        ws.send(json.dumps({"authorize": cfg["api_token"]}))
    def _b_om(ws, raw):
        m = json.loads(raw)
        if m.get("msg_type") == "authorize":
            bal = float(m.get("authorize", {}).get("balance", 1.0))
            bq.put(bal); ws.close()
    bws = websocket.WebSocketApp(
        f"wss://ws.binaryws.com/websockets/v3?app_id={cfg['app_id']}",
        on_open=_b_oo, on_message=_b_om,
        on_error=lambda *_: bq.put(1.0), on_close=lambda *_: None,
    )
    threading.Thread(target=bws.run_forever, daemon=True).start()
    try:
        initial_balance = bq.get(timeout=10)
    except Exception:
        initial_balance = 1.0
    log.info("[Main] Account balance: %.2f USD", initial_balance)

    # ── Phase 3: Start multi-symbol trading ───────────────────────────────
    log.info("\n>> PHASE 3 — Starting live trading on %d symbols", len(symbols))

    staker  = KellyStaker(cfg)
    sprt    = SPRTMonitor(p0=cfg["sprt_p0"], p1=cfg["sprt_p1"],
                          alpha=cfg["sprt_alpha"], beta=cfg["sprt_beta"])
    tlog    = TradeLogger(cfg["trade_log"])
    arbiter = SymbolArbiter(cfg, staker, sprt, tlog)
    arbiter._balance = initial_balance

    workers = []
    for sym in symbols:
        w = SymbolWorker(sym, cfg, arbiter, staker, sprt, tlog)
        w.seed(all_initial_ticks.get(sym, []))
        workers.append(w)

    import signal as _sig
    def _shutdown(s, f):
        log.info("\n[Main] Shutting down all workers ...")
        for w in workers:
            w.stop()
    _sig.signal(_sig.SIGINT, _shutdown)
    try:
        _sig.signal(_sig.SIGTERM, _shutdown)
    except (OSError, ValueError):
        pass

    for w in workers:
        w.start()
        log.info("[Main] Started worker: %s", w.symbol)
        time.sleep(0.5)   # stagger startups slightly

    log.info("[Main] All %d workers running. Bot is live.", len(workers))

    # Keep main thread alive
    try:
        while any(w._running for w in workers):
            time.sleep(5)
    except KeyboardInterrupt:
        _shutdown(None, None)

    log.info("\n[Main] Session complete.")
    log.info("  SPRT : %s", sprt.summary())
    log.info("  Trades logged → %s", cfg["trade_log"])


if __name__ == "__main__":
    main()
