# ไฟล์ calculate_modules/entry_timing.py
"""
calculate_modules/entry_timing.py
--------------------------------------------------------------------
Institutional Quantitative Framework (IKB v2.0) - Audit Response Build

ยึดโครงสร้างไฟล์เดิม (v1.3) ทุกประการ: ลำดับฟังก์ชัน ชื่อฟังก์ชัน
`_empty_result()` / `classify_signal()` / `compute_price_levels()` /
`calculate_timing_module()` และชื่อคีย์ผลลัพธ์เดิมถูกรักษาไว้ครบ
การเปลี่ยนแปลงคือการแก้ตรรกะภายใน และ "เพิ่ม" คีย์ใหม่เท่านั้น
ไม่มีการลบคีย์เดิมออก โมดูลอื่นที่อ่านผลลัพธ์อยู่จึงไม่พัง

Changelog vs v1.3 (ตอบ Feedback_Action_Module_3 ครบ 5 ประเด็น + 1 ที่พบเพิ่ม):

- ISSUE 01: `_empty_result()` ไม่คืน 50.0 / NEUTRAL อีกต่อไป ทุกฟิลด์เชิงตัวเลข
  คืน None และเพิ่ม 'data_status' = 'INSUFFICIENT_DATA' ให้ UI แสดง
  "ข้อมูลไม่เพียงพอ" แทนหน้าปัดสีฟ้า NEUTRAL
  แยกการตรวจเป็น Hard Gate (`_check_data_gate`) และ Soft Gate
  (`min_data_completeness`) เพื่อไม่ให้ขัดกับ ISSUE 02

- ISSUE 02: ตัวหารของเสา Trend / Momentum ถูก "ตรึง" ด้วย
  cfg.trend_criteria_count / cfg.momentum_criteria_count (ค่าเริ่มต้น 3/3)
  ตัวชี้วัดที่หาย = 0 คะแนน ไม่ลดตัวหารอีกต่อไป
  เพิ่ม `_resolve_column()` รองรับชื่อคอลัมน์หลายแบบ (ADX / ADX14 / ADX_14 ...)
  เพราะสาเหตุรากเหง้าจริงของอาการ 50 -> 55 คือชื่อคอลัมน์ไม่ตรง

- ISSUE 03: `compute_risk_reward()` แยก 2 กรณีออกจากกันชัดเจน
  คำนวณไม่ได้ -> rr_ratio = None, rr_status = 'NOT_COMPUTABLE'
  คำนวณได้แต่ต่ำ -> rr_ratio = ค่าจริง (เช่น 0.19), rr_status = 'COMPUTED', score = 0

- ISSUE 04: `compute_volume_context()` ใช้แท่ง -21 ถึง -2 (`iloc[-21:-1]`)
  เป็นฐานค่าเฉลี่ย แยกแท่งปัจจุบัน (-1) ออกจากอดีตอย่างเด็ดขาด

- ISSUE 05: ย้ายค่าเชิง Business Judgment ทั้งหมดเข้า `TimingConfig`
  (frozen dataclass) รองรับ from_dict / from_json / from_env
  ฟังก์ชันหลักรับ `config=` โดยยังเรียกแบบเดิม (ไม่ส่ง config) ได้ตามปกติ

- ISSUE 06 (พบเพิ่มระหว่างแก้): ป้องกันตัวหาร RR เล็กผิดปกติด้วย
  `min_downside_pct` (ค่าเริ่มต้น 1.0%) กรณีราคาเกาะแนวรับจนระยะ stop
  แคบผิดธรรมชาติ (เคส KCE = 90.5 : 1) จะคืน NOT_COMPUTABLE แทนการให้คะแนนเต็ม

- SQLITE COMPATIBILITY: 'missing_fields' และ 'config_snapshot' ถูกแปลงเป็น
  JSON string ก่อนคืนค่า เพื่อไม่ให้เกิด sqlite3.ProgrammingError
  ตอนบันทึกลง cis_summary_scores
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, fields

import numpy as np
import pandas as pd

from calculate_modules.common import clean_float

ALGO_VERSION = "entry_timing/v2.0"
MIN_BARS_FOR_1Y_TREND = 200


# =====================================================================
# ISSUE 05: Parameterization layer
# =====================================================================
@dataclass(frozen=True)
class TimingConfig:
    """ศูนย์รวมค่าเชิง Business Judgment ของ Module 3

    ค่าเริ่มต้นทั้งหมด = ค่าที่ production v1.3 ใช้อยู่เดิม (backward compatible)
    ทุกค่า override ได้จาก dict / JSON / environment variable เพื่อให้ทำ
    grid search + walk-forward backtest บน SET ได้โดยไม่ต้องแก้โค้ด
    """

    # ---- น้ำหนักเสา (รวม 100) ----
    trend_weight: float = 40.0
    momentum_weight: float = 30.0
    rr_weight: float = 30.0

    # ---- ISSUE 02: ตัวหารคงที่ ห้ามลดอัตโนมัติ ----
    trend_criteria_count: int = 3
    momentum_criteria_count: int = 3
    total_criteria_count: int = 7

    # ---- เกณฑ์ตัวชี้วัด ----
    adx_threshold: float = 25.0
    ma200_window: int = 200
    volume_lookback: int = 20
    resistance_window: int = 60
    resistance_window_long: int = 120

    # ---- ขั้นบันได Risk / Reward ----
    rr_tier_high: float = 2.0
    rr_tier_mid: float = 1.5
    rr_tier_low: float = 1.0

    # ---- เกณฑ์แปลผลคะแนนรวม ----
    score_bullish: float = 70.0
    score_neutral: float = 40.0

    # ---- ISSUE 01: ประตูคุณภาพข้อมูล ----
    min_bars_hard_gate: int = 20
    min_data_completeness: float = 0.5

    # ---- ISSUE 06: ป้องกันตัวหาร RR เล็กผิดปกติ (ตั้ง 0.0 = ปิดการป้องกัน) ----
    min_downside_pct: float = 1.0

    @classmethod
    def from_dict(cls, overrides):
        if not overrides:
            return cls()
        valid = {f.name: f.type for f in fields(cls)}
        clean = {}
        for key, value in overrides.items():
            if key not in valid:
                continue
            clean[key] = int(value) if valid[key] in (int, "int") else float(value)
        return cls(**clean)

    @classmethod
    def from_json(cls, path_or_text):
        if path_or_text is None:
            return cls()
        if os.path.exists(str(path_or_text)):
            with open(path_or_text, "r", encoding="utf-8") as fh:
                return cls.from_dict(json.load(fh))
        return cls.from_dict(json.loads(path_or_text))

    @classmethod
    def from_env(cls, prefix="IKB_TIMING_"):
        overrides = {}
        for f in fields(cls):
            raw = os.getenv(f"{prefix}{f.name.upper()}")
            if raw is not None:
                overrides[f.name] = raw
        return cls.from_dict(overrides)

    def as_json(self):
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


DEFAULT_CONFIG = TimingConfig()


# =====================================================================
# ISSUE 02 (root cause): column resolution
# =====================================================================
COLUMN_ALIASES = {
    'close': ('close', 'Close', 'CLOSE', 'close_price'),
    'high': ('high', 'High', 'HIGH'),
    'low': ('low', 'Low', 'LOW'),
    'volume': ('volume', 'Volume', 'vol', 'Vol', 'VOLUME'),
    'ema20': ('EMA20', 'ema20', 'EMA_20', 'ema_20'),
    'ema50': ('EMA50', 'ema50', 'EMA_50', 'ema_50'),
    'ma200': ('MA200', 'ma200', 'MA_200', 'SMA200', 'sma200'),
    'rsi': ('RSI14', 'RSI', 'rsi14', 'rsi', 'RSI_14'),
    'macd': ('MACD', 'macd', 'MACD_line', 'macd_line'),
    'adx': ('ADX', 'ADX14', 'ADX_14', 'adx', 'adx14'),
}


def _resolve_column(df, key):
    """คืนชื่อคอลัมน์จริงที่พบใน DataFrame ตาม alias ที่รองรับ (ไม่พบ = None)

    สาเหตุรากเหง้าจริงของอาการคะแนนกระโดด 50 -> 55 คือคอลัมน์ ADX ในชุดข้อมูล
    ใช้ชื่อ 'ADX14' แต่โค้ดเดิมอ่าน 'ADX' ตรง ๆ จึงถือว่า "ไม่มีข้อมูล" ทุกครั้ง
    แล้วไปลดตัวหารตาม ISSUE 02 ซ้ำอีกชั้นหนึ่ง
    """
    for name in COLUMN_ALIASES.get(key, ()):
        if name in df.columns:
            return name
    return None


def _latest(latest_row, col):
    if col is None:
        return None
    value = latest_row.get(col)
    return None if not pd.notna(value) else value


def _check_data_gate(df_price_ticker, config=None):
    """ISSUE 01 Hard Gate: เงื่อนไขที่ทำให้ 'ประเมินไม่ได้เลย' (ไม่ใช่แค่ตัวชี้วัดขาด)"""
    cfg = config or DEFAULT_CONFIG

    if df_price_ticker is None:
        return False, "ไม่ได้รับชุดข้อมูลราคา"
    if not isinstance(df_price_ticker, pd.DataFrame) or df_price_ticker.empty:
        return False, "ชุดข้อมูลราคาว่างเปล่า"

    close_col = _resolve_column(df_price_ticker, 'close')
    if close_col is None:
        return False, "ไม่พบคอลัมน์ราคาปิดในชุดข้อมูล"
    if len(df_price_ticker) < cfg.min_bars_hard_gate:
        return False, (f"มีข้อมูลเพียง {len(df_price_ticker)} แท่ง "
                       f"(ต้องการอย่างน้อย {cfg.min_bars_hard_gate} แท่ง)")

    last_close = df_price_ticker.iloc[-1].get(close_col)
    if not pd.notna(last_close) or float(last_close) <= 0:
        return False, "ราคาปิดล่าสุดไม่สมบูรณ์หรือไม่เป็นบวก"

    return True, ""


def _empty_result(reason="ข้อมูลไม่เพียงพอสำหรับการประเมินเชิงลึก",
                  missing_fields=None, config=None):
    """ISSUE 01: ไม่คืนค่า 50.0 / NEUTRAL อีกต่อไป

    ฟิลด์เชิงตัวเลขทั้งหมดคืน None เพื่อบังคับให้ชั้น UI ตัดสินใจแสดงผล
    "ข้อมูลไม่เพียงพอ" อย่างชัดเจน — ไม่มีข้อมูล ไม่เท่ากับ ปานกลาง
    """
    cfg = config or DEFAULT_CONFIG
    sig = classify_signal(None, config=cfg)
    return {
        'timing_score': None, 'rsi': None, 'macd': None, 'adx': None,
        'ema20': None, 'ema50': None, 'ma200': None, 'trend_signal': sig['signal_legacy'],
        'resistance_60d': None, 'support_60d': None, 'pivot_point': None,
        'resistance_2': None, 'support_2': None, 'overall_signal': sig['overall_signal'],
        'status_label': sig['status_label'], 'status_color': sig['status_color'],
        'action_th': sig['action_th'], 'rr_ratio': None,
        'trend_score': None, 'mom_score': None, 'rr_score': None,
        'upside_pct': None, 'downside_pct': None, 'readiness': sig['readiness'],
        'summary_text': reason or sig['summary_text'],
        'k15_ok': False, 'k16_ok': False, 'k17_ok': False,
        'k18_ok': False, 'k19_ok': False, 'k20_ok': False, 'k_rr_ok': False,
        'k15_available': False, 'k16_available': False, 'k17_available': False,
        'k18_available': False, 'k19_available': False, 'k20_available': False,
        'k_rr_available': False,
        'trend_available_count': 0, 'mom_available_count': 0,
        'trend_criteria_count': cfg.trend_criteria_count,
        'mom_criteria_count': cfg.momentum_criteria_count,
        'total_criteria_count': cfg.total_criteria_count,
        'rr_status': 'NOT_COMPUTABLE', 'rr_status_reason': 'ยังไม่ได้ประเมิน',
        'vol_last': None, 'vol_avg20': None,
        'data_completeness': 0.0,
        'data_status': 'INSUFFICIENT_DATA',
        'data_status_reason': reason,
        'is_evaluated': False,
        'missing_fields': json.dumps(list(missing_fields or []), ensure_ascii=False),
        'algo_version': ALGO_VERSION,
        'config_snapshot': cfg.as_json(),
    }


def classify_signal(total_score, config=None):
    """Single source of truth ของการแปลคะแนนเป็นสถานะ (UI เรียกใช้ฟังก์ชันนี้)

    ISSUE 01: total_score = None คือสถานะ "ยังไม่ได้ประเมิน" ซึ่งต้องไม่ถูก
    ตีความเป็น NEUTRAL สีฟ้า จึงใช้สีเทา #64748B ที่ไม่ซ้ำกับสถานะใดในระบบ
    """
    cfg = config or DEFAULT_CONFIG

    if total_score is None:
        return dict(
            signal_legacy="NO_DATA", overall_signal="INSUFFICIENT DATA",
            status_label="ข้อมูลไม่เพียงพอ", status_color="#64748B",
            action_th="ระบบยังไม่ได้ประเมินหลักทรัพย์นี้", readiness="NO_DATA",
            summary_text="ข้อมูลไม่เพียงพอสำหรับการประเมินเชิงลึก "
                         "(ไม่มีข้อมูล ไม่เท่ากับ ปานกลาง)",
        )

    if total_score >= cfg.score_bullish:
        return dict(
            signal_legacy="BULLISH", overall_signal="BULLISH (Strong Buy)",
            status_label="STRONG BUY", status_color="#10B981",
            action_th="จังหวะซื้อได้เปรียบสูง", readiness="READY",
            summary_text="ราคายืนในโซนสะสมและโครงสร้างขาขึ้นแข็งแกร่ง พร้อมทยอยสะสม",
        )
    elif total_score >= cfg.score_neutral:
        return dict(
            signal_legacy="NEUTRAL", overall_signal="NEUTRAL",
            status_label="NEUTRAL", status_color="#38BDF8",
            action_th="แกว่งตัว รอดูสัญญาณยืนยัน", readiness="WAIT",
            summary_text="แม้ราคาจะอยู่ในโซนที่น่าสนใจ แต่สัญญาณทางเทคนิคยังไม่ยืนยันการกลับตัว "
                         "ควรรอการยืนยันจากปริมาณซื้อขายและแนวโน้มราคา",
        )
    else:
        return dict(
            signal_legacy="BEARISH", overall_signal="BEARISH (Avoid)",
            status_label="BEARISH", status_color="#EF4444",
            action_th="ขาลง ควรหลีกเลี่ยง", readiness="WAIT",
            summary_text="แนวโน้มหลักยังเป็นขาลงและโมเมนตัมอ่อนแอ "
                         "หลีกเลี่ยงการเข้าลงทุนจนกว่าจะเกิดสัญญาณกลับตัวชัดเจน",
        )


def compute_price_levels(price, df_price_ticker, high_col, low_col, config=None):
    """คงสูตรเดิมทุกประการ เปลี่ยนเฉพาะหน้าต่างเวลาให้ดึงจาก config ได้

    Invariant ที่ต้องเป็นจริงเสมอ: R2 >= R1 และ S2 <= S1
    """
    cfg = config or DEFAULT_CONFIG
    recent60 = df_price_ticker.tail(cfg.resistance_window)
    recent120 = df_price_ticker.tail(cfg.resistance_window_long)

    r1 = float(recent60[high_col].max()) if not recent60.empty else price * 1.05
    r2 = float(recent120[high_col].max()) if not recent120.empty else r1 * 1.05
    s1 = float(recent60[low_col].min()) if not recent60.empty else price * 0.95
    s2 = float(recent120[low_col].min()) if not recent120.empty else s1 * 0.95

    r2 = max(r2, r1)
    s2 = min(s2, s1)

    # Classic pivot point: prior bar's High/Low + current price (classic pivot
    # definition) instead of the 60-day rolling high/low, which was not a
    # standard pivot point.
    if len(df_price_ticker) >= 2:
        prev_bar = df_price_ticker.iloc[-2]
        prev_high = clean_float(prev_bar.get(high_col), default=price)
        prev_low = clean_float(prev_bar.get(low_col), default=price)
    else:
        prev_high = price
        prev_low = price

    pivot_point = round((prev_high + prev_low + price) / 3.0, 2)
    return round(r1, 2), round(r2, 2), round(s1, 2), round(s2, 2), pivot_point


def compute_volume_context(df_price_ticker, vol_col, config=None):
    """ISSUE 04: ฐานค่าเฉลี่ยคือแท่ง -21 ถึง -2 เท่านั้น

    แท่งปัจจุบัน (-1) ถูกกันออกจากการหาค่าเฉลี่ยในอดีตอย่างเด็ดขาด การเปรียบเทียบ
    จึงเป็น "วันนี้ vs. อดีต" จริง ไม่ใช่ "วันนี้ vs. อดีตที่มีวันนี้ปนอยู่"
    และจงใจ *ไม่* ใช้คอลัมน์ Volume_Avg20 ที่ pre-compute มา เพราะคอลัมน์นั้นเป็น
    rolling(20) ที่รวมแท่งปัจจุบันไว้แล้ว

    คืนค่า: (available, ok, vol_last, vol_avg20)
    """
    cfg = config or DEFAULT_CONFIG
    n = cfg.volume_lookback

    if vol_col is None or len(df_price_ticker) < n + 1:
        return False, False, None, None

    baseline = df_price_ticker[vol_col].iloc[-(n + 1):-1]
    vol_last_raw = df_price_ticker[vol_col].iloc[-1]

    if baseline.isna().all() or not pd.notna(vol_last_raw):
        return False, False, None, None

    vol_avg = float(baseline.mean())
    vol_last = float(vol_last_raw)

    if not np.isfinite(vol_avg) or vol_avg <= 0:
        return False, False, round(vol_last, 2), None

    return True, bool(vol_last >= vol_avg), round(vol_last, 2), round(vol_avg, 2)


def compute_risk_reward(price, r1, r2, s2, config=None):
    """ISSUE 03 + ISSUE 06: แยก "คำนวณไม่ได้" ออกจาก "คำนวณได้แต่ไม่คุ้ม"

    NOT_COMPUTABLE -> rr_ratio = None      (UI แสดง N/A พร้อมเหตุผล)
    COMPUTED       -> rr_ratio = ค่าจริง    (เช่น 0.19 แสดงตามจริง แต่ได้ 0 คะแนน)
    """
    cfg = config or DEFAULT_CONFIG

    # reward target upgrades to r2 once price has broken above r1, so a
    # breakout no longer collapses rr_score to 0.
    reward_target = r2 if price >= r1 else r1
    downside_risk = price - s2
    upside_reward = reward_target - price

    if price <= 0:
        return dict(rr_ratio=None, rr_score=0.0, rr_status='NOT_COMPUTABLE',
                    rr_status_reason='ราคาปัจจุบันไม่สมบูรณ์',
                    upside_pct=None, downside_pct=None,
                    k_rr_ok=False, k_rr_available=False)

    if downside_risk <= 0:
        return dict(rr_ratio=None, rr_score=0.0, rr_status='NOT_COMPUTABLE',
                    rr_status_reason='ราคาปัจจุบันอยู่ที่หรือต่ำกว่าแนวรับ 120 วัน '
                                     'จึงนิยามระยะความเสี่ยงไม่ได้',
                    upside_pct=None, downside_pct=None,
                    k_rr_ok=False, k_rr_available=False)

    downside_pct = round((downside_risk / price) * 100, 1)
    upside_pct = round((max(upside_reward, 0.0) / price) * 100, 1)

    # ISSUE 06: ตัวหารเล็กผิดปกติ -> RR พุ่งสูงเทียม (เคส KCE 18.20 vs S2 18.10 = 90.5 : 1)
    if cfg.min_downside_pct > 0 and (downside_risk / price) * 100 < cfg.min_downside_pct:
        return dict(rr_ratio=None, rr_score=0.0, rr_status='NOT_COMPUTABLE',
                    rr_status_reason=(f'ระยะถึงแนวรับเพียง {downside_pct:.1f}% '
                                      f'(ต่ำกว่าเกณฑ์ {cfg.min_downside_pct:.1f}%) '
                                      f'ตัวหารเล็กเกินกว่าจะนิยามความเสี่ยงได้'),
                    upside_pct=upside_pct, downside_pct=downside_pct,
                    k_rr_ok=False, k_rr_available=False)

    rr_ratio = round(max(upside_reward, 0.0) / downside_risk, 2)

    if rr_ratio >= cfg.rr_tier_high:
        rr_score = float(cfg.rr_weight)
    elif rr_ratio >= cfg.rr_tier_mid:
        rr_score = round(cfg.rr_weight * 2.0 / 3.0, 1)
    elif rr_ratio >= cfg.rr_tier_low:
        rr_score = round(cfg.rr_weight * 1.0 / 3.0, 1)
    else:
        rr_score = 0.0

    return dict(rr_ratio=rr_ratio, rr_score=float(rr_score), rr_status='COMPUTED',
                rr_status_reason='', upside_pct=upside_pct, downside_pct=downside_pct,
                k_rr_ok=bool(rr_score > 0), k_rr_available=True)


def calculate_timing_module(df_price_ticker, config=None):
    cfg = config or DEFAULT_CONFIG

    # ---- ISSUE 01: Hard Gate ----
    passed_gate, gate_reason = _check_data_gate(df_price_ticker, cfg)
    if not passed_gate:
        return _empty_result(reason=gate_reason, config=cfg)

    latest = df_price_ticker.iloc[-1]

    # ---- ISSUE 02: resolve columns by alias, ไม่อ่านชื่อคอลัมน์ตรง ๆ ----
    close_col = _resolve_column(df_price_ticker, 'close')
    high_col = _resolve_column(df_price_ticker, 'high') or close_col
    low_col = _resolve_column(df_price_ticker, 'low') or close_col
    vol_col = _resolve_column(df_price_ticker, 'volume')

    price = clean_float(latest.get(close_col), default=0.0)

    ema20_raw = _latest(latest, _resolve_column(df_price_ticker, 'ema20'))
    ema50_raw = _latest(latest, _resolve_column(df_price_ticker, 'ema50'))
    rsi_raw = _latest(latest, _resolve_column(df_price_ticker, 'rsi'))
    macd_raw = _latest(latest, _resolve_column(df_price_ticker, 'macd'))
    adx_raw = _latest(latest, _resolve_column(df_price_ticker, 'adx'))

    ema20_available = ema20_raw is not None
    ema50_available = ema50_raw is not None
    macd_available = macd_raw is not None
    adx_available = adx_raw is not None

    rsi = clean_float(rsi_raw, default=50.0)
    macd = clean_float(macd_raw, default=0.0)
    adx = clean_float(adx_raw, default=0.0)
    ema20 = clean_float(ema20_raw, default=price)
    ema50 = clean_float(ema50_raw, default=price)

    r1, r2, s1, s2, pivot_point = compute_price_levels(
        price, df_price_ticker, high_col, low_col, cfg
    )

    # --- Risk/Reward pillar (30 pts): ISSUE 03 + ISSUE 06 ---
    rr = compute_risk_reward(price, r1, r2, s2, cfg)
    rr_ratio = rr['rr_ratio']
    rr_score = rr['rr_score']
    k_rr_ok = rr['k_rr_ok']
    k_rr_available = rr['k_rr_available']
    upside_pct = rr['upside_pct']
    downside_pct = rr['downside_pct']

    # --- Trend pillar (40 pts): k15 short-term, k16 medium-term, k17 long-term ---
    k15_ok = bool(price > ema20) if ema20_available else False
    k16_available = ema20_available and ema50_available
    k16_ok = bool(ema20 > ema50) if k16_available else False

    # k17: real MA200 comparison. Prefer a precomputed MA200 column if the data
    # pipeline supplies one; otherwise fall back to a rolling 200-bar SMA.
    ma200_col = _resolve_column(df_price_ticker, 'ma200')
    if ma200_col is not None and pd.notna(latest.get(ma200_col)):
        ma200 = clean_float(latest.get(ma200_col), default=price)
        ma200_available = True
    elif len(df_price_ticker) >= cfg.ma200_window:
        ma200 = float(df_price_ticker[close_col].tail(cfg.ma200_window).mean())
        ma200_available = True
    else:
        ma200 = price
        ma200_available = False

    k17_available = ma200_available
    k17_ok = bool(price > ma200) if k17_available else False

    trend_avail = [ema20_available, k16_available, k17_available]
    trend_pass = [k15_ok, k16_ok, k17_ok]
    n_trend_available = sum(trend_avail)

    # ISSUE 02: ตัวหารตรึงที่ cfg.trend_criteria_count เสมอ
    # ตัวชี้วัดที่หาย = 0 คะแนน ไม่ใช่ถูกตัดออกจากตัวหาร
    trend_unit = cfg.trend_weight / float(cfg.trend_criteria_count)
    trend_score = round(sum(1 for p, a in zip(trend_pass, trend_avail) if a and p) * trend_unit, 1)

    # --- Momentum pillar (30 pts): k18 MACD, k19 ADX, k20 Volume ---
    k18_ok = bool(macd > 0) if macd_available else False
    k19_ok = bool(adx >= cfg.adx_threshold) if adx_available else False

    k20_available, k20_ok, vol_last, vol_avg20 = compute_volume_context(
        df_price_ticker, vol_col, cfg
    )

    mom_avail = [macd_available, adx_available, k20_available]
    mom_pass = [k18_ok, k19_ok, k20_ok]
    n_mom_available = sum(mom_avail)

    mom_unit = cfg.momentum_weight / float(cfg.momentum_criteria_count)
    mom_score = round(sum(1 for p, a in zip(mom_pass, mom_avail) if a and p) * mom_unit, 1)

    # --- ISSUE 01: Soft Gate ---
    availability = [
        ('EMA20', ema20_available), ('EMA50', k16_available), ('MA200', k17_available),
        ('MACD', macd_available), ('ADX', adx_available), ('Volume', k20_available),
        ('Risk/Reward', k_rr_available),
    ]
    n_available = sum(1 for _, a in availability if a)
    missing_fields = [name for name, a in availability if not a]
    data_completeness = round(n_available / float(cfg.total_criteria_count), 2)

    if data_completeness < cfg.min_data_completeness:
        reason = (f"ตัวชี้วัดพร้อมใช้ {n_available}/{cfg.total_criteria_count} "
                  f"(ต่ำกว่าเกณฑ์ {int(cfg.min_data_completeness * 100)}%) "
                  f"ขาด: {', '.join(missing_fields)}")
        return _empty_result(reason=reason, missing_fields=missing_fields, config=cfg)

    total_score = round(float(np.clip(trend_score + mom_score + rr_score, 0, 100)))
    sig = classify_signal(total_score, config=cfg)

    return {
        'timing_score': float(total_score), 'rsi': round(rsi, 1), 'macd': round(macd, 3),
        'adx': round(adx, 1), 'ema20': round(ema20, 2), 'ema50': round(ema50, 2),
        'ma200': round(ma200, 2),
        'trend_signal': sig['signal_legacy'], 'resistance_60d': r1, 'support_60d': s1,
        'pivot_point': pivot_point, 'resistance_2': r2, 'support_2': s2,
        'overall_signal': sig['overall_signal'], 'status_label': sig['status_label'],
        'status_color': sig['status_color'], 'action_th': sig['action_th'],
        'rr_ratio': rr_ratio, 'trend_score': trend_score, 'mom_score': mom_score,
        'rr_score': rr_score,
        'upside_pct': upside_pct, 'downside_pct': downside_pct,
        'readiness': sig['readiness'],
        'summary_text': sig['summary_text'],
        'k15_ok': k15_ok, 'k16_ok': k16_ok, 'k17_ok': k17_ok,
        'k18_ok': k18_ok, 'k19_ok': k19_ok, 'k20_ok': k20_ok, 'k_rr_ok': k_rr_ok,
        'k15_available': ema20_available, 'k16_available': k16_available,
        'k17_available': k17_available, 'k18_available': macd_available,
        'k19_available': adx_available, 'k20_available': k20_available,
        'k_rr_available': k_rr_available,
        'trend_available_count': n_trend_available, 'mom_available_count': n_mom_available,
        'trend_criteria_count': cfg.trend_criteria_count,
        'mom_criteria_count': cfg.momentum_criteria_count,
        'total_criteria_count': cfg.total_criteria_count,
        'rr_status': rr['rr_status'], 'rr_status_reason': rr['rr_status_reason'],
        'vol_last': vol_last, 'vol_avg20': vol_avg20,
        'data_completeness': data_completeness,
        'data_status': 'OK',
        'data_status_reason': '',
        'is_evaluated': True,
        'missing_fields': json.dumps(missing_fields, ensure_ascii=False),
        'algo_version': ALGO_VERSION,
        'config_snapshot': cfg.as_json(),
    }
