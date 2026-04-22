import polars as pl
from collections import defaultdict
from datetime import datetime, time
from dataclasses import dataclass

def get_session_expr(start: datetime, end: datetime, shift: int=0):
    return (
        ((pl.col("Time") + pl.duration(minutes=shift)).dt.time() >= start) & 
        ((pl.col("Time") + pl.duration(minutes=shift)).dt.time() <= end)
    )

def get_day_session_expr(shift: int=0):
    return (
        (get_session_expr(time(9,0), time(10,14), shift)) |
        (get_session_expr(time(10,30), time(11,29), shift)) |
        (get_session_expr(time(13,30), time(14,59), shift))
    )

def get_night_session_expr(shift: int=0):
    return (
        (get_session_expr(time(21,0), time(23,59), shift)) |
        (get_session_expr(time(0,0), time(2,29), shift))
    )

def get_market_open_expr(shift: int=0):
    return (
        (get_session_expr(time(9,0), time(9,2), shift)) |
        (get_session_expr(time(10,30), time(10,32), shift)) |
        (get_session_expr(time(13,30), time(13,32), shift)) |
        (get_session_expr(time(21,0), time(21,2), shift))
    )

@dataclass
class DataConfig:
    def __init__(self):
        _ALL_SYMBOLS = ['PP', 'Y', 'OI', 'EB', 'NR', 'I', 'RM', 'CF', 'SN', 'TA', 'TF', 'M', 'J', 'IF', 'AP', 'UR', 'ZC', 'JD', 'CJ', 'P', 'FU', 'PG', 'CS', 'MA', 'TS', 'SS', 'IH', 'IC', 'AL', 'SC', 'NI', 'RB', 'HC', 'RU', 'CU', 'V', 'PB', 'SP', 'ZN', 'FG', 'JM', 'CY', 'BU', 'SF', 'AU', 'RR', 'C', 'FB', 'EG', 'T', 'L', 'B', 'AG', 'A', 'SM', 'SA', 'SR']
        _EXCLUDED = ["T", "TF", "TS", "IF", "IC", "IH"]
        SYMBOLS = [s for s in _ALL_SYMBOLS if s not in _EXCLUDED]
        self.SYMBOLS = SYMBOLS
            
        self.EXPR_DAY_SESSIONS = get_day_session_expr()
        self.EXPR_NIGHT_SESSIONS = get_night_session_expr()
        self.EXPR_MARKET_OPEN = get_market_open_expr()