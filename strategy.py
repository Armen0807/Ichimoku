import datetime as dt
from typing import Dict, Optional
from typing_extensions import TypedDict

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from nautilus_trader.common.instruments import Instrument, Quantity
from nautilus_trader.common.enums import OrderSide, OrderType, PositionSide
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.data import Bar
from nautilus_trader.trading.execution_commands import LimitOrderCommand, MarketOrderCommand
from nautilus_trader.trading.position import Position
from nautilus_trader.common.timeframes import Timeframe
from nautilus_trader.common.periods import Period
from nautilus_trader.common.frequency import Frequency
from nautilus_trader.common.component import Component


class IchimokuTrade:
    def __init__(self, entry_tenkan: float, entry_kijun: float):
        self.entry_tenkan = entry_tenkan
        self.entry_kijun = entry_kijun
        self.pnl = 0.0


class ThresholdDict(BaseModel):
    base_level: int = Field(...)
    calculation_range: int = Field(...)
    base_date: dt.date = Field(...)


class VolTargetConfig(BaseModel):
    conversion_line_periods: int = Field(9, description="Periods for conversion line")
    base_line_periods: int = Field(26, description="Periods for base line")
    lagging_span_periods: int = Field(52, description="Periods for lagging span")
    displacement: int = Field(26, description="Displacement for leading spans")
    threshold: ThresholdDict = Field(...)
    take_profit_multiplier: float = Field(2.0, description="Multiplier for take profit based on entry price")
    stop_loss_multiplier: float = Field(1.0, description="Multiplier for stop loss based on entry price")
    expected_profit: float = Field(0.05, description="Expected profit target")
    asset: str = Field("AAPL.US", description="Asset ticker")
    volatility: Dict[str, int] = Field({"short": 20, "annual": 252}, description="Volatility calculation periods")
    risk_percentage: float = Field(0.01, description="Percentage of capital to risk per trade")
    atr_period: int = Field(14, description="Period for ATR calculation")
    atr_multiplier: float = Field(2.0, description="Multiplier for ATR trailing stop")


class VolTargetDetails:
    strat_level: Dict[dt.date, float] = {}


class IchimokuCalc:
    conversion_line: Dict[dt.date, float] = {}
    base_line: Dict[dt.date, float] = {}
    leading_span_1: Dict[dt.date, float] = {}
    leading_span_2: Dict[dt.date, float] = {}
    lagging_span: Dict[dt.date, float] = {}


class StrategySignals:
    short_entry_condition: Dict[dt.date, bool] = {}
    long_entry_condition: Dict[dt.date, bool] = {}


class VolTargetHistory:
    details: VolTargetDetails = VolTargetDetails()
    ichimoku: IchimokuCalc = IchimokuCalc()
    strategy_signals: StrategySignals = StrategySignals()


class VolTargetBacktestStrategy(Strategy):
    config: VolTargetConfig
    history: VolTargetHistory
    instrument: Optional[Instrument] = None
    open_position: Optional[Position] = None
    entry_trade: Optional[IchimokuTrade] = None
    trailing_stop_loss: Optional[float] = None
    initial_capital: float = 100000.0  # Example initial capital

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = VolTargetConfig(**self.config)
        self.history = VolTargetHistory()
        self.instrument = self.engine.get_instrument(self.config.asset)
        if not self.instrument:
            raise ValueError(f"Instrument {self.config.asset} not found.")
        self.open_position = None
        self.entry_trade = None
        self.trailing_stop_loss = None

    def get_price(self, bar: Bar) -> float:
        if bar and bar.instrument_id == self.instrument.instrument_id:
            return bar.close
        return np.nan

    def get_high(self, bar: Bar) -> float:
        if bar and bar.instrument_id == self.instrument.instrument_id:
            return bar.high
        return np.nan

    def get_low(self, bar: Bar) -> float:
        if bar and bar.instrument_id == self.instrument.instrument_id:
            return bar.low
        return np.nan

    def calc_high_low(self, bars: List[Bar]) -> Tuple[float, float]:
        highest_high = float('-inf')
        lowest_low = float('inf')
        for bar in bars:
            highest_high = max(highest_high, self.get_high(bar))
            lowest_low = min(lowest_low, self.get_low(bar))
        return highest_high, lowest_low

    def get_historical_prices(self, date: dt.date, period: int) -> Optional[pd.DataFrame]:
        history = self.engine.get_historical_data(
            instrument=self.instrument,
            timeframe=Timeframe.DAILY,
            period=Period.range(date - dt.timedelta(days=period), date)
        )
        if not history or len(history) <= period:
            return None
        df = pd.DataFrame([{'ts': bar.ts, 'high': bar.high, 'low': bar.low, 'close': bar.close} for bar in history])
        df['ts'] = pd.to_datetime(df['ts']).dt.date
        df.set_index('ts', inplace=True)
        return df

    def calc_conversion_line(self, date: dt.date) -> Optional[float]:
        df_history = self.get_historical_prices(date, self.config.conversion_line_periods)
        if df_history is None or len(df_history) < self.config.conversion_line_periods:
            return None
        highest_high = df_history['high'].max()
        lowest_low = df_history['low'].min()
        conversion_line_value = (highest_high + lowest_low) / 2
        self.history.ichimoku.conversion_line[date] = conversion_line_value
        return conversion_line_value

    def calc_base_line(self, date: dt.date) -> Optional[float]:
        df_history = self.get_historical_prices(date, self.config.base_line_periods)
        if df_history is None or len(df_history) < self.config.base_line_periods:
            return None
        highest_high = df_history['high'].max()
        lowest_low = df_history['low'].min()
        base_line_value = (highest_high + lowest_low) / 2
        self.history.ichimoku.base_line[date] = base_line_value
        return base_line_value

    def calc_leading_span_1(self, date: dt.date) -> Optional[float]:
        conversion_line = self.calc_conversion_line(date)
        base_line = self.calc_base_line(date)
        if conversion_line is not None and base_line is not None:
            leading_span_1_value = (conversion_line + base_line) / 2
            self.history.ichimoku.leading_span_1[date] = leading_span_1_value
            return leading_span_1_value
        return None

    def calc_leading_span_2(self, date: dt.date) -> Optional[float]:
        future_date = date + dt.timedelta(days=self.config.displacement)
        df_history = self.get_historical_prices(future_date, self.config.base_line_periods)
        if df_history is None or len(df_history) < self.config.base_line_periods:
            return None
        highest_high = df_history['high'].max()
        lowest_low = df_history['low'].min()
        leading_span_2_value = (highest_high + lowest_low) / 2
        self.history.ichimoku.leading_span_2[date] = leading_span_2_value
        return leading_span_2_value

    def calc_lagging_span(self, date: dt.date) -> Optional[float]:
        past_date = date - dt.timedelta(days=self.config.displacement)
        history = self.engine.get_historical_data(
            instrument=self.instrument,
            timeframe=Timeframe.DAILY,
            period=Period.range(past_date, past_date)
        )
        if history and len(history) == 1:
            lagging_span_value = self.get_price(history[0])
            self.history.ichimoku.lagging_span[date] = lagging_span_value
            return lagging_span_value
        return None

    def calc_maximum_deviation_allowed(self, date: dt.date) -> float:
        # Placeholder implementation based on expected profit
        return self.config.expected_profit * 100  # Example: 5% expected profit allows 5 deviation

    def calc_range_volatility(self, date: dt.date, period: int) -> Optional[float]:
        df_history = self.get_historical_prices(date, period)
        if df_history is None or len(df_history) < period:
            return None
        prices = df_history['close'].values
        if len(prices) > 1:
            return np.std(prices) * np.sqrt(252)
        return None

    def calc_adjusted_maximum_deviation(self, date: dt.date) -> float:
        maximum_deviation_allowed = self.calc_maximum_deviation_allowed(date)
        short_vol = self.calc_range_volatility(date, self.config.volatility.get("short", 20))
        annual_vol = self.calc_range_volatility(date, self.config.volatility.get("annual", 252))

        if short_vol is not None and annual_vol is not None and annual_vol != 0:
            return maximum_deviation_allowed * short_vol / annual_vol
        return maximum_deviation_allowed

    def calc_mean_entry_diff(self, date: dt.date) -> float:
        # Placeholder - needs access to historical trades
        return 0.005

    def calc_final_threshold(self, date: dt.date) -> float:
        if date <= self.config.threshold.base_date:
            return float(self.config.threshold.base_level)

        adjusted_max_deviation = self.calc_adjusted_maximum_deviation(date)
        entry_diff_mean = self.calc_mean_entry_diff(date)

        return max(0.0, adjusted_max_deviation - entry_diff_mean)

    def check_long_entry_condition(self, date: dt.date) -> bool:
        conversion_line = self.calc_conversion_line(date)
        base_line = self.calc_base_line(date)
        leading_span_1 = self.calc_leading_span_1(date)
        leading_span_2 = self.calc_leading_span_2(date)
        current_bar = self.engine.get_last_bar(self.instrument, Timeframe.DAILY)
        if not current_bar:
            return False
        close_price = self.get_price(current_bar)

        if leading_span_1 is None or leading_span_2 is None or conversion_line is None or base_line is None:
            return False

        check_1 = (close_price > leading_span_1 and close_price > leading_span_2)
        threshold = self.calc_final_threshold(date)
        check_2 = (conversion_line > base_line + threshold)

        self.history.strategy_signals.long_entry_condition[date] = check_1 and check_2
        return check_1 and check_2

    def check_short_entry_condition(self, date: dt.date) -> bool:
        previous_date = date - dt.timedelta(days=1)
        conversion_line_prev = self.calc_conversion_line(previous_date)
        base_line_prev = self.calc_base_line(previous_date)
        leading_span_1_prev = self.calc_leading_span_1(previous_date)
        leading_span_2_prev = self.calc_leading_span_2(previous_date)
        current_bar = self.engine.get_last_bar(self.instrument, Timeframe.DAILY)
        if not current_bar:
            return False
        close_price_prev = self.get_price(current_bar)

        if leading_span_1_prev is None or leading_span_2_prev is None or conversion_line_prev is None or base_line_prev is None:
            return False

        check_1 = (close_price_prev < leading_span_1_prev and close_price_prev < leading_span_2_prev)
        threshold_prev = self.calc_final_threshold(previous_date)
        check_2 = (conversion_line_prev < base_line_prev - threshold_prev)

        self.history.strategy_signals.short_entry_condition[date] = check_1 and check_2
        return check_1 and check_2

    def calc_stop_loss_level(self, entry_price: float) -> float:
        return entry_price * (1 - self.config.stop_loss_multiplier)

    def calc_take_profit_level(self, entry_price: float) -> float:
        return entry_price * (1 + self.config.take_profit_multiplier)

    def check_long_exit_condition(self, date: dt.date) -> bool:
        conversion_line = self.calc_conversion_line(date)
        base_line = self.calc_base_line(date)
        threshold = self.calc_final_threshold(date)
        return conversion_line is not None and base_line is not None and conversion_line <= base_line + threshold

    def check_short_exit_condition(self, date: dt.date) -> bool:
        conversion_line = self.calc_conversion_line(date)
        base_line = self.calc_base_line(date)
        threshold = self.calc_final_threshold(date)
        return conversion_line is not None and base_line is not None and conversion_line >= base_line - threshold

    def calculate_atr(self, date: dt.date) -> Optional[float]:
        df_history = self.get_historical_prices(date, self.config.atr_period)
        if df_history is None or len(df_history) < self.config.atr_period:
            return None
        df_history['high_low'] = df_history['high'] - df_history['low']
        df_history['high_close_prev'] = abs(df_history['high'] - df_history['close'].shift(1))
        df_history['low_close_prev'] = abs(df_history['low'] - df_history['close'].shift(1))
        atr = df_history[['high_low', 'high_close_prev', 'low_close_prev']].max(axis=1).mean()
        return atr

    def calculate_position_size(self, current_price: float) -> Quantity:
        risk_amount = self.initial_capital * self.config.risk_percentage
        stop_loss_distance = current_price * self.config.stop_loss_multiplier  # Approximate
        if stop_loss_distance > 0:
            shares = risk_amount / stop_loss_distance
            return Quantity.from_int(int(shares))
        return Quantity.from_int(1)

    def on_bar(self, bar: Bar):
        trade_date = bar.trade_date.date()
        conversion_line = self.calc_conversion_line(trade_date)
        base_line = self.calc_base_line(trade_date)
        leading_span_1 = self.calc_leading_span_1(trade_date)
        leading_span_2 = self.calc_leading_span_2(trade_date)
        lagging_span = self.calc_lagging_span(trade_date)

        if conversion_line is None or base_line is None or leading_span_1 is None or leading_span_2 is None or lagging_span is None:
            return

        current_price = self.get_price(bar)
        if np.isnan(current_price):
            return

        if not self.open_position:
            if self.check_long_entry_condition(trade_date):
                quantity = self.calculate_position_size(current_price)
                if quantity.amount > 0:
                    self.submit_order(MarketOrderCommand(self.instrument, OrderSide.BUY, quantity))
                    self.entry_trade = IchimokuTrade(conversion_line, base_line)
                    self.open_position = self.engine.get_position(self.instrument)
                    self.trailing_stop_loss = self.calc_stop_loss_level(current_price) # Initial stop
            elif self.check_short_entry_condition(trade_date):
                quantity = self.calculate_position_size(current_price)
                if quantity.amount > 0:
                    self.submit_order(MarketOrderCommand(self.instrument, OrderSide.SELL, quantity))
                    self.entry_trade = IchimokuTrade(conversion_line, base_line)
                    self.open_position = self.engine.get_position(self.instrument)
                    self.trailing_stop_loss = self.calc_take_profit_level(current_price) # Initial stop

        elif self.open_position:
            if self.open_position.side == PositionSide.LONG:
                if self.check_long_exit_condition(trade_date):
                    self.submit_order(MarketOrderCommand(self.instrument, OrderSide.SELL, self.open_position.net_quantity))
                    self.open_position = None
                    self.entry_trade = None
                    self.trailing_stop_loss = None
                else:
                    # Trailing Stop Loss for Long
                    atr = self.calculate_atr(trade_date)
                    if atr is not None and self.trailing_stop_loss is not None:
                        new_stop = max(self.trailing_stop_loss, current_price - atr * self.config.atr_multiplier)
                        if new_stop > self.trailing_stop_loss:
                            self.trailing_stop_loss = new_stop
                        if current_price <= self.trailing_stop_loss:
                            self.submit_order(MarketOrderCommand(self.instrument, OrderSide.SELL, self.open_position.net_quantity))
                            self.open_position = None
                            self.entry_trade = None
                            self.trailing_stop_loss = None

            elif self.open_position.side == PositionSide.SHORT:
                if self.check_short_exit_condition(trade_date):
                    self.submit_order(MarketOrderCommand(self.instrument, OrderSide.BUY, abs(self.open_position.net_quantity)))
                    self.open_position = None
                    self.entry_trade = None
                    self.trailing_stop_loss = None
                else:
                    # Trailing Stop Loss for Short
                    atr = self.calculate_atr(trade_date)
                    if atr is not None and self.trailing_stop_loss is not None:
                        new_stop = min(self.trailing_stop_loss, current_price + atr * self.config.atr_multiplier)
                        if new_stop < self.trailing_stop_loss:
                            self.trailing_stop_loss = new_stop
                        if current_price >= self.trailing_stop_loss:
                            self.submit_order(MarketOrderCommand(self.instrument, OrderSide.BUY, abs(self.open_position.net_quantity)))
                            self.open_position = None
                            self.entry_trade = None
                            self.trailing_stop_loss = None

            # Basic Stop Loss and Take Profit (can be kept for redundancy or different logic)
            if self.open_position and self.entry_trade:
                stop_loss_level = self.calc_stop_loss_level(self.open_position.average_price)
                take_profit_level = self.calc_take_profit_level(self.open_position.average_price)

                if self.open_position.side == PositionSide.LONG and current_price <= stop_loss_level:
                    self.submit_order(MarketOrderCommand(self.instrument, OrderSide.SELL, self.open_position.net_quantity))
                    self.open_position = None
                    self.entry_trade = None
                    self.trailing_stop_loss = None
                elif self.open_position.side == PositionSide.LONG and current_price >= take_profit_level:
                    self.submit_order(MarketOrderCommand(self.instrument, OrderSide.SELL, self.open_position.net_quantity))
                    self.open_position = None
                    self.entry_trade = None
                    self.trailing_stop_loss = None
                elif self.open_position.side == PositionSide.SHORT and current_price >= stop_loss_level:
                    self.submit_order(MarketOrderCommand(self.instrument, OrderSide.BUY, abs(self.open_position.net_quantity)))
                    self.open_position = None
                    self.entry_trade = None
                    self.trailing_stop_loss = None
                elif self.open_position.side == PositionSide.SHORT and current_price <= take_profit_level:
                    self.submit_order(MarketOrderCommand(self.instrument, OrderSide.BUY, abs(self.open_position.net_quantity)))
                    self.open_position = None
                    self.entry_trade = None
                    self.trailing_stop_loss = None

    def on_trade(self, trade):
        if self.entry_trade and trade.instrument == self.instrument:
            self.entry_trade.pnl += trade.realized_pnl
