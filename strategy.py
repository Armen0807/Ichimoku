import datetime as dt
import numpy as np
from pydantic import field_validator
from loguru import logger
from typing import Dict, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field, Extra

from grt_lib_price_loader import Instrument
from inar_strat_types import AbstractStratConfig, StratHistory, to_strat_history
from grt_lib_orchestrator import AbstractBacktestStrategy
from grt_lib_order_book import Trade


class IchimokuTrade(Trade):
	def __init__(self, entry_tenkan: float, entry_kijun: float):
		super().__init__()
		self.entry_tenkan = entry_tenkan
		self.entry_kijun = entry_kijun


class ThresholdDict(TypedDict):
	base_level: int
	calculation_range: int
	base_date: dt.date

class VolTargetConfig(AbstractStratConfig):
	class Config:
		arbitrary_types_allowed = True

	conversion_line_periods: int
	base_line_periods: int
	lagging_span_periods: int
	displacement: int

	threshold: ThresholdDict

	take_profit: float
	stop_loss: float
	expected_profit: float

	asset: str

	# useful when parsing Instrument in risky_asset for example
	# @field_validator('risky_asset', mode="before")
	# def get_instrument(cls, v):
	# 	if isinstance(v, list):
	# 		return [ins if isinstance(ins, Instrument) else Instrument.from_ric(ins) for ins in v]
	# 	return [v if isinstance(v, Instrument) else Instrument.from_ric(v)]


class VolTargetDetails(StratHistory):
	strat_level: Dict[dt.date, float] = {}


class IchimokuCalc(StratHistory):
	conversion_line: Dict[str, Dict[dt.date, float]] = {}
	base_line: Dict[str, Dict[dt.date, float]] = {}
	leading_span_1: Dict[str, Dict[dt.date, float]] = {}
	leading_span_2: Dict[str, Dict[dt.date, float]] = {}
	lagging_span: Dict[str, Dict[dt.date, float]] = {}

class StrategySignals(StratHistory):
	short_entry_condition: Dict[str, Dict[dt.date, bool]] = {}
	long_entry_condition: Dict[str, Dict[dt.date, bool]] = {}


class VolTargetHistory(StratHistory):
	details: VolTargetDetails = VolTargetDetails()
	ichimoku: IchimokuCalc = IchimokuCalc()
	strategy_signals: StrategySignals = StrategySignals()


class VolTargetBacktestStrategy(AbstractBacktestStrategy):

	config: VolTargetConfig
	history: VolTargetHistory

	open_position = False

	@to_strat_history("ichimoku")
	def calc_conversion_line(self, instrument_ric: str, t: dt.date) -> float:
		"""
		Calculates the Tenkan-sen - Section 4
		:return: float
		"""
		instrument = Instrument.from_ric(instrument_ric)

		tm_conversion_line_period = self.calendar.busday_add(t, -self.config.conversion_line_periods)
		date_range = self.calendar.busday_range(tm_conversion_line_period, t)

		highest_high = float('-inf')
		lowest_low = float('inf')

		for date in date_range:
			highest_high = max(highest_high, self.price_loader.get_price(date, instrument, "High"))
			lowest_low = min(lowest_low, self.price_loader.get_price(date, instrument, "Low"))

		return (highest_high + lowest_low) / 2

	@to_strat_history("ichimoku")
	def calc_base_line(self, instrument_ric: str, t: dt.date) -> float:
		"""
		Calculates the Kijun-sen - Section 4
		:return: float
		"""
		instrument = Instrument.from_ric(instrument_ric)

		tm_base_line_periods = self.calendar.busday_add(t, -self.config.base_line_periods)
		date_range = self.calendar.busday_range(tm_base_line_periods, t)

		highest_high = float('-inf')
		lowest_low = float('inf')

		for date in date_range:
			highest_high = max(highest_high, self.price_loader.get_price(date, instrument, "High"))
			lowest_low = min(lowest_low, self.price_loader.get_price(date, instrument, "Low"))

		return (highest_high + lowest_low) / 2

	@to_strat_history("ichimoku")
	def calc_leading_span_1(self, instrument_ric: str, t: dt.date) -> float:
		"""
		Calculates the Senkou Span A - Section 4
		:return: float
		"""
		conversion_line = self.calc_conversion_line(instrument_ric, t)
		base_line = self.calc_base_line(instrument_ric, t)

		return (conversion_line + base_line) / 2

	@to_strat_history("ichimoku")
	def calc_leading_span_2(self, instrument_ric: str, t: dt.date) -> float:
		"""
		Calculates the Senkou Span B - Section 4
		:return: float
		"""
		instrument = Instrument.from_ric(instrument_ric)

		tm_base_line_periods = self.calendar.busday_add(t, -self.config.base_line_periods)
		date_range = self.calendar.busday_range(tm_base_line_periods, t)

		highest_high = float('-inf')
		lowest_low = float('inf')

		for date in date_range:
			highest_high = max(highest_high, self.price_loader.get_price(date, instrument, "High"))
			lowest_low = min(lowest_low, self.price_loader.get_price(date, instrument, "Low"))

		return (highest_high + lowest_low) / 2

	@to_strat_history("ichimoku")
	def calc_lagging_span(self, instrument_ric: str, t: dt.date) -> float:
		"""
		Calculates the Chikou Span - Section 4
		:return: float
		"""
		tm26 = self.calendar.busday_add(t, -26)
		close_price_tm26 = self.price_loader.get_price(tm26, Instrument.from_ric(instrument_ric))
		return close_price_tm26

	def calc_maximum_deviation_allowed(self, t: dt.date) -> float:
		maximum_deviation_allowed = 0
		for trade in self.order_book._order_book_dict:
			if trade.pnl >= self.config.expected_profit:
				diff = abs(trade.entry_tenkan - trade.entry_kijun)
				maximum_deviation_allowed = min(maximum_deviation_allowed, diff)
		return maximum_deviation_allowed

	def calc_range_volatility(self, instrument: str, vol_range: str, t: dt.date) -> float:
		vol_range_number = self.config.volatility[vol_range]
		tprange = self.calendar.busday_add(t, vol_range_number + 1)
		date_range = self.calendar.busday_range(t, tprange)
		underlying_instrument = Instrument.from_ric(instrument)

		price_average = np.average([self.price_loader.get_price(date, underlying_instrument) for date in date_range])

		var_sum = 0
		for date in date_range:
			var_sum += (self.price_loader.get_price(date, underlying_instrument) - price_average)**2
		return np.sqrt(252 * var_sum/len(date_range))

	def calc_adjusted_maximum_deviation(self, instrument_ric: str, t: dt.date) -> float:
		maximum_deviation_allowed = self.calc_maximum_deviation_allowed(t)
		short_vol = self.calc_range_volatility(instrument_ric, "short", t)
		annual_vol = self.calc_range_volatility(instrument_ric, "annual", t)

		return maximum_deviation_allowed * short_vol / annual_vol

	def calc_mean_entry_diff(self, instrument_ric: str, t: dt.date) -> float:
		entry_diff_sum = 0
		count = 0
		for identifier, trade in self.order_book._order_book_dict.items():
			if trade.instrument_ric == instrument_ric:
				entry_diff_sum += abs(trade.entry_tenkan - trade.entry_kijun)
				count += 1

		return entry_diff_sum / count

	def calc_final_threshold(self, instrument_ric: str, t: dt.date) -> float:
		if t <= self.config.threshold["base_date"]:
			return self.config.threshold["base_level"]

		adjusted_max_deviation = self.calc_adjusted_maximum_deviation(instrument_ric, t)
		entry_diff_mean = self.calc_mean_entry_diff(instrument_ric, t)

		return max(0.0, adjusted_max_deviation - entry_diff_mean)

	def check_valid_trade(self, trade: IchimokuTrade):
		return trade.pnl >= self.config.expected_profit

	@to_strat_history("strategy_signals")
	def check_long_entry_condition(self, instrument_ric: str, t: dt.date) -> bool:
		"""
		Calculates the long entry condition - Section 5.1
		:param instrument_ric: instrument ric
		:param t: current date
		:return: true/false
		"""
		close_price = self.price_loader.get_price(t, Instrument.from_ric(instrument_ric))
		lead_line_1 = self.calc_leading_span_1(t)
		lead_line_2 = self.calc_leading_span_2(t)
		check_1 = (close_price > lead_line_1 and close_price > lead_line_2)

		conversion_line = self.calc_conversion_line(t)
		base_line = self.calc_base_line(t)
		threshold = self.calc_final_threshold(instrument_ric, t)
		check_2 = (conversion_line > base_line + threshold)

		return check_1 and check_2

	@to_strat_history("strategy_signals")
	def check_short_entry_condition(self, instrument_ric: str, t: dt.date) -> bool:
		"""
		Calculates the long entry condition - Section 5.1
		:param instrument_ric: instrument ric
		:param t: current date
		:return: true/false
		"""
		tm1 = self.calendar.busday_add(t, -1)
		close_price_tm1 = self.price_loader.get_price(tm1, Instrument.from_ric(instrument_ric))
		lead_line_1 = self.calc_leading_span_1(tm1)
		lead_line_2 = self.calc_leading_span_2(tm1)
		check_1 = (close_price_tm1 < lead_line_1 and close_price_tm1 < lead_line_2)

		conversion_line = self.calc_conversion_line(tm1)
		base_line = self.calc_base_line(tm1)
		threshold_tm1 = ...
		check_2 = (conversion_line < base_line - threshold_tm1)

		return check_1 and check_2

	def calc_stop_loss(self, instrument_ric: str, t: dt.date) -> float:
		close_price_t = self.price_loader.get_price(t, Instrument.from_ric(instrument_ric))
		return close_price_t * (1 + self.config.take_profit)

	def calc_take_profit(self, instrument_ric: str, t: dt.date) -> float:
		close_price_t = self.price_loader.get_price(t, Instrument.from_ric(instrument_ric))
		return close_price_t * (1 - self.config.stop_loss)

	def check_long_exit_condition(self, instrument_ric: str, t: dt.date) -> bool:
		conversion_line = self.calc_conversion_line(t)
		base_line = self.calc_base_line(t)
		threshold = self.calc_threshold(t)
		close_price_t = self.price_loader.get_price(t, Instrument.from_ric(instrument_ric))
		return conversion_line <= base_line + threshold

	def check_long_exit_condition(self, t: dt.date) -> bool:
		conversion_line = self.calc_conversion_line(t)
		base_line = self.calc_base_line(t)
		threshold = self.calc_threshold(t)
		return conversion_line >= base_line - threshold

	@to_strat_history("details")
	def calc_strat_level(self, t: dt.date) -> float:
		tm1 = self.calendar.busday_add(t, -1)
		asset_ric = self.config.asset
		asset_instrument = Instrument.from_ric(asset_ric)
		open_price = self.price_loader.get_price(t, asset_instrument, "Open")
		if self.check_long_entry_condition(asset_ric, tm1) and not self.open_position:
			self.open_position = True
		elif self.check_short_entry_condition(asset_ric, tm1) and not self.open_position:
			self.open_position = True
