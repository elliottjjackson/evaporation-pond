from abc import abstractmethod, ABC
from collections.abc import Mapping
import csv
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, Iterator
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import calendar
from datetime import date, timedelta, datetime
from math import isclose
from operator import methodcaller
from itertools import cycle


# https://library.dpird.wa.gov.au/cgi/viewcontent.cgi?article=1058&context=rmtr
@dataclass
class Evaporation:
    def __init__(self):
        self.rate_table: dict[str:float] = {
            "jan": 0,
            "feb": 0,
            "mar": 0,
            "apr": 0,
            "may": 0,
            "jun": 0,
            "jul": 0,
            "aug": 0,
            "sep": 0,
            "oct": 0,
            "nov": 0,
            "dec": 0,
        }

    @property
    def rate(self) -> float:
        month = get_current_month()[:3].lower()
        rate = self.rate_table[month]
        return rate


# http://www.bom.gov.au/jsp/ncc/cdio/cvg/av?p_stn_num=008051&p_prim_element_index=18\&p/_display_type=statGraph&period_of_avg=ALL&normals_years=allYearOfData&staticPage=
@dataclass
class Rainfall:
    def __init__(self):
        self.depth_table: dict[str:float] = {
            "jan": 0,
            "feb": 0,
            "mar": 0,
            "apr": 0,
            "may": 0,
            "jun": 0,
            "jul": 0,
            "aug": 0,
            "sep": 0,
            "oct": 0,
            "nov": 0,
            "dec": 0,
        }

    @property
    def depth(self) -> float:
        month = get_current_month()[:3].lower()
        rate = self.depth_table[month]
        return rate


@dataclass
class WeatherData:
    def __init__(self):
        self.evaporation = Evaporation()
        self.rainfall = Rainfall()

    def set_evaporation_table(self, file_path: str, header: int = 0) -> None:
        evaporation_rate_table: dict[str:float] = self._get_weather_table(
            file_path, header
        )
        if self._valid_rate_table(evaporation_rate_table):
            self.evaporation.rate_table = evaporation_rate_table
        else:
            raise ValueError("Evaporation table not formatted with monthly data.")

    def set_rainfall_table(self, file_path: str, header: int = 0) -> None:
        rainfall_depth_table: dict[str:float] = self._get_weather_table(
            file_path, header
        )
        if self._valid_rate_table(rainfall_depth_table):
            self.rainfall.depth_table = rainfall_depth_table
        else:
            raise ValueError("Rainfall table not formatted with monthly data.")

    def _get_weather_table(self, file_path: str, header: int = 0) -> dict[str:float]:
        evaporation_rate_table = {}
        with open(file_path, "r") as file:
            reader = csv.reader(file)
            if header:
                for _ in range(header):
                    next(reader)  # skip header row
            for row in reader:
                try:
                    month, rate = row
                except:
                    raise ValueError(
                        "CSV must only have two parameters per row: month and value."
                    )
                month = str(month)[:3].lower()  # Month string to 'mmm' format
                evaporation_rate_table[month] = float(rate)
        return evaporation_rate_table

    def _valid_rate_table(self, rate_table: dict[str:float]) -> bool:
        valid_month_set = set(
            [
                "jan",
                "feb",
                "mar",
                "apr",
                "may",
                "jun",
                "jul",
                "aug",
                "sep",
                "oct",
                "nov",
                "dec",
            ]
        )
        rate_table_month_set = set(rate_table.keys())
        if len(rate_table_month_set) != len(valid_month_set):
            raise ValueError("Data format requires 12 rows for each month.")
        elif rate_table_month_set != valid_month_set:
            raise ValueError(
                "Rate table format must be month,rate where month is a string.\n"
                "Use the 'header' parameter to set the number of header rows."
            )
        else:
            return True


# Singleton class used prior to the implementation of IX to get_current_time method.
class TimeObject:
    _instance = None

    # Singleton: ensures only one instance of this class is created.
    def __new__(cls, *kwargs):
        if cls._instance is None:
            cls._instance = super(TimeObject, cls).__new__(cls)
            cls._instance.__initialised = False
        return cls._instance

    # __init__ only runs once if the class has not been initialised.
    def __init__(self, date: datetime = date(2024, 1, 1)):
        if self.__initialised:
            return
        self.current_time = date
        self.timestep = self.timestepper()
        self.__initialised = True

    # Generator that timesteps the date by 1 month.
    def timestepper(self) -> Iterator[datetime]:
        while True:
            days_in_month = calendar.monthrange(
                self.current_time.year, self.current_time.month
            )[1]
            self.current_time += timedelta(days=days_in_month)
            yield self.current_time

    def progress_time(self) -> datetime:
        self.current_time = next(self.timestep)
        return self.current_time

    @property
    def month(self) -> str:
        return self.current_time.strftime("%B")


def get_current_time() -> datetime:
    return TimeObject().current_time


def get_current_month() -> str:
    return TimeObject().month


@dataclass
class Timeseries:
    record: list[dict[str : Optional[float]]] = field(default_factory=list)

    def __repr__(self) -> str:
        number = len(self.record)
        entries = "entry" if number == 1 else "entries"
        return f"Timeseries({number} {entries})"

    def __len__(self) -> int:
        return len(self.record)

    def add_record(self, **kwargs: dict[str:Any]) -> None:
        current_time = get_current_time()
        new_record = {"timestamp": current_time}
        if self.record:
            last_record = self.record[-1]
            if last_record["timestamp"] == new_record["timestamp"]:
                self.record.pop()
        new_record.update(kwargs)
        self.record.append(new_record)

    def get_records_by_field(self, field_name: str) -> list[dict]:
        return [
            record for record in self.record if record.get(field_name) == field_name
        ]

    def get_records(self) -> list[dict]:
        self.record[:]


def load_pond_history_csv(filepath: str, header: int = 0) -> list[datetime, str]:
    with open(filepath, "r") as file:
        reader = csv.reader(file)
        for _ in range(header):
            next(reader)
        return list(reader)


@dataclass
class EvaporationPond(Mapping):
    level: Optional[float] = None
    volume: Optional[float] = None
    capacity: float = float("inf")
    max_level: float = float("inf")
    overflow: float = 0
    time_series: Timeseries = field(default_factory=Timeseries)

    # __init__ required to avoid level and volume setter recursion.
    def __init__(
        self,
        level: Optional[float] = None,
        volume: Optional[float] = None,
        capacity: float = float("inf"),
        overflow: float = 0,
    ):
        self.capacity = capacity
        self._overflow = overflow
        self.max_level = self.calculate_level(self.capacity)
        self.time_series = Timeseries()
        if level is not None and volume is not None:
            raise TypeError("Must specify either height, volume or nothing.")
        elif level:
            self._level = level
            self._volume = self.calculate_volume(level)
        elif volume:
            self._volume = volume
            self._level = self.calculate_level(volume)
        self._update_record()

    def __getitem__(self, key: str) -> Optional[float]:
        return self.__dict__[key]

    def __len__(self) -> int:
        return len(self.__dict__)

    def __iter__(self) -> Iterator[Tuple[str, Optional[float]]]:
        for key, value in self.__dict__.items():
            yield (key, value)

    @abstractmethod
    def calculate_level(self):
        pass

    @abstractmethod
    def calculate_volume(self):
        pass

    def is_overflowing(self, this_volume=None) -> bool:
        if this_volume:
            return this_volume > self.capacity
        else:
            return self._volume > self.capacity

    def is_at_capacity(self, this_volume=None) -> bool:
        if this_volume:
            return isclose(this_volume, self.capacity)
        else:
            return isclose(self._volume, self.capacity)

    def remaining_capacity(self) -> float:
        return self.capacity - self._volume

    def _update_record(self):
        record_dict = {}
        for key, value in self.__dict__.items():
            if key != "time_series":
                if key.startswith("_"):
                    record_dict[key[1:]] = value
                else:
                    record_dict[key] = value
        self.time_series.add_record(**record_dict)

    @property
    def overflow(self):
        return self._overflow

    @overflow.setter
    def overflow(self, value):
        self._overflow = value
        self._update_record()

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, new_level):
        if new_level != self._level:
            self._level = new_level
            self._volume = self.calculate_volume(new_level)
        self._update_record()

    @property
    def volume(self):
        return self._volume

    @volume.setter
    def volume(self, new_volume):
        if new_volume != self._volume:
            self._volume = new_volume
            self._level = self.calculate_level(new_volume)
        self._update_record()


class PlantPond(EvaporationPond):
    def enforce_positive_return(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if result < 0:
                return 0
            return result

        return wrapper

    @enforce_positive_return
    def calculate_level(self, volume) -> Optional[float]:
        # TODO Level calculation when the level is above max
        area = 200
        level = volume / area
        return level

    @enforce_positive_return
    def calculate_volume(self, level) -> Optional[float]:
        # TODO Volume calculation when the volume is above max
        area = 200
        volume = level * area
        return volume


class AllocationStrategy(ABC):
    def weather_effect_level_change(self) -> float:
        return weather_data.rainfall.depth / 1000 - weather_data.evaporation.rate / 1000

    @abstractmethod
    def allocate(
        self, volume: float, ponds: list[EvaporationPond]
    ) -> list[float, float]:
        pass


class EvenDistributionStrategy(AllocationStrategy):
    def allocate(
        self, volume: float, ponds: list[EvaporationPond], weather_data: WeatherData
    ) -> list[float, float]:
        # Ponds are filled from lowest capacity to highest so that overflowing ponds
        # can distribute the overflow to higher capacity ponds.
        sorted_ponds = sorted(ponds, key=methodcaller("remaining_capacity"))
        allocated_fill = volume / len(ponds)
        carry_over = 0
        for i, pond in enumerate(sorted_ponds):
            pond.level += self.weather_effect_level_change()
            ponds_left = len(ponds) - i - 1
            remaining_capacity = pond.remaining_capacity()
            if allocated_fill < remaining_capacity:
                pond.volume += allocated_fill
                carry_over = 0
            elif ponds_left > 0:
                carry_over = allocated_fill - remaining_capacity
                allocated_fill = (
                    (allocated_fill * ponds_left) + carry_over
                ) / ponds_left
                pond.volume = pond.capacity
            else:
                carry_over = allocated_fill - remaining_capacity
                pond.volume = pond.capacity

        allocated_overflow = carry_over / len(sorted_ponds)
        for pond in sorted_ponds:
            pond.overflow = allocated_overflow

        return [volume - carry_over, carry_over]


class FillFirstStrategy(AllocationStrategy):
    def __init__(self):
        self.active_pond = None
        self.inactive_ponds = None

    def sort_inactive_ponds(self, ponds: list[EvaporationPond]):
        self.inactive_ponds = sorted(
            ponds, key=methodcaller("remaining_capacity"), reverse=True
        )

        if self.active_pond in self.inactive_ponds:
            self.inactive_ponds.remove(self.active_pond)

    def set_active_pond_to_highest_capacity(self, ponds: list[EvaporationPond]):
        self.sort_inactive_ponds(ponds)
        if self.active_pond == None:
            highest_capacity_pond = self.inactive_ponds[0]
        elif (
            self.active_pond.remaining_capacity()
            >= self.inactive_ponds[0].remaining_capacity()
        ):
            highest_capacity_pond = self.active_pond
        else:
            highest_capacity_pond = self.inactive_ponds[0]

        self.active_pond = highest_capacity_pond

    def allocate(
        self, volume: float, ponds: list[EvaporationPond], weather_data: WeatherData
    ) -> list[float, float]:
        self.carry_over = 0

        def _allocate_to_pond_logic(pond: EvaporationPond, allocated_fill: float):
            remaining_capacity = pond.remaining_capacity()
            if allocated_fill <= remaining_capacity:
                pond.volume += allocated_fill
                self.carry_over = 0
            else:
                self.carry_over = allocated_fill - remaining_capacity
                pond.volume = pond.capacity

        for pond in ponds:
            pond.level += self.weather_effect_level_change()

        if self.active_pond == None:
            self.set_active_pond_to_highest_capacity(ponds)

        _allocate_to_pond_logic(self.active_pond, volume)

        if self.carry_over > 0:
            self.sort_inactive_ponds(ponds)
            for pond in self.inactive_ponds:
                self.active_pond = pond
                _allocate_to_pond_logic(self.active_pond, volume)
                if self.carry_over == 0:
                    break

        allocated_overflow = self.carry_over / len(ponds)
        for pond in ponds:
            pond.overflow = allocated_overflow

        return [volume - self.carry_over, self.carry_over]


class PondAllocator:
    def __init__(self, strategy: AllocationStrategy):
        self.weather_data = WeatherData()
        self.strategy = strategy
        self.ponds = []

    def add_pond(self, pond: EvaporationPond) -> None:
        self.ponds.append(pond)

    def distribute(self, volume):
        return self.strategy.allocate(volume, self.ponds, self.weather_data)

    def set_strategy(self, strategy: AllocationStrategy):
        self.strategy = strategy

    def set_weather_data(self, weather_data: WeatherData):
        self.weather_data = weather_data


if __name__ == "__main__":

    time = TimeObject(date(2025, 1, 1))

    weather_data = WeatherData()
    weather_data.set_evaporation_table("evaporation.csv", header=1)
    weather_data.set_rainfall_table("rainfall.csv", header=1)

    south_pond = PlantPond(level=0.2, capacity=2000)
    north_pond = PlantPond(volume=1000, capacity=2000)
    east_pond = PlantPond(volume=1700, capacity=2000)

    ponds = PondAllocator(FillFirstStrategy())
    ponds.set_weather_data(weather_data)
    ponds.add_pond(south_pond)
    ponds.add_pond(north_pond)
    ponds.add_pond(east_pond)
    time.progress_time()

    for _ in range(50):
        ponds.distribute(300)
        time.progress_time()

    records = [
        north_pond.time_series.record,
        south_pond.time_series.record,
        east_pond.time_series.record,
    ]

    # Plotting
    plt.figure(figsize=(10, 5))  # Set the figure size

    # Loop through each dataset and plot
    overflow_records = []
    timestamps = []
    for i, pond_record in enumerate(records):
        # Entries in quotes will correspond to properties in EvaporationPond class.
        timestamps = [entry["timestamp"] for entry in pond_record]
        volumes = [entry["volume"] for entry in pond_record]
        names = ["North", "South", "East"]
        plt.plot(timestamps, volumes, marker="o", label=f"{names[i]} Pond")
        overflow_records.append([entry["overflow"] for entry in pond_record])

    overflow_total = [sum(values) for values in zip(*overflow_records)]
    plt.plot(timestamps, overflow_total, marker="o", label=f"Overflow")

    # Formatting the plot
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gcf().autofmt_xdate()  # Rotation

    plt.title("Volume over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("volume")
    plt.legend()  # Add a legend
    plt.grid(True)
    plt.tight_layout()

    # Show the plot
    plt.show()

    print(load_pond_history_csv("history.csv"))
