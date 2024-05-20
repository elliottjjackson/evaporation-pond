from abc import abstractmethod, ABC
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Optional, Tuple, Iterator
import calendar
from datetime import date, timedelta, datetime


# https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://library.dpird.wa.gov.au/cgi/viewcontent.cgi%3Farticle%3D1058%26context%3Drmtr&ved=2ahUKEwjolN2SooWGAxXXR2wGHdTzBy8QFnoECBIQAQ&usg=AOvVaw1sRlfWhdNltfpyeWGYx1Jf
@dataclass
class Evaporation:
    # Mt Magnet evaporation rate data
    def __init__(self):
        self._evaporation_rate_table: dict[str:float] = {
            "jan": 519,
            "feb": 443,
            "mar": 391,
            "apr": 264,
            "may": 179,
            "jun": 116,
            "jul": 119,
            "aug": 157,
            "sep": 204,
            "oct": 323,
            "nov": 383,
            "dec": 477,
        }

    @property
    def rate(self) -> float:
        month = get_current_month()[:3].lower()
        rate = self._evaporation_rate_table[month]
        return rate


# http://www.bom.gov.au/jsp/ncc/cdio/cvg/av?p_stn_num=008051&p_prim_element_index=18&p_display_type=statGraph&period_of_avg=ALL&normals_years=allYearOfData&staticPage=
@dataclass
class Rainfall:
    # Geraldton airport 2014
    def __init__(self):
        self._rainfall_depth_table: dict[str:float] = {
            "jan": 5.7,
            "feb": 11.1,
            "mar": 15.9,
            "apr": 23.8,
            "may": 69.5,
            "jun": 97.2,
            "jul": 91.3,
            "aug": 64.7,
            "sep": 32.0,
            "oct": 19.6,
            "nov": 9.1,
            "dec": 5.3,
        }

    @property
    def depth(self) -> float:
        month = get_current_month()[:3].lower()
        rate = self._rainfall_depth_table[month]
        return rate


# Singleton class used prior to the implementation of IX to get_current_time method.
class TimeObject:
    _instance = None

    # Singleton: ensures only one instance of this class is created.
    def __new__(cls):
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

    # Generator that timesteps the date.
    def timestepper(self) -> Iterator[datetime]:
        while True:
            days_in_month = calendar.monthrange(
                self.current_time.year, self.current_time.month
            )[1]
            self.current_time += timedelta(days=days_in_month)
            yield self.current_time

    # Method to progress the TimeObject timestep by one unit using the generator.
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
    time_obj = TimeObject()

    def __repr__(self) -> str:
        number = len(self.record)
        entries = "entry" if number == 1 else "entries"
        return f"Timeseries({number} {entries})"

    def __len__(self) -> int:
        return len(self.record)

    def add_record(self, **kwargs) -> None:
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


@dataclass
class EvaporationPond(Mapping):
    level: Optional[float] = None
    volume: Optional[float] = None
    capacity: float = float("inf")
    max_level: float = (float("inf"),)
    time_series: Timeseries = field(default_factory=Timeseries)

    def __init__(
        self,
        level: Optional[float] = None,
        volume: Optional[float] = None,
        capacity: float = float("inf"),
    ):
        # __init__ required to avoid level and volume setter recursion and to allow for
        # use of private and public variables.
        self.capacity = capacity
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

    def _update_record(self):
        # record_dict = {
        #     key: value for key, value in self.__dict__.items() if key != "time_series"
        # }
        record_dict = {}
        for key, value in self.__dict__.items():
            if key != "time_series":
                if key.startswith("_"):
                    record_dict[key[1:]] = value
                else:
                    record_dict[key] = value
        # Implement a way to replace the private fields with the public ones.
        self.time_series.add_record(**record_dict)

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
    def calculate_level(self, volume) -> Optional[float]:
        area = 200
        level = volume / area
        return level

    def calculate_volume(self, level) -> Optional[float]:
        area = 1000
        volume = level * area
        return volume


if __name__ == "__main__":

    time = TimeObject()

    south_pond = PlantPond(level=3, capacity=9000)
    north_pond = PlantPond(volume=100, capacity=9000)

    time.progress_time()

    north_pond.volume = 400
    north_pond.volume += 400
    south_pond.level = 4

    print(north_pond.time_series.record)
    print(south_pond.time_series.record)

    time.progress_time()

    north_pond.volume += 400
    south_pond.level += 2

    print(north_pond.time_series.record)
    print(south_pond.time_series.record)

    evaporation = Evaporation()
    rainfall = Rainfall()

    print(evaporation.rate)
    print(rainfall.depth)


#############################
# TO BE IMPLEMENTED
# Pond allocation strategies
#############################
# class Container:
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.volume = 0

#     def fill(self, volume):
#         available_capacity = self.capacity - self.volume
#         volume_to_add = min(available_capacity, volume)
#         self.volume += volume_to_add
#         return volume - volume_to_add

# class AllocationStrategy(ABC):
#     @abstractmethod
#     def allocate(self, volume: float, containers: List[Container]) -> None:
#         pass

# class EvenDistributionStrategy(AllocationStrategy):
#     def allocate(self, volume: float, containers: List[Container]) -> None:
#         while volume > 0 and any(c.capacity > c.volume for c in containers):
#             for container in containers:
#                 if container.capacity > container.volume:
#                     volume = container.fill(volume)
#                     if volume == 0:
#                         break

# class FillFirstStrategy(AllocationStrategy):
#     def allocate(self, volume: float, containers: List[Container]) -> None:
#         for container in containers:
#             volume = container.fill(volume)
#             if volume == 0:
#                 break

# class ContainerAllocator:
#     def __init__(self, strategy: AllocationStrategy):
#         self.strategy = strategy
#         self.containers = []

#     def add_container(self, capacity):
#         self.containers.append(Container(capacity))

#     def distribute(self, volume):
#         self.strategy.allocate(volume, self.containers)

#     def set_strategy(self, strategy: AllocationStrategy):
#         self.strategy = strategy

# # Example usage:
# containers = ContainerAllocator(EvenDistributionStrategy())
# containers.add_container(100)
# containers.add_container(150)
# containers.add_container(200)

# containers.distribute(120)  # Distribute 120 units using the chosen strategy

# for i, container in enumerate(containers.containers):
#     print(f"Container {i+1}: {container.volume}/{container.capacity}")

# # Change strategy
# containers.set_strategy(FillFirstStrategy())
# containers.distribute(200)  # Distribute another 200 units using a different strategy
# for i, container in enumerate(containers.containers):
#     print(f"Container {i+1}: {container.volume}/{container.capacity}")
