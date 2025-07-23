from abc import ABC
from dataclasses import dataclass


@dataclass
class StreamerConfig:
    temporal_dims: int
    spatial_dims: list[int]


@dataclass
class StreamersConfig(list[StreamerConfig]): ...


class AcceleratorConfig: ...


class AcceleratorWrapper(ABC):
    @property
    def accelerator(self) -> AcceleratorConfig:
        raise NotImplementedError("This method should be implemented by subclasses.")


@dataclass
class DataMoverConfig(AcceleratorConfig): ...


@dataclass
class DataMoverWrapper(AcceleratorWrapper):
    data_mover: None

    @property
    def accelerator(self) -> AcceleratorConfig:
        return DataMoverConfig()


@dataclass
class SnaxAluConfig(AcceleratorConfig): ...


@dataclass
class SnaxAluWrapper(AcceleratorWrapper):
    snax_alu: None

    @property
    def accelerator(self) -> AcceleratorConfig:
        return SnaxAluConfig()


@dataclass
class GemmxConfig(AcceleratorConfig): ...


@dataclass
class GemmxWrapper(AcceleratorWrapper):
    gemmx: None

    @property
    def accelerator(self) -> AcceleratorConfig:
        return GemmxConfig()


@dataclass
class CoreConfig:
    accelerators: list[DataMoverWrapper | SnaxAluWrapper | GemmxWrapper]


@dataclass
class SnaxMemoryConfig:
    name: str
    start: int
    size: int


@dataclass
class ClusterConfig:
    memory: SnaxMemoryConfig
    cores: list[CoreConfig]


@dataclass
class SystemConfig:
    memory: SnaxMemoryConfig
    clusters: list[ClusterConfig]
