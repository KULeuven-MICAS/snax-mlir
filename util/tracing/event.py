from abc import ABC, abstractmethod


class Event(ABC):
    @abstractmethod
    def to_chrome_tracing(self, hartid: int) -> dict: ...


class DurationEvent(Event):
    name: str
    cycle_start: int
    cycle_duration: int
    categories: list[str]
    args: dict

    def __init__(self, name, cycle_start, cycle_duration, categories, args=None):
        if args is None:
            args = {}

        self.name = name
        self.cycle_start = cycle_start
        self.cycle_duration = cycle_duration
        self.categories = categories
        self.args = args

    def to_chrome_tracing(self, hartid: int) -> dict:
        return {
            "name": self.name,
            "ts": self.cycle_start,
            "dur": self.cycle_duration,
            "ph": "X",
            "cat": ",".join(self.categories),
            "pid": 0,
            "tid": hartid,
            "args": self.args,
        }


class KernelEvent(DurationEvent):
    def __init__(self, name, cycle_start, cycle_duration, is_dm_core, origin, metrics):
        super().__init__(
            name,
            cycle_start,
            cycle_duration,
            ["vm"] if is_dm_core else ["kernel", origin],
            metrics,
        )


class BarrierEvent(DurationEvent):
    def __init__(self, cycle_start, cycle_duration, pc):
        super().__init__(
            "barrier",
            cycle_start,
            cycle_duration,
            ["barrier"],
            {"program counter": hex(pc)},
        )


class StreamingEvent(DurationEvent):
    def __init__(self, cycle_start, cycle_duration):
        super().__init__("streaming", cycle_start, cycle_duration, ["streaming"])


class DMAEvent(DurationEvent):
    def __init__(
        self,
        cycle_start,
        cycle_duration,
        source: int,
        destination: int,
        inner_loop: int,
        transfer_id: int,
        is_2d: bool,
        source_strides: int,
        dest_strides: int,
        outer_loop: int,
    ):
        byte_count = inner_loop
        extra_kw = {}
        if is_2d:
            byte_count *= outer_loop
            extra_kw = {
                "source strides": source_strides,
                "destination strides": dest_strides,
                "inner loop": inner_loop,
                "outer loop": outer_loop,
            }

        super().__init__(
            "DMA Transfer",
            cycle_start,
            cycle_duration,
            ["DMA"],
            {
                "source": hex(source),
                "destination": hex(destination),
                "copied bytes": byte_count,
                "id": transfer_id,
                "2d": is_2d,
                "Bytes per Cycle": byte_count / cycle_duration,
                **extra_kw,
            },
        )
