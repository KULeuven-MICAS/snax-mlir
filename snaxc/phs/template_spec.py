import itertools
from collections.abc import Iterable

from xdsl.ir.affine import AffineMap

from snaxc.accelerators.streamers.streamers import Streamer, StreamerConfiguration, StreamerFlag, StreamerType
from snaxc.ir.dart.access_pattern import Template, TemplatePattern


class TemplateSpec:
    input_maps: tuple[AffineMap, ...]
    output_maps: tuple[AffineMap, ...]
    template_bounds: tuple[int, ...]

    def __init__(
        self, input_maps: tuple[AffineMap, ...], output_maps: tuple[AffineMap, ...], template_bounds: tuple[int, ...]
    ):
        self.input_maps = input_maps
        self.output_maps = output_maps
        self.template_bounds = template_bounds
        assert len(self.input_maps) > 0, "Expect input_maps to be non-empty"
        assert len(self.output_maps) > 0, "Expect output_maps to be non-empty"
        assert self._no_symbols(), "No symbols expected in any affine map of template_spec"
        assert self._same_dims(), "Expect all AffineMaps to have equal number of dims"
        assert len(template_bounds) == self.input_maps[0].num_dims, "Expect number of iterators and bounds to be equal"

    def __str__(self) -> str:
        _str: str = ""
        _str += "maps:\n"
        for i, i_map in enumerate(self.input_maps):
            _str += f"i{i} : {i_map}\n"
        for o, o_map in enumerate(self.output_maps):
            _str += f"o{o} : {o_map}\n"

        _str += "bounds:\n"
        for b, bound in enumerate(self.template_bounds):
            _str += f"d{b} : {bound}\n"
        return _str

    def _no_symbols(self) -> bool:
        comparison = [map.num_symbols == 0 for map in self.input_maps + self.output_maps]
        return all(comparison)

    def _same_dims(self) -> bool:
        first_num_dims = self.input_maps[0].num_dims
        comparison = [map.num_dims == first_num_dims for map in (self.input_maps + self.output_maps)[:1]]
        return all(comparison)

    def _get_sizes(self, maps: tuple[AffineMap, ...]) -> list[tuple[int, ...]]:
        return [map.eval(self.template_bounds, ()) for map in maps]

    def get_input_sizes(self) -> list[tuple[int, ...]]:
        return self._get_sizes(self.input_maps)

    def get_output_sizes(self) -> list[tuple[int, ...]]:
        return self._get_sizes(self.output_maps)

    def get_iterations(self) -> Iterable[tuple[int, ...]]:
        return itertools.product(*[range(bound) for bound in self.template_bounds])

    def get_streamer_config(self) -> StreamerConfiguration:
        reader_streamers: list[Streamer] = []
        for input_size in self.get_input_sizes():
            reader_streamers.append(
                Streamer(
                    StreamerType.Reader, temporal_dims=[StreamerFlag.Normal] * len(input_size), spatial_dims=input_size
                )
            )
        writer_streamers: list[Streamer] = []
        for output_size in self.get_output_sizes():
            writer_streamers.append(
                Streamer(
                    StreamerType.Reader,
                    temporal_dims=[StreamerFlag.Normal] * len(output_size),
                    spatial_dims=output_size,
                )
            )
        return StreamerConfiguration([*reader_streamers, *writer_streamers])

    def get_dart_template(self) -> Template:
        template = [*self.input_maps, *self.output_maps]
        template_bounds = self.template_bounds
        return Template(TemplatePattern(template_bounds, tp) for tp in template)
