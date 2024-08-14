import math

from xdsl.context import MLContext
from xdsl.dialects import builtin, linalg, transform
from xdsl.dialects.builtin import DenseArrayBase, IntegerType, UnitAttr
from xdsl.ir import Block, Operation, Region, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from zigzag.stages.AcceleratorParserStage import AcceleratorParserStage
from zigzag.stages.CostModelStage import CostModelStage
from zigzag.stages.MainStage import MainStage
from zigzag.stages.reduce_stages import MinimalLatencyStage
from zigzag.stages.save_stages import CompleteSaveStage
from zigzag.stages.SpatialMappingGeneratorStage import SpatialMappingGeneratorStage
from zigzag.stages.temporal_mapping_generator_stage import TemporalMappingGeneratorStage
from zigzag.stages.WorkloadParserStage import WorkloadParserStage
from zigzag.stages.WorkloadStage import WorkloadStage
from zigzag.visualization.results.print_mapping import get_temporal_spatial_loops

from compiler.util.zigzag_dicts import get_yaml_files, remove_yaml_files


def getMKN(op: linalg.MatmulOp | linalg.QuantizedMatmulOp):
    M = op.inputs[0].type.shape.data[0].data
    K = op.inputs[0].type.shape.data[1].data
    N = op.inputs[1].type.shape.data[1].data
    return (M, K, N)


def get_WIO_element_type(op: linalg.MatmulOp | linalg.QuantizedMatmulOp):
    W = op.inputs[0].type.element_type.bitwidth
    I = op.inputs[1].type.element_type.bitwidth
    O = op.outputs[0].type.element_type.bitwidth
    return (W, I, O)


def get_module(op: Operation):
    while not isinstance(op, builtin.ModuleOp):
        op = op.parent_op()
    return op


def is_prime(n):
    if n <= 1:
        return False
    if n == 2 or n == 3:
        return True  # 2 and 3 are prime numbers
    if n % 2 == 0 or n % 3 == 0:
        return False  # Exclude multiples of 2 and 3

    # Check divisibility from 5 to the square root of n
    for i in range(5, int(math.sqrt(n)) + 1, 6):
        if n % i == 0 or n % (i + 2) == 0:
            return False

    return True


def find_yield_op(sequence: transform.SequenceOp):
    for block in sequence.body.blocks:
        for op in block.ops:
            if isinstance(op, transform.YieldOp):
                return op
    return None


def get_interchange(order: list[str]):
    translation = {"M": 0, "K": 1, "N": 2}
    return [translation[order_loop] for order_loop in order] + [
        0 for _ in range(3 - len(order))
    ]


def get_tiling_ops(op: linalg.MatmulOp | linalg.QuantizedMatmulOp, target: SSAValue):
    tiling_ops = get_zigzag_order(
        MKN=getMKN(op), WIO_element_type=get_WIO_element_type(op)
    )
    all_tiling_ops = []
    for index, (order, tile_sizes) in enumerate(tiling_ops):
        all_tiling_ops.append(
            transform.TileOp(
                target=target if index == 0 else all_tiling_ops[-1].results[0],
                dynamic_sizes=[],
                scalable_sizes=DenseArrayBase.create_dense_int_or_index(
                    IntegerType(1), [0, 0, 0]
                ),
                static_sizes=DenseArrayBase.create_dense_int_or_index(
                    IntegerType(64), tile_sizes
                ),
                interchange=DenseArrayBase.create_dense_int_or_index(
                    IntegerType(64),
                    get_interchange(order),
                ),
            )
        )
    return all_tiling_ops


def reduce_order_sizes(order_sizes: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """
    Reduces the order sizes by combining consecutive tuples with the same string value.
    """
    reduced_order_sizes = []
    reduced_order_sizes.append(order_sizes[0])
    for i in range(1, len(order_sizes)):
        if reduced_order_sizes[-1][0] == order_sizes[i][0]:
            reduced_order_sizes[-1] = (
                reduced_order_sizes[-1][0],
                reduced_order_sizes[-1][1] * order_sizes[i][1],
            )
        else:
            reduced_order_sizes.append(order_sizes[i])
    return reduced_order_sizes


def find_next_tiling(
    reduced_order_sizes: list[tuple[str, int]]
) -> list[list[tuple[str, int]]]:
    """
    Finds the next set of maximum 3 tiling pairs where both B, K and C can only be present once.
    """
    tiling_ops = []
    tiling_ops.append([])
    for i in range(0, len(reduced_order_sizes)):
        if any(
            reduced_order_sizes[i][0] == tiling_op[0] for tiling_op in tiling_ops[-1]
        ):
            tiling_ops.append([reduced_order_sizes[i]])
        else:
            tiling_ops[-1].append(reduced_order_sizes[i])
    return tiling_ops


def get_loop_sizes(
    loops: tuple[tuple[str, tuple[int, int], tuple[str, str, str]], ...],
    MKN: tuple[int, int, int],
) -> list[tuple[list[str], list[int]]]:
    MKN = [MKN[0], MKN[1], MKN[2]]
    order_sizes = [(loop[0].name, loop[1][1]) for loop in loops]
    reduced_order_sizes = reduce_order_sizes(order_sizes)
    final_tiling = []
    for tiling_op in find_next_tiling(reduced_order_sizes[:-3]):
        order = []
        sizes = [0, 0, 0]
        for loop in tiling_op:
            if loop[0] == "B":
                order.append("M")
                sizes[0] = MKN[0] // loop[1]
                MKN[0] = sizes[0]
            elif loop[0] == "K":
                order.append("K")
                sizes[1] = MKN[1] // loop[1]
                MKN[1] = sizes[1]
            elif loop[0] == "C":
                order.append("N")
                sizes[2] = MKN[2] // loop[1]
                MKN[2] = sizes[2]
        final_tiling.append((order, sizes))
    return final_tiling


def get_zigzag_order(MKN: tuple[int, int, int], WIO_element_type: str):
    file_paths = get_yaml_files(MKN, WIO_element_type)
    mainstage = MainStage(
        [  # Initializes the MainStage as entry point
            WorkloadParserStage,  # Parses the manual definition into the workload
            AcceleratorParserStage,  # Parses the accelerator
            CompleteSaveStage,  # Saves all received CMEs information to a json
            WorkloadStage,  # Iterates through the different layers in the workload
            SpatialMappingGeneratorStage,  # Generates multiple spatial mappings (SM)
            MinimalLatencyStage,  # Reduces all CMEs, returning minimal latency one
            TemporalMappingGeneratorStage,  # Converts defined temporal_ordering to temporal mapping
            CostModelStage,  # Evaluates generated SM and TM through cost model
        ],
        accelerator=file_paths["hardware"],  # required by AcceleratorParserStage
        workload=file_paths["workload"],  # required by ONNXModelParserStage
        mapping=file_paths["mapping"],  # required by ONNXModelParserStage
        dump_folder="compiler/zigzag/",  # where outputs will be saved to
        loma_lpf_limit=100,  # required by LomaStage
        loma_show_progress_bar=True,  # shows a progress bar while iterating over temporal mappings
    )

    # Launch the MainStage
    answers = mainstage.run()
    remove_yaml_files(file_paths)
    loops = get_temporal_spatial_loops(answers[0][0])
    return get_loop_sizes(loops[0], MKN)


class CreateTransformSequence(RewritePattern):

    already_created = False
    sequence_op = None
    current_tag = 0

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: linalg.MatmulOp | linalg.QuantizedMatmulOp, rewriter: PatternRewriter
    ):
        # Create a sequence on the first match
        if not self.already_created:
            self.already_created = True
            # Don't overwrite an existing sequence op
            if any(
                isinstance(op, transform.SequenceOp)
                for op in get_module(op).body.walk()
            ):
                return
            self.sequence_op = transform.SequenceOp(
                failure_propagation_mode=2,
                root=[],
                extra_bindings=[],
                body=Region(
                    [
                        Block(
                            ops=[transform.YieldOp()],
                            arg_types=[transform.AnyOpType()],
                        )
                    ]
                ),
            )
            rewriter.insert_op(
                self.sequence_op,
                insertion_point=InsertPoint.at_end(get_module(op).body.last_block),
            )
        local_MKN = getMKN(op)
        # All dimensions must be known, and be a multiple of 8
        if all(
            local_M_K_N is not None
            and local_M_K_N % 8 == 0
            and not is_prime(local_M_K_N // 8)
            for local_M_K_N in local_MKN
        ):
            if not op.attributes:
                op.attributes = {}
            op.attributes[f"qmatmul_{self.current_tag}"] = UnitAttr()
            structured_match = transform.MatchOp(
                target=self.sequence_op.body.first_block.args[0],
                op_attrs={f"qmatmul_{self.current_tag}": UnitAttr()},
            )
            rewriter.insert_op(
                structured_match,
                insertion_point=InsertPoint.before(find_yield_op(self.sequence_op)),
            )
            tile_ops = get_tiling_ops(op, structured_match.results[0])
            for tile_op in tile_ops:
                rewriter.insert_op(
                    tile_op,
                    insertion_point=InsertPoint.before(find_yield_op(self.sequence_op)),
                )
            self.current_tag += 1


class AddTilingSequence(ModulePass):
    name = "add-tiling-sequence"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            CreateTransformSequence(),
            apply_recursively=False,
        ).rewrite_module(op)
