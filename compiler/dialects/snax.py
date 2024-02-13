from __future__ import annotations

from typing import cast

from xdsl.dialects.memref import MemRefType, UnrankedMemrefType
from xdsl.ir import Attribute, Dialect, Operation, SSAValue
from xdsl.irdl import IRDLOperation, irdl_op_definition, operand_def, result_def
from xdsl.utils.exceptions import VerifyException


@irdl_op_definition
class ClusterSyncOp(IRDLOperation):
    """Cluster sync operation for a snax cluster. This
    translates directly to the C function snrt_cluster_hw_barrier()"""

    name = "snax.cluster_sync_op"


@irdl_op_definition
class LayoutCast(IRDLOperation):
    """LayoutCast operation for memrefs in a snax cluster. This
    operation is used to change the layout of the memref data"""

    name = "snax.layout_cast"

    source = operand_def(MemRefType[Attribute] | UnrankedMemrefType[Attribute])
    dest = result_def(MemRefType[Attribute] | UnrankedMemrefType[Attribute])

    def __init__(
        self,
        source: SSAValue | Operation,
        dest: MemRefType[Attribute] | UnrankedMemrefType[Attribute],
    ):
        super().__init__(operands=[source], result_types=[dest])

    @staticmethod
    def from_type_and_target_layout(
        source: SSAValue | Operation,
        layout: Attribute,
    ) -> LayoutCast:
        assert isinstance(source.type, MemRefType)
        dest = MemRefType(
            source.type.get_element_type(),
            shape=source.type.get_shape(),
            layout=layout,
            memory_space=source.type.memory_space,
        )
        return LayoutCast(source, dest)

    def verify_(self) -> None:
        source = cast(MemRefType[Attribute], self.source.type)
        dest = cast(MemRefType[Attribute], self.dest.type)
        if source.get_shape() != dest.get_shape():
            raise VerifyException(
                "Expected source and destination to have the same shape."
            )
        if source.get_element_type() != dest.get_element_type():
            raise VerifyException(
                "Expected source and destination to have the same element type."
            )
        if source.memory_space != dest.memory_space:
            raise VerifyException(
                "Expected source and destination to have the same memory space."
            )


Snax = Dialect("snax", [ClusterSyncOp, LayoutCast], [])
