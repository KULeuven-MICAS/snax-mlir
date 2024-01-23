from __future__ import annotations

from xdsl.ir import Dialect
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
)


@irdl_op_definition
class ClusterSyncOp(IRDLOperation):
    """Cluster sync operation for a snax cluster. This
    translates directly to the C function snrt_cluster_hw_barrier()"""

    name = "snax.cluster_sync_op"


Snax = Dialect("snax", [ClusterSyncOp], [])
