# `accfg` Dialect Documentation

The `accfg` dialect is meant to serve as a generic optimisation dialect, which may be used by any lowering pipeline
who has certain semantics and wants to use our optimisations. It is therefore paramount that the dialect documents
the semantics of its operations and the assumptions made by its passes.

## Operation Semantics

The `accfg` dialect provides three operations that interact with a stateful accelerator device. These operations are as
follows:

- `accfg.setup` writes to zero or more configuration registers of the client device
- `accfg.launch` signals the client to start an execution that may be asynchronous
- `accfg.await` blocks until the client device is finished executing

Each operation identifies the client it's interacting with uniquely through a compile-time known property called
`accelerator`. **The client must persist values in its configuration registers across `launch` invocations.**
It is also assumed that all sources that effect the accelerators registers are expressed in the `accfg` dialect.

Every `setup` operation produces an `accfg.state` SSA value, representing the current state of the accelerators
registers after the setup operation acted on it. The parameters set by this operation are called *setup parameters*.
A setup may take another `state` value as input. If provided, the output state represents the input state updated
with the setup parameters of this operation. A setup acts destructively on the input state, meaning the input state
**must not** be used after it has been consumed by another `setup` op. If no input state is provided, no assumptions
can be made about the state before the setup operation.

The `launch` operation takes a `state` as input, representing the configuration that it expects to launch. The launch
can be provided with additional configuration parameters, which we call *launch parameters*. The `launch` will produce a
`token` SSA value, which must be consumed by a corresponding `await` operation.


To summarise, our semantics are:
- Only one `accfg.state` variable per client can be alive at any given time.
- Nothing but `accfg` dialect operations modify the accelerator state.
- A `setup` creates a new state (potentially from a previous state) with the provided setup parameters
- *After* a `launch` operation, all state has been copied over to the accelerator, modifying the state will not
  interfere with the current execution of the accelerator.
- *After* an `await`, the client device has completed and is ready to receive another `launch`.


## Semantics of Optimisation Passes:

### `accfg-trace-states`

This pass walks through the provided IR and sees where it can connect state variables generated from previous `setup`
operations with later `setup`s. Since only `accfg` dialect operations may modify the state, we follow the following
assumptions to distinguis wether or not an operation impacts the state:

- If provided, an attribute named `accfg_effects` of either of the two types will overwrite inference.
  These attributes will need to be added by the provider of the IR, as only they know which functions may affect
  the accelerator.
- By default, we assume that function calls modify state (e.g. `func.call` and `llvm.call`).
- All other operation don't modify state, as long as the ops contained within them don't modify state according
  to above criterion.

This pass will connect states through a limited set of `scf` operations, currently `scf.for` and `scf.if`.

This will itself not do any optimisations, but instead serves as the basis for future optimisations.

### `accfg-dedup`

This pass looks at setup operations input states and removes individual setup parameters if it can be statically
determined that the field is already set to this value. This only works if an input state is provided to the
`setup` operation. It is therefore expected that `accfg-trace-states` is run beforehand.

This pass consists of five individual rewrites:

1. Merge two neighboring setup ops
2. Remove empty setups
3. Remove duplicate setups
4. Move setups into conditionals
5. Move loop-invariant setups

The first two rewrites only act locally, either on single ops, or pairs of setup operations within the same block.
This makes reasoning simple and effective.

The deduplication pattern recursively looks at the operations producing the input state to determine if a value
has already been set. If it sees that a field is set to the exact same SSA value again, it removes that setup. This
requires the input IR to be in a canonical form where identical operation sequences have been deduplicated. This will
miss some cases, but is an easy over-approximation, as we know that SSA values will never be re-assigned.

The last two rewrites focus on moving setup operations around so that they can be optimised by the previous three
patterns. (4) moves setup operations upwards into conditional (given that there are other setup operations inside
the conditional). The move is only considered legal if there are no launch operations between the setup and the
conditional.

The final rewrite (5) inspects a `setup` operation inside an `scf.for` loop. If it finds at least one field that is
set ot a loop-independent value, it creates a new setup operation before the loop that sets the loop invariant fields.
Pattern (3) then takes care of erasing the redundant fields inside the for loop.


### `accfg-setup-overlap`

This pass tries to move setup operations upwards *before* `await` operations. This assumes a few things:

- We can freely modify the accelerator state while it's running
- We have a truly asynchronous accelerator (e.g. `launch` will not block until the accelerator has completed)
- The accelerator dominate the runtime, meaning we can do some work while we wait (e.g. calculating the next setup values)

In detail, it performs the following optimisations:

1. Block level await overlap
2. Loop level await overlap

Block level await overlap tries to find a `launch() -> await() -> setup()` triplet inside the same block operating on
the same accelerator. It then moves the `setup()` upwards until it happens between the `launch` and the `await`
resulting in `launch() -> setup() -> await()`. The operations producing the values that are set up are moved together
with the setup, as long as they don't have any effects (e.g. memory writes). If any of the operations that produce the
inputs cannot be determined to be effect free, the move is aborted.

The loop level await overlap peels the setup operation one loop iteration "upwards", resulting in a setup *before* the
loop operating on the initial loop counters, together with a replaced version at the end of the loop body that operates
on the loop counter incremented by the loop step. This effectively moves only the setup operation "upwards" inside the
loop. The resulting IR contains one more `setup` call than the original version (as we now also set up the accelerator
to the `ub+1` state), but since this state is never launched, no adverse effects are expected. This pattern effectively
transforms a loop like this `scf.for (i = 0..10) {setup(i) -> launch() -> await()}` into this loop:
`setup(0) -> scf.for (i = 0..10) {launch() -> await() -> setup(i+1)}`. The loop body can then be optimised by (1) to
result in the final `setup(0) -> scf.for (i = 0..10) {launch() -> setup(i+1) -> await()}`.

## State Reset:

I propose an extension to the dialect in the `accfg.reset` operation. This operation consumes a `state` variable
destructively. The semantics of this operation are that it resets all fields that are modified in the input state
back to the "default" state for this client.

*Rationale:* Some accelerators in shared systems (e.g. Intel AMX, snitch DMA engines) are used by multiple independent
parties (processes, threads or kernels). We want to be able to express programs which are required to reset the client
devices registers back to a known ground-truth value.

*How this affects the other optimisations:**
- `accfg-trace-states`: This pass is allowed to do the following replacement:
  `s0 = setup() ... reset(s0) ... s1 = setup()` to `s0 = setup() ... s1 = setup(input=s0)`
  - for `scf.if`, the `reset` can be moved outside the conditional
  - for `scf.for`, a single `reset` at the end of the loop suffices
- `accfg-dedup`: No impact as it only acts on sequences of connected setups
- `accfg-setup-overlap`: No impact, same reasoning

## Target information:

In order to convert from `accfg` to accelerator setup instructions, target information is required. TODO.

