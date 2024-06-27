# `accfg` Dialect Documentation

The `accfg` dialect is meant to server as a genric optimisation dialect, which may be used by any lowering pipieline
who has certain semantics and wants to use our optimisations. It is therefore paramount that the dialect documents
the semantics of its operations and the assumptions made by its passes.  

## Operation Semantics

The `accfg` dialect provides three operations that interact with a stateful "client" device. These operations are as
follows:

- `accfg.setup` writes to zero or more configuration registers of the client device
- `accfg.launch` signals the client to start an execution that may be asynchronous
- `accfg.await` blocks until the client device is finished executing

Each operation identifies the client it's interacting with uniquely through a compile-time known property called
`accelerator`. **The client must persist values in its configuration registers across `launch` invocations.**
It is also assumed that all sources that effect the accelerators registers is expressed in the `accfg` dialect.

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


## Pass Semantics:

### `trace-states`

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

### `dedup`

This pass looks at operations input states and removes individual setup parameters if it can be statically determined
that the field is already set to this value. This only works if an input state is provided to the `setup` operation.

### `setup-overlap`

This pass tries to move setup operations upwards *before* `await` operations. This assumes a few things:

- We can freely modify the accelerator state while it's running
- We have a truly asynchronous accelerator (e.g. `launch` will not block until the accelerator has completed)
- The accelerator will take some time during which the host could do some work (e.g. calculating the next setup values)



