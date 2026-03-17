import re

with open("snaxc/ir/dart/cost_models.py", "r") as f:
    text = f.read()

OLD_TEXT = """    # --- Per-streamer state ---

    BUFFER_DEPTH = 2

    # Each streamer's "step pointer": the next global step index to fetch/store
    streamer_step = [0] * num_ops

    # Buffers: list of deques. For readers, entries are step indices of data
    # that has been fetched. For writers, entries are step indices of data
    # from the accelerator waiting to be written.
    buffers: list[deque[int]] = [deque() for _ in range(num_ops)]"""

OLD_END = """        buffers = next_buffers
        pending_banks = next_pending_banks
        streamer_step = next_streamer_step
        pending_step = next_pending_step
        cycles_for_step_i[-1] += 1  # for debugging: count this cycle towards the current global step


    return cycle"""

NEW_CONTENT = """    # --- Per-streamer state ---

    BUFFER_DEPTH = 2
    AGU_QUEUE_DEPTH = 4  # Typical output buffer depth for AGU

    # Each streamer's AGU "step pointer": the next global step index to generate addresses for
    agu_step = [0] * num_ops

    # Address buffers decoupled from the memory requests
    # Represents the outputBuffer of the AGU module
    address_buffers: list[deque[int]] = [deque() for _ in range(num_ops)]

    # Buffers: list of deques. For readers, entries are step indices of data
    # that has been fetched. For writers, entries are step indices of data
    # from the accelerator waiting to be written.
    buffers: list[deque[int]] = [deque() for _ in range(num_ops)]

    # Per-streamer in-flight burst state: which banks of the current burst
    # still need to be serviced. None means no burst in progress.
    pending_banks: list[set[int] | None] = [None] * num_ops
    # The step index of the currently in-flight burst
    pending_step: list[int] = [0] * num_ops

    # Accelerator step counter
    acc_step = 0

    # Round-robin priority counter for bank arbitration
    rr_priority = 0

    cycle = 0
    MAX_CYCLES = total_steps * num_ops * num_banks * 10  # safety bound

    cycles_for_step_i = [0]  # for debugging: track cycles taken by each global step

    while cycle < MAX_CYCLES:
        # Check termination: AGU done generating addresses, all memory requests have resolved,
        # and buffers empty up to total_steps completion.
        all_done = all(agu_step[op] >= total_steps and
                       len(address_buffers[op]) == 0 and
                       pending_banks[op] is None and
                       len(buffers[op]) == 0
                       for op in range(num_ops))
        if all_done and acc_step >= total_steps:
            break

        cycle += 1

        # ==================================================================
        # Phase 0: Address Generation Unit (AGU)
        # ==================================================================
        
        # Advance agu_step for all ops past non-access steps and append necessary steps to address_buffers
        # Hardware can generate at most 1 access per cycle.
        for op in range(num_ops):
            while agu_step[op] < total_steps and not streamer_needs_access(op, agu_step[op]):
                agu_step[op] += 1
            
            if agu_step[op] < total_steps and len(address_buffers[op]) < AGU_QUEUE_DEPTH:
                address_buffers[op].append(agu_step[op])
                agu_step[op] += 1

        # ==================================================================
        # Phase 1: Determine which streamers want to issue memory requests
        # ==================================================================

        # Collect all individual bank requests for this cycle.
        # A request is (op_idx, bank_idx).
        bank_requests: list[tuple[int, int]] = []

        reader_writer_writing = False  # track if any RW streamer is in its write phase this cycle


        for op in range(num_ops):
            desc = operand_descriptors[op]
            is_reader_writer = desc.kind == OperandKind.READER_WRITER
            is_reader = desc.kind == OperandKind.READER
            is_writer = desc.kind == OperandKind.WRITER

            if is_writer:
                pass
            if is_reader_writer:
                reader_writer_writing
                pass
            
            # Continue a pending burst?
            if pending_banks[op] is not None:
                for bank in pending_banks[op]:
                    bank_requests.append((op, bank))
                if is_writer:
                    reader_writer_writing = True
                continue

            if len(address_buffers[op]) == 0:
                # No addresses to generate memory requests for
                reader_writer_writing = False
                continue

            step = address_buffers[op][0]

            # Check buffer capacity
            if is_reader or is_reader_writer:
                # Reader: must have space in buffer to put fetched data
                reader_writer_writing = False
                if len(buffers[op]) >= BUFFER_DEPTH:
                    continue
                if is_reader_writer and reader_writer_writing:
                    # ReaderWriter in write phase this cycle → read phase is stalled
                    continue

            elif is_writer:
                # Writer: must have data in buffer to write
                if len(buffers[op]) == 0:
                    reader_writer_writing = False
                    continue
                reader_writer_writing = True

            # Start a new burst
            burst_banks = compute_burst_banks(op, step)
            pending_banks[op] = set(burst_banks)
            pending_step[op] = step

            if op == 0:
                pass

            if op == 1:
                pass

            for bank in pending_banks[op]:
                bank_requests.append((op, bank))
            
        if cycle == 12:
            pass

        # ==================================================================
        # Phase 2: Resolve banking conflicts (round-robin arbitration)
        # ==================================================================

        # Group requests by bank
        bank_to_ops: dict[int, list[int]] = {}
        for op, bank in bank_requests:
            bank_to_ops.setdefault(bank, [])
            if op not in bank_to_ops[bank]:
                bank_to_ops[bank].append(op)

        granted: dict[int, set[int]] = {op: set() for op in range(num_ops)}
        for bank, ops in bank_to_ops.items():
            if len(ops) == 1:
                granted[ops[0]].add(bank)
            else:
                # Give priority to the lowest operand
                winner = min(ops)
                granted[winner].add(bank)

        # ==================================================================
        # Phase 2b: Update pending bursts based on grants
        # ==================================================================

        # Track which streamers complete their burst this cycle
        # Use next-state tracking to apply updates atomically at end of cycle
        next_address_buffers: list[deque[int]] = [deque(b) for b in address_buffers]
        next_buffers: list[deque[int]] = [deque(b) for b in buffers]
        next_pending_banks: list[set[int] | None] = list(pending_banks)
        next_agu_step: list[int] = list(agu_step)
        next_pending_step: list[int] = list(pending_step)

        for op in range(num_ops):
            if pending_banks[op] is None:
                continue

            # Remove granted banks from pending set
            remaining = pending_banks[op] - granted[op]
            if len(remaining) == 0:
                # Burst complete
                desc = operand_descriptors[op]
                is_reader = desc.kind in (OperandKind.READER, OperandKind.READER_WRITER)
                is_writer = desc.kind == OperandKind.WRITER

                if is_reader:
                    next_buffers[op].append(pending_step[op])
                elif is_writer:
                    # Writer: data was in buffer, now written to memory
                    # Pop the oldest entry from the buffer
                    if next_buffers[op]:
                        next_buffers[op].popleft()

                next_pending_banks[op] = None
                
                # Consume this completed address from the AGU queue
                if next_address_buffers[op]:
                    next_address_buffers[op].popleft()

                # Advance past any subsequent non-access steps
                # (handled by AGU now)
            else:
                next_pending_banks[op] = remaining

        # ==================================================================
        # Phase 3: Accelerator fire logic
        # ==================================================================

        # The accelerator fires if:
        # 1. For each reader-type streamer: either the required data (for
        #    acc_step) is in its buffer, or no access is needed at acc_step.
        # 2. For each writer-type streamer: there is space in its buffer.

        if cycle == 13:
            pass

        acc_can_fire = acc_step < total_steps
        if acc_step == 514:
            pass
        if acc_can_fire:
            for op in range(num_ops):
                desc = operand_descriptors[op]
                is_reader = desc.kind in (OperandKind.READER, OperandKind.READER_WRITER)
                is_writer = desc.kind == OperandKind.WRITER

                if is_reader:
                    needs_access = streamer_needs_access(op, acc_step)
                    if needs_access:
                        # Data for acc_step must be in the buffer.
                        # Check both current and next-state buffers (data arriving
                        # this cycle is visible to the accelerator).
                        if acc_step not in next_buffers[op]:
                            acc_can_fire = False
                            break
                    # If no access needed, the reader doesn't block the accelerator.

                elif is_writer:
                    # Writer needs space in its buffer to accept the result
                    if len(next_buffers[op]) >= BUFFER_DEPTH:
                        acc_can_fire = False
                        break

        if not acc_can_fire:
            pass

        if acc_can_fire:
            cycles_for_step_i.append(0)
            # Pop consumed data from reader buffers; push to writer buffers
            for op in range(num_ops):
                desc = operand_descriptors[op]
                is_reader = desc.kind in (OperandKind.READER, OperandKind.READER_WRITER)
                is_writer = desc.kind == OperandKind.WRITER

                if is_reader:
                    needs_access = streamer_needs_access(op, acc_step)
                    if needs_access and acc_step in next_buffers[op]:
                        next_buffers[op].remove(acc_step)

                elif is_writer:
                    needs_access = streamer_needs_access(op, acc_step)
                    if needs_access and len(next_buffers[op]) < BUFFER_DEPTH:
                        next_buffers[op].append(acc_step)

            acc_step += 1

        # ==================================================================
        # Phase 4: Commit next-state
        # ==================================================================

        address_buffers = next_address_buffers
        buffers = next_buffers
        pending_banks = next_pending_banks
        agu_step = next_agu_step
        pending_step = next_pending_step
        cycles_for_step_i[-1] += 1  # for debugging: count this cycle towards the current global step


    return cycle"""

start_idx = text.find("    # --- Per-streamer state ---")
end_idx = text.find("return cycle", start_idx) + len("return cycle")

if start_idx != -1 and end_idx != -1:
    with open("snaxc/ir/dart/cost_models.py", "w") as f:
        f.write(text[:start_idx] + NEW_CONTENT + text[end_idx:])
    print("Patched successfully")
else:
    print("Could not find patch bounds")

