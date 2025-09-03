// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

.section .text.startup

.globl _start
_start:
#   la          sp, __stack_pointer$
#   la          gp, __global_pointer$
# Save the registers into stack
    addi        sp, sp, -16
    sd          ra, 8(sp)
    sd          s0, 0(sp)
    call        initialize_bss
#   call        initialize_wide_spm
#   call        initialize_narrow_spm
    call        enable_fpu
    call        main

_exit:
    # Process return value
    slli        a0, a0, 1
    ori         a0, a0, 1   # (exit code << 1) | 1

    # Return exit code to host to terminate fesvr simulation
    la          t0, tohost
    sw          a0, 0(t0)

    # Return exit code to host to terminate hemaia simulation
    # Load chip_id to t1, and get base address
    csrr        t0, 0xf15
    slli        t0, t0, 40
    la          t1, __return_pointer$
    or          t0, t1, t0
    sw          a0, 0(t0)
    fence.i

    # Resume the data from the stack
    ld          ra, 8(sp)
    ld          s0, 0(sp)
    addi        sp, sp, 16
    ret

# HTIF sections
.pushsection .htif,"aw",@progbits;
.align 6; .global tohost; tohost: .dword 0;
.align 6; .global fromhost; fromhost: .dword 0;

.section .devicebin,"a",@progbits
.globl snitch_main
.align 4
snitch_main:
#ifdef DEVICEBIN
    .incbin DEVICEBIN
#endif
