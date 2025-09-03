/* Copyright 2020 ETH Zurich and University of Bologna. */
/* Solderpad Hardware License, Version 0.51, see LICENSE for details. */
/* SPDX-License-Identifier: SHL-0.51 */

ENTRY(_start)

MEMORY
{
    NARROW_SPM (rwxa) : ORIGIN = 0x70000000, LENGTH = 32K
    WIDE_SPM (rwxa) : ORIGIN = 0x80000000, LENGTH = 16M
}

SECTIONS
{
  /* The program code and data goes into WIDE_SPM */
  .appl :
  {
    /* __stack_pointer$  = . + 0x70000; */
    /* __global_pointer$ = . + 0x7f0; */
    __return_pointer$ = ORIGIN(WIDE_SPM) + LENGTH(WIDE_SPM) - 4;
    *(.text.startup)
    *(.text .text.*)
    __SDATA_BEGIN__ = .;
    *(.sdata)
    *(.srodata.cst16) *(.srodata.cst8) *(.srodata.cst4) *(.srodata.cst2) *(.srodata .srodata.*)
    *(.sdata .sdata.* .gnu.linkonce.s.*)
    *(.data)
    *(.rodata .rodata.*)
  } > WIDE_SPM

  /* HTIF section for FESVR */
  .htif : { *(.htif) } > WIDE_SPM

  .bss (NOLOAD) :
  {
    . = ALIGN(8);
    __bss_start = . ;
    *(.sbss .sbss.*)
    *(.bss .bss.*)
    *(COMMON)
    . = ALIGN(8);
    __bss_end = . ;
  } > WIDE_SPM

  .wide_spm :
  {
    __wide_spm_start = ORIGIN(WIDE_SPM);
    __wide_spm_end = ORIGIN(WIDE_SPM) + LENGTH(WIDE_SPM);
  } > WIDE_SPM

  .narrow_spm :
  {
    __narrow_spm_start = ORIGIN(NARROW_SPM);
    __narrow_spm_end = ORIGIN(NARROW_SPM) + LENGTH(NARROW_SPM);
  } > WIDE_SPM

  __end = .;

  .devicebin : { *(.devicebin) } > WIDE_SPM

  /* Discard sections */
  /DISCARD/ : { *(.riscv.attributes) *(.comment) }
}
