// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "chip_id.h"
#include "occamy.h"
#include "occamy_memory_map.h"

// *Note*: to ensure that the usr_data field is at the same offset
// in the host and device (resp. 64b and 32b architectures)
// usr_data is an explicitly-sized integer field instead of a pointer
typedef struct {
  volatile uint32_t lock;
  volatile uint32_t chip_id;
  volatile uint32_t usr_data_ptr;
  volatile uint32_t chip_barrier;
  // Chip Level synchronization mechanism: 16x16 chip matrix
  volatile uint8_t chip_level_checkpoint[256];
} comm_buffer_t;

/**************/
/* Interrupts */
/**************/

inline static void set_host_sw_interrupt(uint8_t chip_id) {
#if __riscv_xlen == 64
  volatile uint32_t *msip_ptr =
      (uint32_t *)(((uintptr_t)clint_msip_ptr(0)) |
                   ((uintptr_t)get_chip_baseaddress(chip_id)));
  *msip_ptr = 1;
#elif __riscv_xlen == 32
  volatile uint32_t *msip_ptr = clint_msip_ptr(0);
  uint32_t target_addrh = get_chip_baseaddress_value(chip_id) >> 32;
  uint32_t current_addrh = get_current_chip_baseaddress_value() >> 32;

  register uint32_t reg_target_addrh asm("t0") = target_addrh;
  register uint32_t reg_return_value asm("t1") = 1;
  register uint32_t reg_msip_ptr asm("t2") = (uint32_t)msip_ptr;
  register uint32_t reg_current_addrh asm("t3") = current_addrh;

  asm volatile("csrw 0xbc0, t0;"
               "sw t1, 0(t2);"
               "csrw 0xbc0, t3;"
               :
               : "r"(reg_target_addrh), "r"(reg_return_value),
                 "r"(reg_msip_ptr), "r"(reg_current_addrh)
               : "memory");
#endif
}

inline void clear_host_sw_interrupt_unsafe(uint8_t chip_id) {
#if __riscv_xlen == 64
  volatile uint32_t *msip_ptr =
      (uint32_t *)(((uintptr_t)clint_msip_ptr(0)) |
                   ((uintptr_t)get_chip_baseaddress(chip_id)));
  *msip_ptr = 0;
#elif __riscv_xlen == 32
  volatile uint32_t *msip_ptr = clint_msip_ptr(0);
  uint32_t target_addrh = get_chip_baseaddress_value(chip_id) >> 32;
  uint32_t current_addrh = get_current_chip_baseaddress_value() >> 32;

  register uint32_t reg_target_addrh asm("t0") = target_addrh;
  register uint32_t reg_return_value asm("t1") = 0;
  register uint32_t reg_msip_ptr asm("t2") = (uint32_t)msip_ptr;
  register uint32_t reg_current_addrh asm("t3") = current_addrh;

  asm volatile("csrw 0xbc0, t0;"
               "sw t1, 0(t2);"
               "csrw 0xbc0, t3;"
               :
               : "r"(reg_target_addrh), "r"(reg_return_value),
                 "r"(reg_msip_ptr), "r"(reg_current_addrh)
               : "memory");
#endif
}

inline void wait_host_sw_interrupt_clear(uint8_t chip_id) {
#if __riscv_xlen == 64
  volatile uint32_t *msip_ptr =
      (uint32_t *)(((uintptr_t)clint_msip_ptr(0)) |
                   ((uintptr_t)get_chip_baseaddress(chip_id)));
  while (*msip_ptr)
    ;
#elif __riscv_xlen == 32
  volatile uint32_t *msip_ptr = clint_msip_ptr(0);
  uint32_t target_addrh = get_chip_baseaddress_value(chip_id) >> 32;
  uint32_t current_addrh = get_current_chip_baseaddress_value() >> 32;

  register uint32_t reg_target_addrh asm("t0") = target_addrh;
  register uint32_t reg_value asm("t1");
  register uint32_t reg_msip_ptr asm("t2") = (uint32_t)msip_ptr;
  register uint32_t reg_current_addrh asm("t3") = current_addrh;

  do {
    asm volatile("csrw 0xbc0, t0;"
                 "lw t1, 0(t2);"
                 "csrw 0xbc0, t3;"
                 : "=r"(reg_value)
                 : "r"(reg_target_addrh), "r"(reg_msip_ptr),
                   "r"(reg_current_addrh)
                 : "memory");
  } while (reg_value);
#endif
}

static inline void clear_host_sw_interrupt(uint8_t chip_id) {
  clear_host_sw_interrupt_unsafe(chip_id);
  wait_host_sw_interrupt_clear(chip_id);
}

/**************************/
/* Quadrant configuration */
/**************************/

// Configure RO cache address range
inline void configure_read_only_cache_addr_rule(uint8_t chip_id,
                                                uint32_t quad_idx,
                                                uint32_t rule_idx,
                                                uint64_t start_addr,
                                                uint64_t end_addr) {
  volatile uint64_t *rule_ptr =
      (uint64_t *)(((uintptr_t)quad_cfg_ro_cache_addr_rule_ptr(quad_idx,
                                                               rule_idx)) |
                   ((uintptr_t)get_chip_baseaddress(chip_id)));
  *(rule_ptr) = start_addr;
  *(rule_ptr + 1) = end_addr;
}

// Enable RO cache
inline void enable_read_only_cache(uint8_t chip_id, uint32_t quad_idx) {
  volatile uint32_t *enable_ptr =
      (uint32_t *)(((uintptr_t)quad_cfg_ro_cache_enable_ptr(quad_idx)) |
                   ((uintptr_t)get_chip_baseaddress(chip_id)));
  *enable_ptr = 1;
}
