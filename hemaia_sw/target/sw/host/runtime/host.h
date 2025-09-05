// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// #include "host.h"
#include "chip_id.h"
#include "heterogeneous_runtime.h"
#include "occamy.h"
#include "sys_dma.h"
#include "uart.c"
#include <stddef.h>
#include <stdint.h>

extern uint64_t __narrow_spm_start;
extern uint64_t __narrow_spm_end;
extern uint64_t __wide_spm_start;
extern uint64_t __wide_spm_end;

// Handle multireg degeneration to single register
#if OCCAMY_SOC_ISOLATE_MULTIREG_COUNT == 1
#define OCCAMY_SOC_ISOLATE_0_REG_OFFSET OCCAMY_SOC_ISOLATE_REG_OFFSET
#define OCCAMY_SOC_ISOLATE_0_ISOLATE_0_MASK OCCAMY_SOC_ISOLATE_ISOLATE_0_MASK
#endif
#if OCCAMY_SOC_ISOLATED_MULTIREG_COUNT == 1
#define OCCAMY_SOC_ISOLATED_0_REG_OFFSET OCCAMY_SOC_ISOLATED_REG_OFFSET
#endif
#if OCCAMY_SOC_SCRATCH_MULTIREG_COUNT == 1
#define OCCAMY_SOC_SCRATCH_0_REG_OFFSET OCCAMY_SOC_SCRATCH_0_REG_OFFSET
#endif
#if OCCAMY_SOC_SCRATCH_MULTIREG_COUNT == 1
#define OCCAMY_SOC_SCRATCH_0_REG_OFFSET OCCAMY_SOC_SCRATCH_REG_OFFSET
#endif

//===============================================================
// RISC-V
//===============================================================

#define MIP_MTIP_OFFSET 7
#define MIP_MSIP_OFFSET 3
#define MIE_MSIE_OFFSET 3
#define MIE_MTIE_OFFSET 7
#define MSTATUS_MIE_OFFSET 3
#define MSTATUS_FS_OFFSET 13

// //===============================================================
// // Memory map pointers
// //===============================================================

// #if SELECT_FLL == 0  // ETH FLL
// volatile uint32_t* const fll_system_base =
//     (volatile uint32_t*)FLL_SYSTEM_BASE_ADDR;
// volatile uint32_t* const fll_periph_base =
//     (volatile uint32_t*)FLL_PERIPH_BASE_ADDR;
// volatile uint32_t* const fll_hbm2e_base =
//     (volatile uint32_t*)FLL_HBM2E_BASE_ADDR;
// #elif SELECT_FLL == 1  // GF FLL
// volatile uint32_t* const fll_system_base =
//     (volatile uint32_t*)FLL_SYSTEM_BASE_ADDR + (0x200 >> 2);
// volatile uint32_t* const fll_periph_base =
//     (volatile uint32_t*)FLL_PERIPH_BASE_ADDR + (0x200 >> 2);
// volatile uint32_t* const fll_hbm2e_base =
//     (volatile uint32_t*)FLL_HBM2E_BASE_ADDR + (0x200 >> 2);
// #endif

// volatile uint32_t* const fll_base[N_CLOCKS] = {fll_system_base,
// fll_periph_base, fll_hbm2e_base};

volatile uint64_t *const clint_mtime_ptr =
    (volatile uint64_t *)(CLINT_BASE_ADDR + CLINT_MTIME_LOW_REG_OFFSET);
volatile uint64_t *const clint_mtimecmp0_ptr =
    (volatile uint64_t *)(CLINT_BASE_ADDR + CLINT_MTIMECMP_LOW0_REG_OFFSET);

//===============================================================
// Globals
//===============================================================

// volatile comm_buffer_t comm_buffer __attribute__((aligned(8)));

//===============================================================
// Anticipated function declarations
//===============================================================

static inline void set_sw_interrupts_unsafe(uint8_t chip_id,
                                            uint32_t base_hartid,
                                            uint32_t num_harts,
                                            uint32_t stride);

static inline void set_sw_interrupt(uint8_t chip_id,
                                    volatile comm_buffer_t *comm_buffer_ptr,
                                    uint32_t hartid);

void delay_ns(uint64_t delay);

static inline volatile uint32_t *
get_shared_lock(volatile comm_buffer_t *comm_buffer_ptr);

static inline void wait_sw_interrupt();

static inline void clear_sw_interrupt(uint8_t chip_id,
                                      volatile comm_buffer_t *comm_buffer_ptr,
                                      uint32_t hartid);

//===============================================================
// Initialization
//===============================================================

void initialize_bss() {
  extern volatile uint64_t __bss_start, __bss_end;

  size_t bss_size = (size_t)(&__bss_end) - (size_t)(&__bss_start);
  if (bss_size)
    sys_dma_blk_memcpy((uint64_t)(&__bss_start), WIDE_ZERO_MEM_BASE_ADDR,
                       bss_size);
}

void initialize_wide_spm() {
  size_t wide_spm_size =
      (size_t)(&__wide_spm_end) - (size_t)(&__wide_spm_start);
  if (wide_spm_size)
    sys_dma_blk_memcpy(
        (uint64_t)get_current_chip_baseaddress() | SPM_WIDE_BASE_ADDR,
        (uint64_t)get_current_chip_baseaddress() | WIDE_ZERO_MEM_BASE_ADDR,
        wide_spm_size);
}

void initialize_narrow_spm() {
  size_t narrow_spm_size =
      (size_t)(&__narrow_spm_end) - (size_t)(&__narrow_spm_start);
  if (narrow_spm_size)
    sys_dma_blk_memcpy(
        (uint64_t)get_current_chip_baseaddress() | SPM_NARROW_BASE_ADDR,
        (uint64_t)get_current_chip_baseaddress() | WIDE_ZERO_MEM_BASE_ADDR,
        narrow_spm_size);
}

void initialize_comm_buffer(comm_buffer_t *comm_buffer_ptr) {
  sys_dma_blk_memcpy((uint64_t)comm_buffer_ptr,
                     (uint64_t)get_current_chip_baseaddress() |
                         WIDE_ZERO_MEM_BASE_ADDR,
                     sizeof(comm_buffer_t));
  asm volatile("fence" ::: "memory");
}

void enable_fpu() {
  uint64_t mstatus;

  asm volatile("csrr %[mstatus], mstatus" : [mstatus] "=r"(mstatus));
  mstatus |= (1 << MSTATUS_FS_OFFSET);
  asm volatile("csrw mstatus, %[mstatus]" : : [mstatus] "r"(mstatus));
}

void set_d_cache_enable(uint16_t ena) {
  asm volatile("csrw 0x701, %0" ::"r"(ena));
}

//===============================================================
// Synchronization and mutual exclusion
//===============================================================

static inline void fence() { asm volatile("fence" : : : "memory"); }

/**
 * @brief lock a mutex, blocking
 * @details test-and-set (tas) implementation of a lock.
 *          Declare mutex with `static volatile uint32_t mtx = 0;`
 */
void mutex_tas_acquire(volatile uint32_t *pmtx) {
  asm volatile("li            x5,1          # x5 = 1\n"
               "1:\n"
               "  amoswap.w.aq  x5,x5,(%0)   # x5 = oldlock & lock = 1\n"
               "  bnez          x5,1b      # Retry if previously set)\n"
               : "+r"(pmtx)
               :
               : "x5");
}

/**
 * @brief lock a mutex, blocking
 * @details test-and-test-and-set (ttas) implementation of a lock.
 *          Declare mutex with `static volatile uint32_t mtx = 0;`
 */
static inline void mutex_ttas_acquire(volatile uint32_t *pmtx) {
  asm volatile("1:\n"
               "  lw x5, 0(%0)\n"
               "  bnez x5, 1b\n"
               "  li x5,1          # x5 = 1\n"
               "2:\n"
               "  amoswap.w.aq  x5,x5,(%0)   # x5 = oldlock & lock = 1\n"
               "  bnez          x5,2b      # Retry if previously set)\n"
               : "+r"(pmtx)
               :
               : "x5");
}

/**
 * @brief Release the mutex
 */
static inline void mutex_release(volatile uint32_t *pmtx) {
  asm volatile("amoswap.w.rl  x0,x0,(%0)   # Release lock by storing 0\n"
               : "+r"(pmtx));
}

//===============================================================
// Device programming
//===============================================================

extern void snitch_main();

static inline void wakeup_snitch(uint8_t chip_id,
                                 volatile comm_buffer_t *comm_buffer_ptr,
                                 uint32_t hartid) {
  set_sw_interrupt(chip_id, comm_buffer_ptr, hartid);
}

/**
 * @brief Waits until snitches are parked in a `wfi` instruction
 *
 * @detail delays execution to wait for the Snitch cores to be ready.
 *         After being parked, the Snitch cores can accept an interrupt
 *         and start executing its binary
 */
// TODO: implement in a more robust manner
void wait_snitches_parked(uint32_t timeout) { delay_ns(100000); }

/**
 * @brief Programs the Snitches with the Snitch binary
 *
 * @detail After boot, the Snitches are "parked" on a WFI
 *         until they receive a software interrupt. Upon
 *         wakeup, the Snitch jumps to a minimal interrupt
 *         handler in boot ROM which loads the address of the
 *         user binary from the soc_ctrl_scratch_0 register.
 *         This routine programs the soc_ctrl_scratch_0 register
 *         with the address of the user binary.
 */
static inline void program_snitches(uint8_t chip_id,
                                    volatile comm_buffer_t *comm_buffer_ptr) {
  uintptr_t base_addr = (uintptr_t)get_chip_baseaddress(chip_id);
  *(volatile uint32_t *)((uintptr_t)soc_ctrl_scratch_ptr(1) | base_addr) =
      (uintptr_t)snitch_main;
  *(volatile uint32_t *)((uintptr_t)soc_ctrl_scratch_ptr(2) | base_addr) =
      (uintptr_t)comm_buffer_ptr;
}

/**
 * @brief Wake-up a Snitch cluster
 *
 * @detail Send a cluster interrupt to all Snitches in a cluster
 */

static inline void wakeup_cluster(uint8_t chip_id, uint32_t cluster_id) {
  uintptr_t base_addr = (uintptr_t)get_chip_baseaddress(chip_id);
  *(volatile uint32_t *)((uintptr_t)cluster_clint_set_ptr(cluster_id) |
                         base_addr) = 511;
}

/**
 * @brief Wake-up Snitches
 *
 * @detail All Snitches are "parked" in a WFI. A SW interrupt
 *         must be issued to "unpark" every Snitch. This function
 *         sends a SW interrupt to all Snitches.
 */
void wakeup_snitches(uint8_t chip_id, volatile comm_buffer_t *comm_buffer_ptr) {
  volatile uint32_t *lock = get_shared_lock(comm_buffer_ptr);

  mutex_ttas_acquire(lock);
  set_sw_interrupts_unsafe(chip_id, 1, N_SNITCHES, 1);
  mutex_release(lock);
}

/**
 * @brief Wake-up Snitches
 *
 * @detail Send a cluster interrupt to all Snitches
 */
static inline void wakeup_snitches_cl(uint8_t chip_id) {
  for (int i = 0; i < N_CLUSTERS; i++)
    wakeup_cluster(chip_id, i);
}

/**
 * @brief Wake-up Snitches
 *
 * @detail All Snitches are "parked" in a WFI. A SW interrupt
 *         must be issued to "unpark" every Snitch. This function
 *         sends a SW interrupt to a given range of Snitches.
 */
void wakeup_snitches_selective(uint8_t chip_id,
                               volatile comm_buffer_t *comm_buffer_ptr,
                               uint32_t base_hartid, uint32_t num_harts,
                               uint32_t stride) {
  volatile uint32_t *lock = get_shared_lock(comm_buffer_ptr);

  mutex_ttas_acquire(lock);
  set_sw_interrupts_unsafe(chip_id, base_hartid, num_harts, stride);
  mutex_release(lock);
}

// temporary deprecate this function since it uses the N_CORES_PER_CLUSTER

// /**
//  * @brief Wake-up Snitches
//  *
//  * @detail All Snitches are "parked" in a WFI. A SW interrupt
//  *         must be issued to "unpark" every Snitch. This function
//  *         sends a SW interrupt to one Snitch in every cluster,
//  *         the so called "master" of the cluster. The "master" is
//  *         then expected to wake-up all the other Snitches in its
//  *         cluster. The "master" Snitches can use the cluster-local
//  *         CLINTs without sending requests outside the cluster,
//  *         avoiding congestion.
//  */
// void wakeup_master_snitches() {
//     volatile uint32_t* lock = get_shared_lock();

//     mutex_ttas_acquire(lock);
//     set_sw_interrupts_unsafe(1, N_CLUSTERS, N_CORES_PER_CLUSTER);
//     mutex_release(lock);
// }

/**
 * @brief Waits until snitches are done executing
 */
static inline int wait_snitches_done(uint8_t chip_id) {
  wait_sw_interrupt();
  uint8_t current_chip_id = get_current_chip_id();
  clear_host_sw_interrupt(current_chip_id);

  uintptr_t baseaddress = (uintptr_t)get_chip_baseaddress(chip_id);
  uint32_t *retval_ptr =
      (uint32_t *)((uintptr_t)soc_ctrl_scratch_ptr(3) | baseaddress);
  int retval = *retval_ptr;
  // LSB signals completion
  if (retval & 1)
    return retval >> 1;
  else
    return -1;
}

static inline volatile uint32_t *
get_shared_lock(volatile comm_buffer_t *comm_buffer_ptr) {
  return &((*comm_buffer_ptr).lock);
}

//===============================================================
// Reset and clock gating
//===============================================================

static inline void set_clk_ena_quad(uint8_t chip_id, uint32_t quad_idx,
                                    uint32_t value,
                                    uint32_t cluster_clk_enable_mask) {
  volatile uint32_t *clk_ena_ptr =
      (volatile uint32_t *)((uintptr_t)quad_cfg_clk_ena_ptr(quad_idx) |
                            (uintptr_t)get_chip_baseaddress(chip_id));
  *clk_ena_ptr = value & cluster_clk_enable_mask;
}

// static inline void set_clk_ena_quad(uint32_t quad_idx, uint32_t value) {
//     *quad_cfg_clk_ena_ptr(quad_idx) = value & 0x1;
// }

static inline void set_reset_n_quad(uint8_t chip_id, uint32_t quad_idx,
                                    uint32_t value) {
  volatile uint32_t *reset_n_ptr =
      (volatile uint32_t *)((uintptr_t)quad_cfg_reset_n_ptr(quad_idx) |
                            (uintptr_t)get_chip_baseaddress(chip_id));
  *reset_n_ptr = value & 0x1;
}

static inline void reset_and_ungate_quad(uint8_t chip_id, uint32_t quadrant_idx,
                                         uint32_t cluster_clk_enable_mask) {
  set_reset_n_quad(chip_id, quadrant_idx, 0);
  set_clk_ena_quad(chip_id, quadrant_idx, 0, cluster_clk_enable_mask);
  set_reset_n_quad(chip_id, quadrant_idx, 0xFFFFFFFF);
  set_clk_ena_quad(chip_id, quadrant_idx, 0xFFFFFFFF, cluster_clk_enable_mask);
}

static inline void
reset_and_ungate_quadrants(uint8_t chip_id, uint32_t cluster_clk_enable_mask) {
  for (int i = 0; i < N_QUADS; i++)
    reset_and_ungate_quad(chip_id, i, cluster_clk_enable_mask);
}

static inline void reset_and_ungate_quadrants_all(uint8_t chip_id) {
  for (int i = 0; i < N_QUADS; i++)
    reset_and_ungate_quad(chip_id, i, 0xFFFFFFFF);
}
//===============================================================
// Interrupts
//===============================================================

static inline void wfi() { asm volatile("wfi"); }

static inline void enable_sw_interrupts() {
  uint64_t mie;

  asm volatile("csrr %[mie], mie" : [mie] "=r"(mie));
  mie |= (1 << MIE_MSIE_OFFSET);
  asm volatile("csrw mie, %[mie]" : : [mie] "r"(mie));
}

static inline uint32_t get_clint_msip_hart(uint8_t chip_id, uint32_t hartid) {
  uintptr_t base_addr = (uintptr_t)get_chip_baseaddress(chip_id);
  uint32_t field_offset = hartid % CLINT_MSIP_P_FIELDS_PER_REG;
  uint32_t lsb_offset = field_offset * CLINT_MSIP_P_FIELD_WIDTH;
  return (*(volatile uint32_t *)((uintptr_t)clint_msip_ptr(hartid) |
                                 base_addr) >>
          lsb_offset) &
         1;
}

/**
 * @brief Gets SW interrupt pending status from local CSR
 *
 * @detail Use this in favour of remote_sw_interrupt_pending()
 *         when polling a core's own interrupt pending
 *         status. This avoids unnecessary congestion on the
 *         interconnect and shared CLINT.
 */
static inline uint32_t sw_interrupt_pending() {
  uint64_t mip;

  asm volatile("csrr %[mip], mip" : [mip] "=r"(mip));
  return mip & (1 << MIP_MSIP_OFFSET);
}

// TODO: for portability to architectures where WFI is implemented as a NOP
//       also sw_interrupts_enabled() should be checked
static inline void wait_sw_interrupt() {
  do
    wfi();
  while (!sw_interrupt_pending());
}

static inline void clear_sw_interrupt_unsafe(uint8_t chip_id, uint32_t hartid) {
  uintptr_t base_addr = (uintptr_t)get_chip_baseaddress(chip_id);
  uint32_t field_offset = hartid % CLINT_MSIP_P_FIELDS_PER_REG;
  uint32_t lsb_offset = field_offset * CLINT_MSIP_P_FIELD_WIDTH;

  *(volatile uint32_t *)((uintptr_t)clint_msip_ptr(hartid) | base_addr) &=
      ~(1 << lsb_offset);
}

static inline void clear_sw_interrupt(uint8_t chip_id,
                                      volatile comm_buffer_t *comm_buffer_ptr,
                                      uint32_t hartid) {
  volatile uint32_t *shared_lock = get_shared_lock(comm_buffer_ptr);

  mutex_tas_acquire(shared_lock);
  clear_sw_interrupt_unsafe(chip_id, hartid);
  mutex_release(shared_lock);
}

/**
 * @brief Gets SW interrupt pending status from CLINT
 *
 * @detail Use sw_interrupt_pending() in favour of this
 *         when polling a core's own interrupt pending
 *         status. That function interrogates a local CSR
 *         instead of the shared CLINT.
 */
static inline uint32_t remote_sw_interrupt_pending(uint8_t chip_id,
                                                   uint32_t hartid) {
  return get_clint_msip_hart(chip_id, hartid);
}

static inline uint32_t timer_interrupts_enabled() {
  uint64_t mie;
  asm volatile("csrr %[mie], mie" : [mie] "=r"(mie));
  return (mie >> MIE_MTIE_OFFSET) & 1;
}

static inline void set_sw_interrupt_unsafe(uint8_t chip_id, uint32_t hartid) {
  uintptr_t base_addr = (uintptr_t)get_chip_baseaddress(chip_id);
  uint32_t field_offset = hartid % CLINT_MSIP_P_FIELDS_PER_REG;
  uint32_t lsb_offset = field_offset * CLINT_MSIP_P_FIELD_WIDTH;

  *(volatile uint32_t *)((uintptr_t)clint_msip_ptr(hartid) | base_addr) |=
      (1 << lsb_offset);
}

void set_sw_interrupt(uint8_t chip_id, volatile comm_buffer_t *comm_buffer_ptr,
                      uint32_t hartid) {
  volatile uint32_t *shared_lock = get_shared_lock(comm_buffer_ptr);
  mutex_ttas_acquire(shared_lock);
  set_sw_interrupt_unsafe(chip_id, hartid);
  mutex_release(shared_lock);
}

static inline void set_sw_interrupts_unsafe(uint8_t chip_id,
                                            uint32_t base_hartid,
                                            uint32_t num_harts,
                                            uint32_t stride) {
  uintptr_t base_addr = (uintptr_t)get_chip_baseaddress(chip_id);
  volatile uint32_t *ptr =
      (uint32_t *)((uintptr_t)clint_msip_ptr(base_hartid) | base_addr);

  uint32_t num_fields = num_harts;
  uint32_t field_idx = base_hartid;
  uint32_t field_offset = field_idx % CLINT_MSIP_P_FIELDS_PER_REG;
  uint32_t reg_idx = field_idx / CLINT_MSIP_P_FIELDS_PER_REG;
  uint32_t prev_reg_idx = reg_idx;
  uint32_t mask = 0;
  uint32_t reg_jump;
  uint32_t last_field = num_fields - 1;

  for (uint32_t i = 0; i < num_fields; i++) {
    // put field in mask
    mask |= 1 << field_offset;

    // calculate next field info
    field_idx += stride;
    field_offset = field_idx % CLINT_MSIP_P_FIELDS_PER_REG;
    reg_idx = field_idx / CLINT_MSIP_P_FIELDS_PER_REG;
    reg_jump = reg_idx - prev_reg_idx;

    // if next value is in another register
    if (i != last_field && reg_jump) {
      // store mask
      if (mask == (uint32_t)(-1))
        *ptr = mask;
      else
        *ptr |= mask;
      // update pointer and reset mask
      ptr += reg_jump;
      prev_reg_idx = reg_idx;
      mask = 0;
    }
  }

  // store last mask
  *ptr |= mask;
}

void set_cluster_interrupt(uint8_t chip_id, uint32_t cluster_id,
                           uint32_t core_id) {
  uintptr_t base_addr = (uintptr_t)get_chip_baseaddress(chip_id);
  *(volatile uint32_t *)((uintptr_t)cluster_clint_set_ptr(cluster_id) |
                         base_addr) = (1 << core_id);
}

static inline uint32_t timer_interrupt_pending() {
  uint64_t mip;

  asm volatile("csrr %[mip], mip" : [mip] "=r"(mip));
  return mip & (1 << MIP_MTIP_OFFSET);
}

void wait_timer_interrupt() {
  do
    wfi();
  while (!timer_interrupt_pending() && timer_interrupts_enabled());
}

void enable_global_interrupts() {
  uint64_t mstatus;

  asm volatile("csrr %[mstatus], mstatus" : [mstatus] "=r"(mstatus));
  mstatus |= (1 << MSTATUS_MIE_OFFSET);
  asm volatile("csrw mstatus, %[mstatus]" : : [mstatus] "r"(mstatus));
}

void enable_timer_interrupts() {
  uint64_t mie;

  asm volatile("csrr %[mie], mie" : [mie] "=r"(mie));
  mie |= (1 << MIE_MTIE_OFFSET);
  asm volatile("csrw mie, %[mie]" : : [mie] "r"(mie));
}

void disable_timer_interrupts() {
  uint64_t mie;

  asm volatile("csrr %[mie], mie" : [mie] "=r"(mie));
  mie &= ~(1 << MIE_MTIE_OFFSET);
  asm volatile("csrw mie, %[mie]" : : [mie] "r"(mie));
}

void disable_sw_interrupts() {
  uint64_t mie;

  asm volatile("csrr %[mie], mie" : [mie] "=r"(mie));
  mie &= ~(1 << MIE_MSIE_OFFSET);
  asm volatile("csrw mie, %[mie]" : : [mie] "r"(mie));
}

/**
 * @brief Gets SW interrupt pending status from local CSR
 *
 * @detail Use this in favour of wait_remote_sw_interrupt_pending()
 *         when polling a core's own interrupt pending
 *         status. This avoids unnecessary congestion on the
 *         interconnect and shared CLINT.
 */
void wait_sw_interrupt_cleared() {
  while (sw_interrupt_pending())
    ;
}

/**
 * @brief Gets SW interrupt pending status from shared CLINT
 *
 * @detail Use wait_sw_interrupt_cleared() in favour of this
 *         when polling a core's own interrupt pending
 *         status. That function polls a local CSR instead
 *         of the shared CLINT.
 */
void wait_remote_sw_interrupt_pending(uint8_t chip_id, uint32_t hartid) {
  while (remote_sw_interrupt_pending(chip_id, hartid))
    ;
}

//===============================================================
// Timers
//===============================================================

static const float rtc_period = 30517.58; // ns

static inline uint64_t mcycle() {
  register uint64_t r;
  asm volatile("csrr %0, mcycle" : "=r"(r));
  return r;
}

static inline uint64_t mtime(uint8_t chip_id) {
  uintptr_t base_addr = (uintptr_t)get_chip_baseaddress(chip_id);
  return *(volatile uint64_t *)((uintptr_t)clint_mtime_ptr | base_addr);
}

void set_timer_interrupt(uint8_t chip_id, uint64_t interval_ns) {
  // Convert ns to RTC unit
  uint64_t rtc_interval = interval_ns / (int64_t)rtc_period;

  // Calculate the base address for the chip
  uintptr_t base_addr = (uintptr_t)get_chip_baseaddress(chip_id);

  // Offset interval by current time and set the timer interrupt
  *(volatile uint64_t *)((uintptr_t)clint_mtimecmp0_ptr | base_addr) =
      mtime(chip_id) + rtc_interval;
}

/**
 * @brief Clears timer interrupt
 *
 * @detail Pending timer interrupts are cleared in HW when
 *         writing to the mtimecmp register. Note that
 *         eventually the mtime register is going to be greater
 *         than the newly programmed mtimecmp register, reasserting
 *         the pending bit. If this is not desired, it is safer
 *         to disable the timer interrupt before clearing it.
 */
void clear_timer_interrupt(uint8_t chip_id) {
  uintptr_t base_addr = (uintptr_t)get_chip_baseaddress(chip_id);
  *(volatile uint64_t *)((uintptr_t)clint_mtimecmp0_ptr | base_addr) =
      mtime(chip_id) + 1;
}

// Minimum delay is of one RTC period
void delay_ns(uint64_t delay) {
  uint8_t chip_id = get_current_chip_id();
  set_timer_interrupt(chip_id, delay);

  // Wait for set_timer_interrupt() to have effect
  fence();
  enable_timer_interrupts();

  wait_timer_interrupt();
  disable_timer_interrupts();
  clear_timer_interrupt(chip_id);
}

//===============================================================
// Clocks and FLLs
//===============================================================

// #define N_LOCK_CYCLES 10

// typedef enum { SYSTEM_CLK = 0, PERIPH_CLK = 1, HBM2E_CLK = 2 } clk_t;

// static inline void fll_reg_write_u32(clk_t clk, uint32_t byte_offset,
// uint32_t val) {
//     *(fll_base[clk] + (byte_offset / 4)) = val;
// }

// static inline uint32_t fll_reg_read_u32(clk_t clk, uint32_t byte_offset) {
//     return *(fll_base[clk] + (byte_offset / 4));
// }

/**
 * @brief Returns the multiplier to the reference frequency of the FLL
 */
// uint32_t get_fll_freq(clk_t clk) {
// #if SELECT_FLL==0 // ETH FLL
//     return fll_reg_read_u32(clk, ETH_FLL_STATUS_I_REG_OFFSET) &
//     ETH_FLL_STATUS_I_MULTIPLIER_MASK;
// #elif SELECT_FLL==1 // GF FLL
//     return fll_reg_read_u32(clk, FLL_FREQ_REG_OFFSET);
// #endif
// }

// uint32_t fll_locked(clk_t clk) {
// #if SELECT_FLL==0 // ETH FLL
//     return fll_reg_read_u32(clk, ETH_FLL_LOCK_REG_OFFSET) &
//     ETH_FLL_LOCK_LOCKED_MASK;
// #elif SELECT_FLL==1 // GF FLL
//     return fll_reg_read_u32(clk, FLL_STATE_REG_OFFSET) == 3;
// #endif
// }

/**
 * @brief Measures frequency of clock source
 *
 * @return Frequency in GHz
 */
// float measure_frequency(clk_t clk) {
//     return freq_meter_ref_freqs[clk] * get_fll_freq(clk);
// }

/**
 * @brief Derives system frequency through RISC-V's
 *        mtime memory-mapped register and mcycle CSR
 *
 * @param rtc_cycles Number of RTC cycles to wait for measurement.
 *                   The higher it is, the more precise the measurement.
 * @return Frequency in GHz
 */
// float measure_system_frequency(uint32_t rtc_cycles) {
//     uint64_t start_cycle;
//     uint64_t end_cycle;
//     float time_delta;
//     uint64_t cycle_delta;

//     // Compute time delta
//     time_delta = rtc_cycles * rtc_period;

//     // Measure cycle delta
//     start_cycle = mcycle();
//     delay_ns(time_delta);
//     end_cycle = mcycle();
//     cycle_delta = end_cycle - start_cycle;

//     // Return frequency
//     return cycle_delta / time_delta;
// }

/**
 * @brief Reprogram the FLL in closed-loop mode with the specified divider
 * @detail Blocking function, returns after the new frequency is locked
 */
// void program_fll(clk_t clk, uint32_t divider) {
// #if SELECT_FLL==0 // ETH FLL
//     // Reconfigure FLL
//     uint32_t val = 0;
//     val |= 1 << 31;     // Select closed loop mode
//     val |= 1 << 30;     // Gate output by LOCK signal
//     val |= 1 << 26;     // Set post-clock divider to 1 (neutral)
//     val |= divider - 1; // Set refclk multiplier to specified value
//     fll_reg_write_u32(clk, ETH_FLL_CONFIG_I_REG_OFFSET, val);
//     // Wait new frequency locked
//     while (!fll_locked(clk));
// #elif SELECT_FLL==1 // GF FLL
//     // Fallback to reference clock during reconfiguration
//     fll_reg_write_u32(clk, FLL_BYPASS_REG_OFFSET,      1);
//     // Disable DFG IP clock generation during reconfiguration
//     fll_reg_write_u32(clk, FLL_CLKGENEN_REG_OFFSET,    0);
//     // Reconfigure DFG IP input signals
//     fll_reg_write_u32(clk, FLL_FIXLENMODE_REG_OFFSET,  0); // Closed-loop
//     mode fll_reg_write_u32(clk, FLL_FBDIV_REG_OFFSET,       divider - 1);
//     fll_reg_write_u32(clk, FLL_CLKDIV_REG_OFFSET,      0);
//     fll_reg_write_u32(clk, FLL_CLKSRCSEL_REG_OFFSET,   1);
//     // Reconfigure lock settings
//     fll_reg_write_u32(clk, FLL_UPPERTHRESH_REG_OFFSET, divider + 1);
//     fll_reg_write_u32(clk, FLL_LOWERTHRESH_REG_OFFSET, divider - 1);
//     fll_reg_write_u32(clk, FLL_LOCKCYCLES_REG_OFFSET,  N_LOCK_CYCLES);
//     // Enable DFG IP clock generation after new settings are applied
//     fll_reg_write_u32(clk, FLL_CLKGENEN_REG_OFFSET,    1);
//     // Wait new frequency locked
//     while (!fll_locked(clk));
//     // Disable bypass of DFG clock
//     fll_reg_write_u32(clk, FLL_BYPASS_REG_OFFSET,      0);
// #endif
// }

//===============================================================
// Isolation
//===============================================================

uint32_t const ISO_MASK_ALL = 0b1111;
uint32_t const ISO_MASK_NONE = 0;

static inline void deisolate_quad(uint8_t chip_id, uint32_t quad_idx,
                                  uint32_t iso_mask) {
  uintptr_t base_addr = (uintptr_t)get_chip_baseaddress(chip_id);
  volatile uint32_t *isolate_ptr =
      (volatile uint32_t *)((uintptr_t)quad_cfg_isolate_ptr(quad_idx) |
                            base_addr);
  *isolate_ptr &= ~iso_mask;
}

/**
 * @brief Loads the "isolated" register field for the quadrant requested
 *
 * @return Masked register field realigned to start at LSB
 */
static inline uint32_t get_quad_cfg_isolated(uint8_t chip_id,
                                             uint32_t quad_idx) {
  uintptr_t base_addr = (uintptr_t)get_chip_baseaddress(chip_id);
  return *(volatile uint32_t *)((uintptr_t)quad_cfg_isolated_ptr(quad_idx) |
                                base_addr) &
         ISO_MASK_ALL;
}

void isolate_quad(uint8_t chip_id, uint32_t quad_idx, uint32_t iso_mask) {
  uintptr_t base_addr = (uintptr_t)get_chip_baseaddress(chip_id);
  volatile uint32_t *isolate_ptr =
      (volatile uint32_t *)((uintptr_t)quad_cfg_isolate_ptr(quad_idx) |
                            base_addr);
  *isolate_ptr |= iso_mask;
  fence();
}

static inline void deisolate_all(uint8_t chip_id) {
  for (uint32_t i = 0; i < N_QUADS; ++i) {
    deisolate_quad(chip_id, i, ISO_MASK_ALL);
  }
}

/**
 * @brief Check quadrant isolated or not
 *
 * @param iso_mask set bit to 1 to check if path is isolated, 0 de-isolated
 * @return 1 is check passes, 0 otherwise
 */
uint32_t check_isolated_timeout(uint8_t chip_id, uint32_t max_tries,
                                uint32_t quadrant_idx, uint32_t iso_mask) {
  for (uint32_t i = 0; i < max_tries; ++i) {
    if (get_quad_cfg_isolated(chip_id, quadrant_idx) == iso_mask) {
      return 1;
    }
  }
  return 0;
}

//===============================================================
// SoC configuration
//===============================================================

// void activate_interleaved_mode_hbm() {
//     uint64_t addr =
//         OCCAMY_HBM_XBAR_INTERLEAVED_ENA_REG_OFFSET + HBM_XBAR_CFG_BASE_ADDR;
//     *((volatile uint32_t*)addr) = 1;
// }

// void deactivate_interleaved_mode_hbm() {
//     uint64_t addr =
//         OCCAMY_HBM_XBAR_INTERLEAVED_ENA_REG_OFFSET + HBM_XBAR_CFG_BASE_ADDR;
//     *((volatile uint32_t*)addr) = 1;
// }

//===============================================================
// Chip Level Synchronization Mechanism
//===============================================================

void announce_chip_checkpoint(volatile comm_buffer_t *comm_buffer_ptr,
                              uint8_t checkpoint) {
  volatile uint8_t *this_chip_checkpoint =
      &((*comm_buffer_ptr).chip_level_checkpoint[get_current_chip_id()]);
  // Broadcast to all Chips
  this_chip_checkpoint =
      (uint8_t *)(((uint64_t)this_chip_checkpoint) | (((uint64_t)0xFF) << 40));
  *this_chip_checkpoint = checkpoint;
}

void wait_chip_checkpoint(volatile comm_buffer_t *comm_buffer_ptr,
                          uint8_t chip_id, uint8_t checkpoint) {
  volatile uint8_t *target_chip_checkpoint =
      &((*comm_buffer_ptr).chip_level_checkpoint[chip_id]);
  // Broadcast to all Chips
  while (*target_chip_checkpoint != checkpoint) {
    asm volatile("fence.i" ::: "memory");
  }
}

void wait_chips_checkpoint(volatile comm_buffer_t *comm_buffer_ptr,
                           uint8_t top_left_chip_id,
                           uint8_t bottom_right_chip_id, uint8_t checkpoint) {
  volatile uint8_t *chip_level_checkpoint =
      &((*comm_buffer_ptr).chip_level_checkpoint[0]);
  uint8_t current_chip_id = get_current_chip_id();
  uint8_t continue_loop = 1;
  while (continue_loop) {
    continue_loop = 0;
    asm volatile("fence.i" ::: "memory");
    for (uint8_t i = top_left_chip_id >> 4; i <= (bottom_right_chip_id >> 4);
         i++) {
      for (uint8_t j = top_left_chip_id & 0xF;
           j <= (bottom_right_chip_id & 0xF); j++) {
        if ((*(chip_level_checkpoint + ((i << 4) + j)) != checkpoint) &&
            (current_chip_id != ((i << 4) + j))) {
          continue_loop = 1;
          break;
        }
      }
    }
  }
}

// Barrier is realized in software, to ensure that all other chips have reached
// a certain checkpoint
void chip_barrier(volatile comm_buffer_t *comm_buffer_ptr,
                  uint8_t top_left_chip_id, uint8_t bottom_right_chip_id,
                  uint8_t checkpoint) {
  volatile uint8_t *chip_level_checkpoint =
      &((*comm_buffer_ptr).chip_level_checkpoint[0]);
  // Broadcast to all other chip on the progress of the chip
  announce_chip_checkpoint(comm_buffer_ptr, checkpoint);
  // Change the pointer back
  wait_chips_checkpoint(comm_buffer_ptr, top_left_chip_id, bottom_right_chip_id,
                        checkpoint);
}
