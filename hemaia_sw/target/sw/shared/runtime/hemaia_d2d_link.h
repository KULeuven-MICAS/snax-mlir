// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Yunhao Deng <yunhao.deng@kuleuven.be>

#pragma once
#include "chip_id.h"
#include "hemaia_d2d_link_peripheral.h"
#include "occamy_memory_map.h"
#include <stdint.h>

#define CHANNELS_PER_DIRECTION 3
#define HEMAIA_D2D_LINK_NUM_DELAYS 8
#define HEMAIA_D2D_LINK_BROKEN_LINK_REG_SIZE 20
#define MAX_CFG_ROUND 3

#define HEMAIA_D2D_LINK_DEFAULT_TEST_CYCLES 2000

typedef enum {
  D2D_DIRECTION_EAST = 0,
  D2D_DIRECTION_WEST = 1,
  D2D_DIRECTION_NORTH = 2,
  D2D_DIRECTION_SOUTH = 3
} Direction;

typedef uint8_t bool;
#define true 1
#define false 0

inline void delay_cycles(uint64_t cycle) {
  uint64_t target_cycle, current_cycle;
  __asm__ volatile("csrr %0, mcycle;" : "=r"(current_cycle));
  target_cycle = current_cycle + cycle;
  while (current_cycle < target_cycle) {
    __asm__ volatile("csrr %0, mcycle;" : "=r"(current_cycle));
  }
}

// Reset the D2D link
inline void reset_d2d_link_digital(uint32_t delay) {
  volatile uint32_t *hemaia_d2d_link_reset_addr =
      (volatile uint32_t *)(((uintptr_t)get_current_chip_baseaddress() |
                             HEMAIA_D2D_LINK_BASE_ADDR) +
                            HEMAIA_D2D_LINK_RESET_REGISTER_REG_OFFSET);
  *hemaia_d2d_link_reset_addr = 0xFFFFFFFF; // Reset the d2d link
  delay_cycles(delay);
  *hemaia_d2d_link_reset_addr = 0x0; // Release the reset
}

// Test mode
inline void set_d2d_link_test_mode(Direction direction, bool enable) {
  volatile uint32_t *hemaia_d2d_link_test_mode_addr =
      (volatile uint32_t *)(((uintptr_t)get_current_chip_baseaddress() |
                             HEMAIA_D2D_LINK_BASE_ADDR) +
                            HEMAIA_D2D_LINK_TEST_MODE_REGISTER_REG_OFFSET);
  if (enable) {
    *hemaia_d2d_link_test_mode_addr |= (1 << direction);
  } else {
    *hemaia_d2d_link_test_mode_addr &= ~(1 << direction);
  }
}

inline void set_all_d2d_link_test_mode(bool enable) {
  volatile uint32_t *hemaia_d2d_link_test_mode_addr =
      (volatile uint32_t *)(((uintptr_t)get_current_chip_baseaddress() |
                             HEMAIA_D2D_LINK_BASE_ADDR) +
                            HEMAIA_D2D_LINK_TEST_MODE_REGISTER_REG_OFFSET);
  if (enable) {
    *hemaia_d2d_link_test_mode_addr |= 0x0F; // Set all test modes
  } else {
    *hemaia_d2d_link_test_mode_addr &= ~0x0F; // Clear all test modes
  }
}

inline bool get_d2d_link_being_tested(Direction direction) {
  volatile uint32_t *hemaia_d2d_link_test_mode_addr =
      (volatile uint32_t *)(((uintptr_t)get_current_chip_baseaddress() |
                             HEMAIA_D2D_LINK_BASE_ADDR) +
                            HEMAIA_D2D_LINK_TEST_MODE_REGISTER_REG_OFFSET);
  uint32_t test_mode_status = *hemaia_d2d_link_test_mode_addr;
  return (test_mode_status & (1 << (direction + 4))) != 0;
}

// Link availability
inline void set_d2d_link_availability(Direction direction, bool avail) {
  volatile uint32_t *hemaia_d2d_link_availability_addr =
      (volatile uint32_t *)(((uintptr_t)get_current_chip_baseaddress() |
                             HEMAIA_D2D_LINK_BASE_ADDR) +
                            HEMAIA_D2D_LINK_AVAILABILITY_REGISTER_REG_OFFSET);
  if (avail) {
    *hemaia_d2d_link_availability_addr |= (1 << direction);
  } else {
    *hemaia_d2d_link_availability_addr &= ~(1 << direction);
  }
}
inline void set_all_d2d_link_availability(bool avail) {
  volatile uint32_t *hemaia_d2d_link_availability_addr =
      (volatile uint32_t *)(((uintptr_t)get_current_chip_baseaddress() |
                             HEMAIA_D2D_LINK_BASE_ADDR) +
                            HEMAIA_D2D_LINK_AVAILABILITY_REGISTER_REG_OFFSET);
  if (avail) {
    *hemaia_d2d_link_availability_addr |= 0x0F; // Set all link availability
  } else {
    *hemaia_d2d_link_availability_addr &= ~0x0F; // Clear all link availability
  }
}

inline bool get_d2d_link_availability(Direction direction) {
  volatile uint32_t *hemaia_d2d_link_availability_addr =
      (volatile uint32_t *)(((uintptr_t)get_current_chip_baseaddress() |
                             HEMAIA_D2D_LINK_BASE_ADDR) +
                            HEMAIA_D2D_LINK_AVAILABILITY_REGISTER_REG_OFFSET);
  uint32_t availability_status = *hemaia_d2d_link_availability_addr;
  return (availability_status & (1 << direction)) != 0;
}

// Error count and total count of the D2D link BER test
inline uint32_t get_d2d_link_tested_cycle(Direction direction,
                                          uint8_t channel) {
  uintptr_t base =
      (uintptr_t)get_current_chip_baseaddress() | HEMAIA_D2D_LINK_BASE_ADDR;
  uint32_t offset = 0;
  switch (direction) {
  case D2D_DIRECTION_EAST:
    offset = HEMAIA_D2D_LINK_EAST_C0_TEST_MODE_TOTAL_CYCLE_REGISTER_REG_OFFSET;
    break;
  case D2D_DIRECTION_WEST:
    offset = HEMAIA_D2D_LINK_WEST_C0_TEST_MODE_TOTAL_CYCLE_REGISTER_REG_OFFSET;
    break;
  case D2D_DIRECTION_NORTH:
    offset = HEMAIA_D2D_LINK_NORTH_C0_TEST_MODE_TOTAL_CYCLE_REGISTER_REG_OFFSET;
    break;
  default:
    offset = HEMAIA_D2D_LINK_SOUTH_C0_TEST_MODE_TOTAL_CYCLE_REGISTER_REG_OFFSET;
    break;
  }
  return *((volatile uint32_t *)(base + offset + channel * 4));
}

inline uint8_t get_d2d_link_error_cycle_one_wire(Direction direction,
                                                 uint8_t channel,
                                                 uint8_t wire) {
  uintptr_t base =
      (uintptr_t)get_current_chip_baseaddress() | HEMAIA_D2D_LINK_BASE_ADDR;
  uint8_t group = wire / 4;
  uint8_t subChannel = wire % 4;
  uint32_t offset = 0;
  switch (direction) {
  case D2D_DIRECTION_EAST:
    offset = HEMAIA_D2D_LINK_EAST_C0_TEST_MODE_ERROR_REGISTER_0_REG_OFFSET +
             channel * 20 + group * 4;
    break;
  case D2D_DIRECTION_WEST:
    offset = HEMAIA_D2D_LINK_WEST_C0_TEST_MODE_ERROR_REGISTER_0_REG_OFFSET +
             channel * 20 + group * 4;
    break;
  case D2D_DIRECTION_NORTH:
    offset = HEMAIA_D2D_LINK_NORTH_C0_TEST_MODE_ERROR_REGISTER_0_REG_OFFSET +
             channel * 20 + group * 4;
    break;
  default:
    offset = HEMAIA_D2D_LINK_SOUTH_C0_TEST_MODE_ERROR_REGISTER_0_REG_OFFSET +
             channel * 20 + group * 4;
    break;
  }
  uint32_t value = *((volatile uint32_t *)(base + offset));
  return (value >> (8 * subChannel)) & 0xFF;
}

inline void get_d2d_link_error_cycle_one_channel(Direction direction,
                                                 uint8_t channel,
                                                 uint8_t *dest) {
  uintptr_t base =
      (uintptr_t)get_current_chip_baseaddress() | HEMAIA_D2D_LINK_BASE_ADDR;
  switch (direction) {
  case D2D_DIRECTION_EAST:
    base = base +
           HEMAIA_D2D_LINK_EAST_C0_TEST_MODE_ERROR_REGISTER_0_REG_OFFSET +
           channel * 20;
    break;
  case D2D_DIRECTION_WEST:
    base = base +
           HEMAIA_D2D_LINK_WEST_C0_TEST_MODE_ERROR_REGISTER_0_REG_OFFSET +
           channel * 20;
    break;
  case D2D_DIRECTION_NORTH:
    base = base +
           HEMAIA_D2D_LINK_NORTH_C0_TEST_MODE_ERROR_REGISTER_0_REG_OFFSET +
           channel * 20;
    break;
  default:
    base = base +
           HEMAIA_D2D_LINK_SOUTH_C0_TEST_MODE_ERROR_REGISTER_0_REG_OFFSET +
           channel * 20;
    break;
  }

  uint32_t *dest_addr = (uint32_t *)dest;
  for (uint8_t i = 0; i < 5; i++) {
    dest_addr[i] = ((uint32_t *)base)[i];
  }
}

// FEC unrecoverable error count
inline uint32_t get_fec_unrecoverable_error_count(Direction direction) {
  uint32_t *fec_unrecoverable_error_count_addr =
      (uint32_t *)(((uintptr_t)get_current_chip_baseaddress() |
                    HEMAIA_D2D_LINK_BASE_ADDR) +
                   HEMAIA_D2D_LINK_FEC_UNRECOVERABLE_ERROR_REGISTER_REG_OFFSET);
  return (uint8_t)((*fec_unrecoverable_error_count_addr >> (direction * 8)) &
                   0xFF);
}

// Programmable clock delay
inline void set_d2d_link_clock_delay(Direction direction, uint8_t channel,
                                     uint8_t delay) {
  uintptr_t base =
      (uintptr_t)get_current_chip_baseaddress() | HEMAIA_D2D_LINK_BASE_ADDR;
  switch (direction) {
  case D2D_DIRECTION_EAST:
    base = base +
           HEMAIA_D2D_LINK_EAST_PROGRAMMABLE_CLOCK_DELAY_REGISTER_REG_OFFSET;
    break;
  case D2D_DIRECTION_WEST:
    base = base +
           HEMAIA_D2D_LINK_WEST_PROGRAMMABLE_CLOCK_DELAY_REGISTER_REG_OFFSET;
    break;
  case D2D_DIRECTION_NORTH:
    base = base +
           HEMAIA_D2D_LINK_NORTH_PROGRAMMABLE_CLOCK_DELAY_REGISTER_REG_OFFSET;
    break;
  default:
    base = base +
           HEMAIA_D2D_LINK_SOUTH_PROGRAMMABLE_CLOCK_DELAY_REGISTER_REG_OFFSET;
    break;
  }

  volatile uint32_t *reg = (volatile uint32_t *)(base);
  uint32_t current = *reg;
  uint32_t shift = channel * 8;
  current &= ~(0xFF << shift);
  current |= ((uint32_t)delay & 0xFF) << shift;
  *reg = current;
}

inline void set_d2d_link_clock_delay_all_channels(Direction direction,
                                                  uint8_t delay) {
  uintptr_t base =
      (uintptr_t)get_current_chip_baseaddress() | HEMAIA_D2D_LINK_BASE_ADDR;
  switch (direction) {
  case D2D_DIRECTION_EAST:
    base = base +
           HEMAIA_D2D_LINK_EAST_PROGRAMMABLE_CLOCK_DELAY_REGISTER_REG_OFFSET;
    break;
  case D2D_DIRECTION_WEST:
    base = base +
           HEMAIA_D2D_LINK_WEST_PROGRAMMABLE_CLOCK_DELAY_REGISTER_REG_OFFSET;
    break;
  case D2D_DIRECTION_NORTH:
    base = base +
           HEMAIA_D2D_LINK_NORTH_PROGRAMMABLE_CLOCK_DELAY_REGISTER_REG_OFFSET;
    break;
  default:
    base = base +
           HEMAIA_D2D_LINK_SOUTH_PROGRAMMABLE_CLOCK_DELAY_REGISTER_REG_OFFSET;
    break;
  }

  volatile uint32_t *reg = (volatile uint32_t *)(base);
  *reg = ((uint32_t)delay << 24) | ((uint32_t)delay << 16) |
         ((uint32_t)delay << 8) | delay;
}

inline uint8_t get_d2d_link_clock_delay(Direction direction, uint8_t channel) {
  uintptr_t base =
      (uintptr_t)get_current_chip_baseaddress() | HEMAIA_D2D_LINK_BASE_ADDR;
  switch (direction) {
  case D2D_DIRECTION_EAST:
    base = base +
           HEMAIA_D2D_LINK_EAST_PROGRAMMABLE_CLOCK_DELAY_REGISTER_REG_OFFSET;
    break;
  case D2D_DIRECTION_WEST:
    base = base +
           HEMAIA_D2D_LINK_WEST_PROGRAMMABLE_CLOCK_DELAY_REGISTER_REG_OFFSET;
    break;
  case D2D_DIRECTION_NORTH:
    base = base +
           HEMAIA_D2D_LINK_NORTH_PROGRAMMABLE_CLOCK_DELAY_REGISTER_REG_OFFSET;
    break;
  default:
    base = base +
           HEMAIA_D2D_LINK_SOUTH_PROGRAMMABLE_CLOCK_DELAY_REGISTER_REG_OFFSET;
    break;
  }

  volatile uint32_t *reg = (volatile uint32_t *)(base);
  return (uint8_t)((*reg >> (channel * 8)) & 0xFF);
}

// Programmable fault link bypass
inline void set_d2d_link_broken_link(Direction direction, uint8_t channel,
                                     uint8_t broken_link) {
  uintptr_t base =
      (uintptr_t)get_current_chip_baseaddress() | HEMAIA_D2D_LINK_BASE_ADDR;
  switch (direction) {
  case D2D_DIRECTION_EAST:
    base = base + HEMAIA_D2D_LINK_EAST_BROKEN_WIRE_REGISTER_REG_OFFSET;
    break;
  case D2D_DIRECTION_WEST:
    base = base + HEMAIA_D2D_LINK_WEST_BROKEN_WIRE_REGISTER_REG_OFFSET;
    break;
  case D2D_DIRECTION_NORTH:
    base = base + HEMAIA_D2D_LINK_NORTH_BROKEN_WIRE_REGISTER_REG_OFFSET;
    break;
  default:
    base = base + HEMAIA_D2D_LINK_SOUTH_BROKEN_WIRE_REGISTER_REG_OFFSET;
    break;
  }

  volatile uint32_t *reg = (volatile uint32_t *)(base);
  uint32_t current = *reg;
  uint32_t shift = channel * 8;
  current &= ~(0xFF << shift);
  current |= ((uint32_t)broken_link & 0xFF) << shift;
  *reg = current;
}

inline uint8_t get_d2d_link_broken_link(Direction direction, uint8_t channel) {
  uintptr_t base =
      (uintptr_t)get_current_chip_baseaddress() | HEMAIA_D2D_LINK_BASE_ADDR;
  switch (direction) {
  case D2D_DIRECTION_EAST:
    base = base + HEMAIA_D2D_LINK_EAST_BROKEN_WIRE_REGISTER_REG_OFFSET;
    break;
  case D2D_DIRECTION_WEST:
    base = base + HEMAIA_D2D_LINK_WEST_BROKEN_WIRE_REGISTER_REG_OFFSET;
    break;
  case D2D_DIRECTION_NORTH:
    base = base + HEMAIA_D2D_LINK_NORTH_BROKEN_WIRE_REGISTER_REG_OFFSET;
    break;
  default:
    base = base + HEMAIA_D2D_LINK_SOUTH_BROKEN_WIRE_REGISTER_REG_OFFSET;
    break;
  }

  volatile uint32_t *reg = (volatile uint32_t *)(base);
  return (uint8_t)((*reg >> (channel * 8)) & 0xFF);
}

// Driving strength
inline void set_d2d_link_driving_strength(uint8_t strength,
                                          Direction direction) {
  volatile uint32_t *hemaia_d2d_link_driving_strength_addr =
      (volatile uint32_t
           *)(((uintptr_t)get_current_chip_baseaddress() |
               HEMAIA_D2D_LINK_BASE_ADDR) +
              HEMAIA_D2D_LINK_PROGRAMMABLE_DRIVE_STRENGTH_REGISTER_REG_OFFSET);
  *hemaia_d2d_link_driving_strength_addr =
      (*hemaia_d2d_link_driving_strength_addr & ~(0xFF << (direction * 8))) |
      (strength << (direction * 8));
}

inline void set_all_d2d_link_driving_strength(uint8_t strength) {
  volatile uint32_t *hemaia_d2d_link_driving_strength_addr =
      (volatile uint32_t
           *)(((uintptr_t)get_current_chip_baseaddress() |
               HEMAIA_D2D_LINK_BASE_ADDR) +
              HEMAIA_D2D_LINK_PROGRAMMABLE_DRIVE_STRENGTH_REGISTER_REG_OFFSET);
  *hemaia_d2d_link_driving_strength_addr =
      (strength << 24) | (strength << 16) | (strength << 8) | strength;
}

inline uint8_t get_d2d_link_driving_strength(Direction direction) {
  volatile uint32_t *hemaia_d2d_link_driving_strength_addr =
      (volatile uint32_t
           *)(((uintptr_t)get_current_chip_baseaddress() |
               HEMAIA_D2D_LINK_BASE_ADDR) +
              HEMAIA_D2D_LINK_PROGRAMMABLE_DRIVE_STRENGTH_REGISTER_REG_OFFSET);
  uint32_t strength_value = *hemaia_d2d_link_driving_strength_addr;
  return (strength_value >> (direction * 8)) & 0xFF;
}

// RX buffer threshold
inline void set_d2d_link_rx_buffer_threshold(uint8_t threshold,
                                             Direction direction) {
  volatile uint32_t *hemaia_d2d_link_threshold_addr =
      (volatile uint32_t
           *)(((uintptr_t)get_current_chip_baseaddress() |
               HEMAIA_D2D_LINK_BASE_ADDR) +
              HEMAIA_D2D_LINK_RX_BUFFER_THRESHOLD_REGISTER_REG_OFFSET);
  *hemaia_d2d_link_threshold_addr =
      (*hemaia_d2d_link_threshold_addr & ~(0xFF << (direction * 8))) |
      (threshold << (direction * 8));
}

inline void set_all_d2d_link_rx_buffer_threshold(uint8_t threshold) {
  volatile uint32_t *hemaia_d2d_link_threshold_addr =
      (volatile uint32_t
           *)(((uintptr_t)get_current_chip_baseaddress() |
               HEMAIA_D2D_LINK_BASE_ADDR) +
              HEMAIA_D2D_LINK_RX_BUFFER_THRESHOLD_REGISTER_REG_OFFSET);
  *hemaia_d2d_link_threshold_addr =
      (threshold << 24) | (threshold << 16) | (threshold << 8) |
      threshold; // Set the threshold to the passed parameter
}

inline uint8_t get_d2d_link_rx_buffer_threshold(Direction direction) {
  volatile uint32_t *hemaia_d2d_link_threshold_addr =
      (volatile uint32_t
           *)(((uintptr_t)get_current_chip_baseaddress() |
               HEMAIA_D2D_LINK_BASE_ADDR) +
              HEMAIA_D2D_LINK_RX_BUFFER_THRESHOLD_REGISTER_REG_OFFSET);
  uint32_t threshold_value = *hemaia_d2d_link_threshold_addr;
  return (threshold_value >> (direction * 8)) & 0xFF;
}

// TX Backoff period
inline void set_d2d_link_tx_backoff_period(uint8_t period,
                                           Direction direction) {
  volatile uint32_t *reg =
      (volatile uint32_t
           *)(((uintptr_t)get_current_chip_baseaddress() |
               HEMAIA_D2D_LINK_BASE_ADDR) +
              HEMAIA_D2D_LINK_TX_BACKOFF_PERIOD_REGISTER_REG_OFFSET);
  *reg = (*reg & ~(0xFF << (direction * 8))) | (period << (direction * 8));
}

inline void set_all_d2d_link_tx_backoff_period(uint8_t period) {
  volatile uint32_t *reg =
      (volatile uint32_t
           *)(((uintptr_t)get_current_chip_baseaddress() |
               HEMAIA_D2D_LINK_BASE_ADDR) +
              HEMAIA_D2D_LINK_TX_BACKOFF_PERIOD_REGISTER_REG_OFFSET);
  *reg = (period << 24) | (period << 16) | (period << 8) | period;
}

inline uint8_t get_d2d_link_tx_backoff_period(Direction direction) {
  volatile uint32_t *reg =
      (volatile uint32_t
           *)(((uintptr_t)get_current_chip_baseaddress() |
               HEMAIA_D2D_LINK_BASE_ADDR) +
              HEMAIA_D2D_LINK_TX_BACKOFF_PERIOD_REGISTER_REG_OFFSET);
  uint32_t val = *reg;
  return (val >> (direction * 8)) & 0xFF;
}

// TX Hold period
inline void set_d2d_link_tx_hold_period(uint8_t period, Direction direction) {
  uintptr_t base =
      (uintptr_t)get_current_chip_baseaddress() | HEMAIA_D2D_LINK_BASE_ADDR;
  base += HEMAIA_D2D_LINK_TX_HOLD_PERIOD_REGISTER_REG_OFFSET;
  volatile uint32_t *reg = (volatile uint32_t *)base;
  uint32_t val = *reg;
  uint32_t shift = direction * 8U;
  val &= ~(0xFFU << shift);
  val |= ((uint32_t)period & 0xFFU) << shift;
  *reg = val;
}

inline void set_all_d2d_link_tx_hold_period(uint8_t period) {
  uintptr_t base =
      (uintptr_t)get_current_chip_baseaddress() | HEMAIA_D2D_LINK_BASE_ADDR;
  base += HEMAIA_D2D_LINK_TX_HOLD_PERIOD_REGISTER_REG_OFFSET;
  volatile uint32_t *reg = (volatile uint32_t *)base;
  uint32_t val = ((uint32_t)period << 24) | ((uint32_t)period << 16) |
                 ((uint32_t)period << 8) | period;
  *reg = val;
}

inline uint8_t get_d2d_link_tx_hold_period(Direction direction) {
  uintptr_t base =
      (uintptr_t)get_current_chip_baseaddress() | HEMAIA_D2D_LINK_BASE_ADDR;
  base += HEMAIA_D2D_LINK_TX_HOLD_PERIOD_REGISTER_REG_OFFSET;
  volatile uint32_t *reg = (volatile uint32_t *)base;
  uint32_t val = *reg;
  return (val >> (direction * 8U)) & 0xFFU;
}

// Operating mode (False means RX mode; True means TX mode)
inline bool get_d2d_link_operating_mode(Direction direction) {
  uintptr_t base =
      (uintptr_t)get_current_chip_baseaddress() | HEMAIA_D2D_LINK_BASE_ADDR;
  volatile uint32_t *reg =
      (volatile uint32_t
           *)(base + HEMAIA_D2D_LINK_TX_MODE_MONITOR_REGISTER_REG_OFFSET);
  uint32_t val = *reg;
  uint8_t shift;
  switch (direction) {
  case D2D_DIRECTION_EAST:
    shift = HEMAIA_D2D_LINK_TX_MODE_MONITOR_REGISTER_EAST_TX_MODE_BIT;
    break;
  case D2D_DIRECTION_WEST:
    shift = HEMAIA_D2D_LINK_TX_MODE_MONITOR_REGISTER_WEST_TX_MODE_BIT;
    break;
  case D2D_DIRECTION_NORTH:
    shift = HEMAIA_D2D_LINK_TX_MODE_MONITOR_REGISTER_NORTH_TX_MODE_BIT;
    break;
  default:
    shift = HEMAIA_D2D_LINK_TX_MODE_MONITOR_REGISTER_SOUTH_TX_MODE_BIT;
    break;
  }
  return ((val >> shift) & 1U) != 0U;
}

// TX mode clock frequency
inline void set_d2d_link_tx_mode_clock_frequency(uint8_t frequency,
                                                 Direction direction) {
  volatile uint32_t *hemaia_d2d_link_tx_mode_clock_frequency_addr =
      (volatile uint32_t
           *)(((uintptr_t)get_current_chip_baseaddress() |
               HEMAIA_D2D_LINK_BASE_ADDR) +
              HEMAIA_D2D_LINK_TX_MODE_CLOCK_FREQUENCY_REGISTER_REG_OFFSET);
  *hemaia_d2d_link_tx_mode_clock_frequency_addr =
      (*hemaia_d2d_link_tx_mode_clock_frequency_addr &
       ~(0xFF << (direction * 8))) |
      (frequency << (direction * 8));
}

inline void set_all_d2d_link_tx_mode_clock_frequency(uint8_t frequency) {
  volatile uint32_t *hemaia_d2d_link_tx_mode_clock_frequency_addr =
      (volatile uint32_t
           *)(((uintptr_t)get_current_chip_baseaddress() |
               HEMAIA_D2D_LINK_BASE_ADDR) +
              HEMAIA_D2D_LINK_TX_MODE_CLOCK_FREQUENCY_REGISTER_REG_OFFSET);
  *hemaia_d2d_link_tx_mode_clock_frequency_addr =
      (frequency << 24) | (frequency << 16) | (frequency << 8) | frequency;
}

inline uint8_t get_d2d_link_tx_mode_clock_frequency(Direction direction) {
  volatile uint32_t *hemaia_d2d_link_tx_mode_clock_frequency_addr =
      (volatile uint32_t
           *)(((uintptr_t)get_current_chip_baseaddress() |
               HEMAIA_D2D_LINK_BASE_ADDR) +
              HEMAIA_D2D_LINK_TX_MODE_CLOCK_FREQUENCY_REGISTER_REG_OFFSET);
  uint32_t frequency_value = *hemaia_d2d_link_tx_mode_clock_frequency_addr;
  return (frequency_value >> (direction * 8)) & 0xFF;
}

// Clock gating delay
inline void set_d2d_link_clock_gating_delay(uint8_t delay,
                                            Direction direction) {
  volatile uint32_t *hemaia_d2d_link_clock_gating_delay_addr =
      (volatile uint32_t
           *)(((uintptr_t)get_current_chip_baseaddress() |
               HEMAIA_D2D_LINK_BASE_ADDR) +
              HEMAIA_D2D_LINK_PHY_CLOCK_GATING_DELAY_REGISTER_REG_OFFSET);
  *hemaia_d2d_link_clock_gating_delay_addr =
      (*hemaia_d2d_link_clock_gating_delay_addr & ~(0xFF << (direction * 8))) |
      (delay << (direction * 8));
}
inline void set_all_d2d_link_clock_gating_delay(uint8_t delay) {
  volatile uint32_t *hemaia_d2d_link_clock_gating_delay_addr =
      (volatile uint32_t
           *)(((uintptr_t)get_current_chip_baseaddress() |
               HEMAIA_D2D_LINK_BASE_ADDR) +
              HEMAIA_D2D_LINK_PHY_CLOCK_GATING_DELAY_REGISTER_REG_OFFSET);
  *hemaia_d2d_link_clock_gating_delay_addr =
      (delay << 24) | (delay << 16) | (delay << 8) | delay;
}
inline uint8_t get_d2d_link_clock_gating_delay(Direction direction) {
  volatile uint32_t *hemaia_d2d_link_clock_gating_delay_addr =
      (volatile uint32_t
           *)(((uintptr_t)get_current_chip_baseaddress() |
               HEMAIA_D2D_LINK_BASE_ADDR) +
              HEMAIA_D2D_LINK_PHY_CLOCK_GATING_DELAY_REGISTER_REG_OFFSET);
  uint32_t delay_value = *hemaia_d2d_link_clock_gating_delay_addr;
  return (delay_value >> (direction * 8)) & 0xFF;
}

// The period to enable Clock Transmission before data transmission. Should be
// not smaller than 1 Cycle
inline void set_d2d_link_data_transmission_delay(uint8_t delay,
                                                 Direction direction) {
  volatile uint32_t *hemaia_d2d_link_data_transmission_delay_addr =
      (volatile uint32_t
           *)(((uintptr_t)get_current_chip_baseaddress() |
               HEMAIA_D2D_LINK_BASE_ADDR) +
              HEMAIA_D2D_LINK_PHY_DATA_TRANSMISSION_DELAY_REGISTER_REG_OFFSET);
  *hemaia_d2d_link_data_transmission_delay_addr =
      (*hemaia_d2d_link_data_transmission_delay_addr &
       ~(0xFF << (direction * 8))) |
      (delay << (direction * 8));
}

inline void set_all_d2d_link_data_transmission_delay(uint8_t delay) {
  volatile uint32_t *hemaia_d2d_link_data_transmission_delay_addr =
      (volatile uint32_t
           *)(((uintptr_t)get_current_chip_baseaddress() |
               HEMAIA_D2D_LINK_BASE_ADDR) +
              HEMAIA_D2D_LINK_PHY_DATA_TRANSMISSION_DELAY_REGISTER_REG_OFFSET);
  *hemaia_d2d_link_data_transmission_delay_addr =
      (delay << 24) | (delay << 16) | (delay << 8) | delay;
}

inline uint8_t get_d2d_link_data_transmission_delay(Direction direction) {
  volatile uint32_t *hemaia_d2d_link_data_transmission_delay_addr =
      (volatile uint32_t
           *)(((uintptr_t)get_current_chip_baseaddress() |
               HEMAIA_D2D_LINK_BASE_ADDR) +
              HEMAIA_D2D_LINK_PHY_DATA_TRANSMISSION_DELAY_REGISTER_REG_OFFSET);
  uint32_t delay_value = *hemaia_d2d_link_data_transmission_delay_addr;
  return (delay_value >> (direction * 8)) & 0xFF;
}
