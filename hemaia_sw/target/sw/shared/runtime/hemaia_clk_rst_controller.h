// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Yunhao Deng <yunhao.deng@kuleuven.be>

#pragma once
#include "chip_id.h"
#include "hemaia_clk_rst_controller_peripheral.h"
#include "occamy_memory_map.h"
#include <stdint.h>

// Reset the HeMAIA domain
inline void reset_all_clk_domain() {
  volatile uint32_t *hemaia_clk_rst_controller_reset_addr =
      (volatile uint32_t
           *)(((uintptr_t)get_current_chip_baseaddress() |
               HEMAIA_CLK_RST_CONTROLLER_BASE_ADDR) +
              HEMAIA_CLK_RST_CONTROLLER_RESET_REGISTER_REG_OFFSET);
  *hemaia_clk_rst_controller_reset_addr = 0xFFFFFFFF; // Reset the clk domain
}

// Reset one specific clock domain
inline void reset_clk_domain(uint8_t domain) {
  volatile uint32_t *hemaia_clk_rst_controller_reset_addr =
      (volatile uint32_t
           *)(((uintptr_t)get_current_chip_baseaddress() |
               HEMAIA_CLK_RST_CONTROLLER_BASE_ADDR) +
              HEMAIA_CLK_RST_CONTROLLER_RESET_REGISTER_REG_OFFSET);
  *hemaia_clk_rst_controller_reset_addr = (1 << domain); // Reset the clk domain
}

// Disable the clock of one specific clock domain
inline void disable_clk_domain(uint8_t domain) {
  volatile uint32_t *hemaia_clk_rst_controller_clock_valid_reg =
      (volatile uint32_t
           *)(((uintptr_t)get_current_chip_baseaddress() |
               HEMAIA_CLK_RST_CONTROLLER_BASE_ADDR) +
              HEMAIA_CLK_RST_CONTROLLER_CLOCK_VALID_REGISTER_REG_OFFSET);

  volatile uint32_t *hemaia_clk_rst_controller_clock_division_reg =
      (volatile uint32_t
           *)(((uintptr_t)get_current_chip_baseaddress() |
               HEMAIA_CLK_RST_CONTROLLER_BASE_ADDR) +
              HEMAIA_CLK_RST_CONTROLLER_CLOCK_DIVISION_REGISTER_C0_C3_REG_OFFSET +
              domain / 4 * 4);
  uint32_t mask = 0xFF << ((domain % 4) * 8);
  *hemaia_clk_rst_controller_clock_division_reg &=
      ~mask; // Clear the corresponding byte
  *hemaia_clk_rst_controller_clock_valid_reg = 1 << domain; // Set the valid bit
}

inline void enable_clk_domain(uint8_t domain, uint8_t division) {
  volatile uint32_t *hemaia_clk_rst_controller_clock_valid_reg =
      (volatile uint32_t
           *)(((uintptr_t)get_current_chip_baseaddress() |
               HEMAIA_CLK_RST_CONTROLLER_BASE_ADDR) +
              HEMAIA_CLK_RST_CONTROLLER_CLOCK_VALID_REGISTER_REG_OFFSET);

  volatile uint32_t *hemaia_clk_rst_controller_clock_division_reg =
      (volatile uint32_t
           *)(((uintptr_t)get_current_chip_baseaddress() |
               HEMAIA_CLK_RST_CONTROLLER_BASE_ADDR) +
              HEMAIA_CLK_RST_CONTROLLER_CLOCK_DIVISION_REGISTER_C0_C3_REG_OFFSET +
              domain / 4 * 4);
  uint32_t division_register_val =
      *hemaia_clk_rst_controller_clock_division_reg;
  uint32_t shift = (domain % 4) * 8;
  division_register_val &= ~(0xFF << shift); // Clear the corresponding byte
  division_register_val |=
      ((division & 0xFF) << shift); // Set the new division value
  *hemaia_clk_rst_controller_clock_division_reg = division_register_val;
  *hemaia_clk_rst_controller_clock_valid_reg =
      (1 << domain); // Set the valid bit
}
