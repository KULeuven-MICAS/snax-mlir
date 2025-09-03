// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdarg.h>
#include <stdint.h>
#include "chip_id.h"

#include "occamy_base_addr.h"

#define UART_RBR UART_BASE_ADDR + 0
#define UART_THR UART_BASE_ADDR + 0
#define UART_INTERRUPT_ENABLE UART_BASE_ADDR + 4
#define UART_INTERRUPT_IDENT UART_BASE_ADDR + 8
#define UART_FIFO_CONTROL UART_BASE_ADDR + 8
#define UART_LINE_CONTROL UART_BASE_ADDR + 12
#define UART_MODEM_CONTROL UART_BASE_ADDR + 16
#define UART_LINE_STATUS UART_BASE_ADDR + 20
#define UART_MODEM_STATUS UART_BASE_ADDR + 24
#define UART_DLAB_LSB UART_BASE_ADDR + 0
#define UART_DLAB_MSB UART_BASE_ADDR + 4

/*
    UART_LINE_CONTROL[1:0]: iLCR_WLS Word Length Select, 2'b11 for two bits mode
    UART_LINE_CONTROL[2]: iLCE_STB, 1'b0 for one bit stop bit
    UART_LINE_CONTROL[3]: iLCR_PEN, Parity Enable, disable as there is higher
   level parity check algorithm UART_LINE_CONTROL[4]: iLCR_PES, also related to
   parity check UART_LINE_CONTROL[5]: iLCR_SP, also related to parity check
    UART_LINE_CONTROL[6]: iLCR_BC, signaling a break control
    UART_LINE_CONTROL[7]: iLCR_DLAB, don't know what it is

*/
/*
    UART_MODEM_CONTROL[0]: iMCR_DTR (DTR output, not used)
    UART_MODEM_CONTROL[1]: iMCR_RTS (RTS output, set 1 to inform the device is
   ready to receive the data) UART_MODEM_CONTROL[2]: iMCR_OUT1 (General Purpose
   Output 1, not used) UART_MODEM_CONTROL[3]: iMCR_OUT2 (General Purpose Output
   2, not used) UART_MODEM_CONTROL[4]: iMCR_LOOP (Internal Loopback, should set
   to 0) UART_MODEM_CONTROL[5]: iMCR_AFE (Automatic Flow Control, set to 1 to
   automatically manage DTR and RTS)
*/

inline static void write_reg_u8(uintptr_t addr, uint8_t value) {
    volatile uint8_t *loc_addr = (volatile uint8_t *)addr;
    *loc_addr = value;
}

inline static uint8_t read_reg_u8(uintptr_t addr) {
    return *(volatile uint8_t *)addr;
}

inline static int is_data_ready(uintptr_t address_prefix) {
    return read_reg_u8(address_prefix | UART_LINE_STATUS) & 0x01;
}

inline static int is_data_overrun(uintptr_t address_prefix) {
    return read_reg_u8(address_prefix | UART_LINE_STATUS) & 0x02;
}

inline static int is_transmit_empty(uintptr_t address_prefix) {
    return read_reg_u8(address_prefix | UART_LINE_STATUS) & 0x20;
}

inline static int is_transmit_done(uintptr_t address_prefix) {
    return read_reg_u8(address_prefix | UART_LINE_STATUS) & 0x40;
}

inline static void init_uart(uintptr_t address_prefix, uint32_t freq,
                             uint32_t baud) {
    uint32_t divisor = freq / (baud << 4);

    write_reg_u8(address_prefix | UART_INTERRUPT_ENABLE,
                 0x00);  // Disable all interrupts
    write_reg_u8(address_prefix | UART_LINE_CONTROL,
                 0x80);  // Enable DLAB (set baud rate divisor)
    write_reg_u8(address_prefix | UART_DLAB_LSB, divisor);  // divisor (lo byte)
    write_reg_u8(address_prefix | UART_DLAB_MSB,
                 (divisor >> 8) & 0xFF);  // divisor (hi byte)
    write_reg_u8(address_prefix | UART_LINE_CONTROL,
                 0x03);  // 8 bits, no parity, one stop bit
    write_reg_u8(address_prefix | UART_FIFO_CONTROL,
                 0xC7);  // Enable FIFO, clear them, with 14-byte threshold
    write_reg_u8(address_prefix | UART_MODEM_CONTROL,
                 0x22);  // Flow control enabled, auto flow control mode
}
inline static void print_char(uintptr_t address_prefix, char a) {
    while (is_transmit_empty(address_prefix) == 0) {
    };

    write_reg_u8(address_prefix | UART_THR, a);
}

inline static uint8_t scan_char(uintptr_t address_prefix) {
    while (is_data_ready(address_prefix) == 0) {
    };

    return read_reg_u8(address_prefix | UART_RBR);
}

int printf(const char *fmt, ...);
int scanf(const char *fmt, ...);
