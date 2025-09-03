// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
#include "uart.h"

static uintptr_t base_address;

static inline int out_ch(char c) {
    print_char(base_address, c);
    return 0;
}

static int print_unsigned(uint32_t value, uint32_t base, int uppercase,
                          int width, int zero_pad) {
    char buf[11];  // enough for 32‑bit 0xFFFFFFFF or 10 decimal digits
    int idx = 0;

    // 1. Convert to string in reverse order
    do {
        uint32_t digit = value % base;
        value /= base;
        buf[idx++] = (digit < 10) ? ('0' + digit)
                                  : ((uppercase ? 'A' : 'a') + (digit - 10));
    } while (value && idx < (int)sizeof(buf));

    // 2. Padding (if width > length)
    int pad_len = (width > idx) ? (width - idx) : 0;
    for (int i = 0; i < pad_len; ++i) out_ch(zero_pad ? '0' : ' ');

    // 3. Output in correct order
    int count = pad_len;
    while (idx--) count += out_ch(buf[idx]);

    return count;
}

// ---------------------------------------------------------------------------
// print_signed() – handles sign then delegates to print_unsigned().
// ---------------------------------------------------------------------------
static int print_signed(int32_t value, int width, int zero_pad) {
    int count = 0;
    if (value < 0) {
        count += out_ch('-');
        value = -value;
        if (width) width--;  // sign consumed one slot
    }
    count += print_unsigned((uint32_t)value, 10, 0, width, zero_pad);
    return count;
}

// ---------------------------------------------------------------------------
// printf() – minimal subset
// ---------------------------------------------------------------------------
int printf(const char *fmt, ...) {
    base_address = (uintptr_t)get_current_chip_baseaddress();
    va_list ap;
    va_start(ap, fmt);
    int total = 0;

    while (*fmt) {
        if (*fmt != '%') {
            total += out_ch(*fmt++);
            continue;
        }

        // We are at a '%'
        ++fmt;
        int zero_pad = 0;
        int width = 0;

        // ---- Parse flags ----
        if (*fmt == '0') {
            zero_pad = 1;
            ++fmt;
        }

        // ---- Parse width ----
        while (*fmt >= '0' && *fmt <= '9') {
            width = width * 10 + (*fmt++ - '0');
        }

        // ---- Conversion specifier ----
        char spec = *fmt ? *fmt++ : '\0';
        switch (spec) {
            case 'c': {
                char c = (char)va_arg(ap, int);
                total += out_ch(c);
            } break;

            case 's': {
                const char *s = va_arg(ap, const char *);
                if (!s) s = "(null)";
                int len = 0;
                while (s[len]) len++;
                // Width & padding
                if (width > len) {
                    for (int i = 0; i < width - len; ++i) total += out_ch(' ');
                }
                for (int i = 0; s[i]; ++i) total += out_ch(s[i]);
            } break;

            case 'd':
                total += print_signed(va_arg(ap, int), width, zero_pad);
                break;

            case 'u':
                total += print_unsigned(va_arg(ap, unsigned int), 10, 0, width,
                                        zero_pad);
                break;
            case 'p':
                total += print_unsigned(va_arg(ap, unsigned int), 16, 0, width,
                                        zero_pad);
                break;
            case 'x':
                total += print_unsigned(va_arg(ap, unsigned int), 16, 0, width,
                                        zero_pad);
                break;

            case 'X':
                total += print_unsigned(va_arg(ap, unsigned int), 16, 1, width,
                                        zero_pad);
                break;

            case '%':
                total += out_ch('%');
                break;

            default:  // Unknown spec => output verbatim
                total += out_ch('%');
                total += out_ch(spec);
                break;
        }
    }

    va_end(ap);
    return total;
}

// Scanf implementation
static int peek;
static int getc_in(void)  // fetch next char or peeked
{
    int c = peek;
    if (c >= 0) {
        peek = -1;
        return c;
    } else
        return scan_char(base_address);
}
static void ungetc_in(int c)  // push char back
{
    peek = c;
}
static int is_space(int c)  // simple isspace()
{
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
           c == '\v';
}
static int hex_val(int c)  // returns 0‑15 or -1
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}
// Read unsigned integer of given base, max "width" chars (0 = unlimited).
static int read_uint(uint32_t *out, int base, int width) {
    uint32_t val = 0;
    int got = 0;
    while (1) {
        if (width && got >= width) break;
        int c = getc_in();
        int d;
        if (base == 10) {
            if (c < '0' || c > '9') {
                ungetc_in(c);
                break;
            }
            d = c - '0';
        } else {
            d = hex_val(c);
            if (d < 0 || d >= base) {
                ungetc_in(c);
                break;
            }
        }
        val = val * base + (uint32_t)d;
        ++got;
    }
    if (!got) return 0;
    *out = val;
    return 1;
}
// Read signed decimal (optionally width), sets *out, returns 1 on success.
static int read_int(int32_t *out, int width) {
    int c, sign = 1;
    do {
        c = getc_in();
    } while (is_space(c));
    if (c == '+' || c == '-') {
        if (c == '-') sign = -1;
        width = width ? width - 1 : 0;
    } else
        ungetc_in(c);
    uint32_t u;
    if (!read_uint(&u, 10, width)) return 0;
    *out = (int32_t)u * sign;
    return 1;
}
// Read 0x / 0X prefix if present.
static void eat_0x_prefix(int *width) {
    int c1 = getc_in();
    int c2 = getc_in();
    if (c1 == '0' && (c2 == 'x' || c2 == 'X')) {
        if (*width > 2)
            *width -= 2;
        else if (*width)
            *width = 0;
    } else {
        ungetc_in(c2);
        ungetc_in(c1);
    }
}
// Read string: stops at space or width limit.
static int read_str(char *dst, int width) {
    int c;  // skip leading spaces
    do {
        c = getc_in();
    } while (is_space(c));
    if (c < 0) return 0;
    int cnt = 0;
    while (c >= 0 && !is_space(c) && (width == 0 || cnt < width - 1)) {
        dst[cnt++] = (char)c;
        c = getc_in();
    }
    dst[cnt] = '\0';
    if (c >= 0) ungetc_in(c);
    return cnt ? 1 : 0;
}
// Skip whitespace in both format and input.
static void skip_ws_fmt_and_input(void) {
    int c;
    do c = getc_in();
    while (is_space(c));
    ungetc_in(c);
}

int scanf(const char *fmt, ...) {
    base_address = (uintptr_t)get_current_chip_baseaddress();
    peek = -1;                  // simple one‑char push‑back
    va_list ap;
    va_start(ap, fmt);
    int assigned = 0;
    while (*fmt) {
        if (is_space(*fmt)) {  // space in format → eat any amount of input ws
            while (is_space(*fmt)) ++fmt;
            skip_ws_fmt_and_input();
            continue;
        }
        if (*fmt != '%') {  // literal char must match
            int c = getc_in();
            if (c != *fmt++) {
                ungetc_in(c);
                break;
            }
            continue;
        }
        // --- conversion specifier ---
        ++fmt;
        int width = 0;
        while (*fmt >= '0' && *fmt <= '9') width = width * 10 + (*fmt++ - '0');
        char sp = *fmt ? *fmt++ : '\0';
        switch (sp) {
            case 'c': {
                int *p = va_arg(ap, int *);
                int c = getc_in();
                if (c < 0) goto end;
                *p = c;
                ++assigned;
            } break;
            case 's': {
                char *p = va_arg(ap, char *);
                if (!read_str(p, width)) goto end;
                ++assigned;
            } break;
            case 'd': {
                int32_t *p = va_arg(ap, int32_t *);
                if (!read_int(p, width)) goto end;
                ++assigned;
            } break;
            case 'u': {
                skip_ws_fmt_and_input();
                uint32_t v;
                if (!read_uint(&v, 10, width)) goto end;
                uint32_t *p = va_arg(ap, uint32_t *);
                *p = v;
                ++assigned;
            } break;
            case 'x':
            case 'X': {
                skip_ws_fmt_and_input();
                if (width) eat_0x_prefix(&width);
                uint32_t v;
                if (!read_uint(&v, 16, width)) goto end;
                uint32_t *p = va_arg(ap, uint32_t *);
                *p = v;
                ++assigned;
            } break;
            case '%': {
                int c = getc_in();
                if (c != '%') {
                    ungetc_in(c);
                    goto end;
                }
            } break;
            default:  // unknown spec – abort
                goto end;
        }
    }
end:
    va_end(ap);
    return assigned;
}
