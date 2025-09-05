#include "hemaia_d2d_link.h"

void hemaia_d2d_link_require_test_one_direction(Direction direction,
                                                uint32_t cycles) {
  while (get_d2d_link_being_tested(direction)) {
    // Wait until the link is not being tested
  }
  set_d2d_link_test_mode(direction, true);
  delay_cycles(cycles);
  set_d2d_link_test_mode(direction, false);
}

void hemaia_d2d_link_require_test_all_directions(uint32_t cycles) {
  while (get_d2d_link_being_tested(D2D_DIRECTION_EAST) ||
         get_d2d_link_being_tested(D2D_DIRECTION_WEST) ||
         get_d2d_link_being_tested(D2D_DIRECTION_NORTH) ||
         get_d2d_link_being_tested(D2D_DIRECTION_SOUTH)) {
    // Wait until the link is not being tested
  }
  set_d2d_link_test_mode(D2D_DIRECTION_EAST, true);
  set_d2d_link_test_mode(D2D_DIRECTION_WEST, true);
  set_d2d_link_test_mode(D2D_DIRECTION_NORTH, true);
  set_d2d_link_test_mode(D2D_DIRECTION_SOUTH, true);
  delay_cycles(cycles);
  set_d2d_link_test_mode(D2D_DIRECTION_EAST, false);
  set_d2d_link_test_mode(D2D_DIRECTION_WEST, false);
  set_d2d_link_test_mode(D2D_DIRECTION_NORTH, false);
  set_d2d_link_test_mode(D2D_DIRECTION_SOUTH, false);
}

// Helpers to get the error cycle of one channel
// For uint8_t *array

inline uint32_t get_non_zero_element_index_u8(uint8_t *array,
                                              uint32_t start_index,
                                              uint32_t size) {
  for (uint32_t i = start_index; i < size; ++i) {
    if (array[i] != 0) {
      return i;
    }
  }
  return size;
}

inline uint32_t get_zero_element_index_u8(uint8_t *array, uint32_t start_index,
                                          uint32_t size) {
  for (uint32_t i = start_index; i < size; ++i) {
    if (array[i] == 0) {
      return i;
    }
  }
  return size;
}

inline uint32_t get_smallest_element_index_u8(uint8_t *array, uint32_t size) {
  uint32_t min_index = 0;
  for (uint32_t i = 1; i < size; ++i) {
    if (array[i] < array[min_index]) {
      min_index = i;
    }
  }
  return min_index;
}

inline uint32_t get_array_sum_u8(uint8_t *array, uint32_t size) {
  uint32_t sum = 0;
  for (uint32_t i = 0; i < size; ++i) {
    sum += array[i];
  }
  return sum;
}

// For uint32_t *array

inline uint32_t get_non_zero_element_index_u32(uint32_t *array,
                                               uint32_t start_index,
                                               uint32_t size) {
  for (uint32_t i = start_index; i < size; ++i) {
    if (array[i] != 0) {
      return i;
    }
  }
  return size;
}

inline uint32_t get_zero_element_index_u32(uint32_t *array,
                                           uint32_t start_index,
                                           uint32_t size) {
  for (uint32_t i = start_index; i < size; ++i) {
    if (array[i] == 0) {
      return i;
    }
  }
  return size;
}

inline uint32_t get_smallest_element_index_u32(uint32_t *array, uint32_t size) {
  uint32_t min_index = 0;
  for (uint32_t i = 1; i < size; ++i) {
    if (array[i] < array[min_index]) {
      min_index = i;
    }
  }
  return min_index;
}

inline uint32_t get_array_sum_u32(uint32_t *array, uint32_t size) {
  uint32_t sum = 0;
  for (uint32_t i = 0; i < size; ++i) {
    sum += array[i];
  }
  return sum;
}

// The function to set the delay for each channel in a direction
void hemaia_d2d_link_set_delay(Direction direction) {
  uint32_t ber[CHANNELS_PER_DIRECTION][HEMAIA_D2D_LINK_NUM_DELAYS] = {0};
  for (uint8_t i = 0; i < HEMAIA_D2D_LINK_NUM_DELAYS; i++) {
    // Set the delay and require one test
    set_d2d_link_clock_delay_all_channels(direction, i);
    asm volatile("fence" : : : "memory");
    hemaia_d2d_link_require_test_one_direction(
        direction, HEMAIA_D2D_LINK_DEFAULT_TEST_CYCLES);

    // Calculate the BER Index for each channels
    for (uint8_t j = 0; j < CHANNELS_PER_DIRECTION; j++) {
      uint8_t ber_counter_result[HEMAIA_D2D_LINK_BROKEN_LINK_REG_SIZE] = {0};
      get_d2d_link_error_cycle_one_channel(direction, j, ber_counter_result);
      uint64_t total_bit_error = (uint64_t)(get_array_sum_u8(
          ber_counter_result, HEMAIA_D2D_LINK_BROKEN_LINK_REG_SIZE));
      // Exclude the total bit error of the channel that is being bypassed
      total_bit_error -= get_d2d_link_error_cycle_one_wire(
          direction, j, get_d2d_link_broken_link(direction, j));
      uint32_t total_cycle = get_d2d_link_tested_cycle(direction, j);

      if (total_cycle > 0) {
        // When total cycle is larger than 0, it means that the
        // pseudorandom code is successfully locked, so that calculating
        // ber index is meaningful
        ber[j][i] =
            (total_bit_error << 32) / get_d2d_link_tested_cycle(direction, j);
      } else {
        // Otherwise, the corresponding ber is set to maximal value in
        // uint32_t
        ber[j][i] = 0xFFFFFFFF;
      }
    }
  }

  // Followed by getting the BER index, this function will set the optimal
  // delays for each channels
  for (uint8_t i = 0; i < CHANNELS_PER_DIRECTION; i++) {
    // First, take a look at whether there is a zero
    uint32_t first_zero_element_index =
        get_zero_element_index_u32(ber[i], 0, HEMAIA_D2D_LINK_NUM_DELAYS);

    if (first_zero_element_index == HEMAIA_D2D_LINK_NUM_DELAYS) {
      // There is no zeros, so possibly there is a broken link
      uint32_t smallest_element_index =
          get_smallest_element_index_u32(ber[i], HEMAIA_D2D_LINK_NUM_DELAYS);
      // Set the delay to the smallest element
      set_d2d_link_clock_delay(direction, i, smallest_element_index);
      // Since there is no zero, it means that the link is still not
      // available
      set_d2d_link_availability(direction, false);
    } else {
      // There is one or multiple zeros, so find the beginning and end of
      // the zeros and set the delay to the middle of the zeros
      uint32_t second_zero_element_index = first_zero_element_index;
      while (second_zero_element_index < HEMAIA_D2D_LINK_NUM_DELAYS) {
        uint32_t temp_index = get_zero_element_index_u32(
            ber[i], second_zero_element_index + 1, HEMAIA_D2D_LINK_NUM_DELAYS);
        if (temp_index != second_zero_element_index + 1)
          break;
        else
          second_zero_element_index = temp_index;
      }
      // Set the delay to the middle of the zeros
      set_d2d_link_clock_delay(
          direction, i,
          (first_zero_element_index + second_zero_element_index) / 2);

      // Since there is a zero, it means that the link is available from
      // this moment on
      set_d2d_link_availability(direction, true);
    }
  }
}

// The function to set the broken link for one channel in a direction
int32_t hemaia_d2d_link_set_bypass_link(Direction direction, uint8_t channel) {
  uint8_t ber_counter_result[HEMAIA_D2D_LINK_BROKEN_LINK_REG_SIZE] = {0};
  get_d2d_link_error_cycle_one_channel(direction, channel, ber_counter_result);
  uint32_t index = get_non_zero_element_index_u8(
      ber_counter_result, 0, HEMAIA_D2D_LINK_BROKEN_LINK_REG_SIZE);
  if (index < HEMAIA_D2D_LINK_BROKEN_LINK_REG_SIZE) {
    set_d2d_link_broken_link(direction, channel, index);
    // There is one zero, so try to find another zero
    index = get_non_zero_element_index_u8(ber_counter_result, index,
                                          HEMAIA_D2D_LINK_BROKEN_LINK_REG_SIZE);
    if (index == HEMAIA_D2D_LINK_BROKEN_LINK_REG_SIZE) {
      // There is no second zero, so normal return. The bypassed link is
      // returned
      return index;
    } else
      // There is second zero, so more than one wire is broken.
      return -1;
  }
}
