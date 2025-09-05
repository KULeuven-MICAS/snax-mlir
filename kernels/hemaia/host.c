#include "host.h"
#include "hemaia_clk_rst_controller.h"

void host_main() {
  // Hello from CVA6
  printf("Hello from CVA6!\n");
}

// Global Variables for communication buffer
volatile comm_buffer_t *comm_buffer_ptr = (comm_buffer_t *)0;

int main() {
  // Set clk manager to 1 division for a faster simulation time
  enable_clk_domain(0, 1);
  enable_clk_domain(1, 1);
  // Reset and ungate all quadrants, deisolate
  uintptr_t current_chip_address_prefix =

      (uintptr_t)get_current_chip_baseaddress();
  uint32_t current_chip_id = get_current_chip_id();

  init_uart(current_chip_address_prefix, 32, 1);
  comm_buffer_ptr = (comm_buffer_t *)(((uint64_t)&__narrow_spm_start) |
                                      current_chip_address_prefix);

  reset_and_ungate_quadrants_all(current_chip_id);
  deisolate_all(current_chip_id);
  enable_sw_interrupts();
  comm_buffer_ptr->lock = 0;
  comm_buffer_ptr->chip_id = current_chip_id;
  program_snitches(current_chip_id, comm_buffer_ptr);

  asm volatile("fence.i" ::: "memory");

  // Start Snitches
  wakeup_snitches_cl(current_chip_id);

  // Run own host code:
  host_main();

  // Wait for snitches to be done
  int ret = wait_snitches_done(current_chip_id);

  // Return Snitch exit code
  return 0;
}
