/* Link-only virtual layout, shared by the two documented bare-metal targets.
 * This is deliberately not a board memory map or a bootable firmware image. */
ENTRY(no_std_entry)
SECTIONS
{
  . = 0x10000;
  .text : { KEEP(*(.text.no_std_entry)) *(.text .text.*) }
  .rodata : { *(.rodata .rodata.*) }
  .ARM.extab : { *(.ARM.extab*) }
  .ARM.exidx : { *(.ARM.exidx*) }
  .data : { *(.data .data.*) *(.sdata .sdata.*) }
  .bss (NOLOAD) : { *(.bss .bss.*) *(.sbss .sbss.*) *(COMMON) }
}
