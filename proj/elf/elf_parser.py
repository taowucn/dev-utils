#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys

def parse_elf_by_cmd(elf_file: str) -> None:
    header_cmd = ["readelf", "-h", elf_file]
    symbol_cmd = ["readelf", "-s", elf_file]
    program_cmd = ["readelf", "-l", elf_file]

    header = subprocess.run(header_cmd, capture_output=True, text=True, check=True)
    symbols = subprocess.run(symbol_cmd, capture_output=True, text=True, check=True)
    programs = subprocess.run(program_cmd, capture_output=True, text=True, check=True)

    print("ELF header:")
    for line in header.stdout.splitlines():
        if any(keyword in line for keyword in ["Class:", "Data:", "Type:", "Machine:", "Entry point address:"]):
            print(line.strip())

    print("\nRelevant symbols:")
    for line in symbols.stdout.splitlines():
        if "sum" in line or "main" in line:
            print(line.strip())

    print("\nProgram headers:")
    in_headers = False
    for line in programs.stdout.splitlines():
        if line.startswith("Program Headers:"):
            in_headers = True
            continue
        if in_headers and line.startswith(" Section to Segment mapping:"):
            break
        if in_headers:
            print(line.rstrip())

    print("\nSection summary:")
    for line in programs.stdout.splitlines():
        if line.strip().startswith("[") and ".text" in line or ".data" in line or ".bss" in line or ".rodata" in line:
            print(line.rstrip())


def parse_elf_by_lib(elf_file: str) -> None:
    try:
        from elftools.elf.elffile import ELFFile
    except ImportError:
        print("pyelftools is not installed; falling back to readelf output.")
        parse_elf_by_cmd(elf_file)
        return

    with open(elf_file, 'rb') as f:
        elf = ELFFile(f)

        print("ELF header:")
        print(f"  Class: {elf['e_ident']['EI_CLASS']}")
        print(f"  Data: {elf['e_ident']['EI_DATA']}")
        print(f"  Type: {elf['e_type']}")
        print(f"  Machine: {elf['e_machine']}")
        print(f"  Entry point address: {hex(elf['e_entry'])}")

        print("\nSymbols:")
        for section in elf.iter_sections():
            if section.header['sh_type'] == 'SHT_SYMTAB':
                for symbol in section.iter_symbols():
                    if symbol['st_size']:
                        print(f"  {symbol.name}: {hex(symbol['st_value'])}, size: {symbol['st_size']}")

        print("\nSection summary:")
        for section in elf.iter_sections():
            if section.name in [".text", ".data", ".rodata", ".bss", ".symtab", ".strtab"]:
                print(
                    f"  {section.name}: size=0x{section.header['sh_size']:x}, "
                    f"type={section.header['sh_type']}, "
                    f"addr=0x{section.header['sh_addr']:x}"
                )

        print("\nProgram headers:")
        for idx, program in enumerate(elf.iter_segments()):
            print(
                f"  [{idx}] type={program.header['p_type']}, "
                f"offset=0x{program.header['p_offset']:x}, "
                f"vaddr=0x{program.header['p_vaddr']:x}, "
                f"paddr=0x{program.header['p_paddr']:x}, "
                f"filesz=0x{program.header['p_filesz']:x}, "
                f"memsz=0x{program.header['p_memsz']:x}"
            )

def parse_elf(elf_file: str, elf_meth: int) -> None:
    if not os.path.exists(elf_file):
        raise FileNotFoundError(f"ELF file not found: {elf_file}")

    print(f"ELF file: {elf_file}")
    print("=" * 60)

    if elf_meth == 0:
        parse_elf_by_cmd(elf_file)
    elif elf_meth == 1:
        parse_elf_by_lib(elf_file)
    else:
        raise ValueError(f"Invalid method: {elf_meth}. Use 0 for command-line or 1 for library.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse an ELF file using readelf")
    parser.add_argument("-f", "--file", required=True, 
        help="Path to the ELF file")
    parser.add_argument("-m", "--method", required=True, type=int, 
        help="Method to use for parsing: 0: cmd; 1: library", choices=[0, 1])
    args = parser.parse_args()
    
    parse_elf(args.file, args.method)
