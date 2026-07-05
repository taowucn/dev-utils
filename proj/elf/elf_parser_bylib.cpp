#include <elf.h>
#include <fcntl.h>
#include <gelf.h>
#include <libelf.h>
#include <iostream>
#include <string>
#include <unistd.h>

static void print_usage(const char *prog) {
    std::cerr << "Usage: " << prog << " <elf-file>\n";
}

int main(int argc, char **argv) {
    if (argc != 2) {
        print_usage(argv[0]);
        return 1;
    }

    if (elf_version(EV_CURRENT) == EV_NONE) {
        std::cerr << "libelf initialization failed: " << elf_errmsg(-1) << "\n";
        return 1;
    }

    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) {
        std::perror("open");
        return 1;
    }

    Elf *elf = elf_begin(fd, ELF_C_READ, nullptr);
    if (elf == nullptr) {
        std::cerr << "elf_begin failed: " << elf_errmsg(-1) << "\n";
        close(fd);
        return 1;
    }

    if (elf_kind(elf) != ELF_K_ELF) {
        std::cerr << argv[1] << " is not an ELF file\n";
        elf_end(elf);
        close(fd);
        return 1;
    }

    GElf_Ehdr ehdr;
    if (gelf_getehdr(elf, &ehdr) == nullptr) {
        std::cerr << "gelf_getehdr failed: " << elf_errmsg(-1) << "\n";
        elf_end(elf);
        close(fd);
        return 1;
    }

    std::cout << "ELF class: " << (gelf_getclass(elf) == ELFCLASS64 ? "ELF64" : "ELF32") << "\n";
    std::cout << "Data encoding: " << (ehdr.e_ident[EI_DATA] == ELFDATA2LSB ? "little-endian" : "big-endian") << "\n";
    std::cout << "Entry point: 0x" << std::hex << ehdr.e_entry << std::dec << "\n";
    std::cout << "Section header offset: 0x" << std::hex << ehdr.e_shoff << std::dec << "\n";
    std::cout << "Number of section headers: " << ehdr.e_shnum << "\n";

    size_t shstrndx = 0;
    if (elf_getshdrstrndx(elf, &shstrndx) != 0) {
        std::cerr << "elf_getshdrstrndx failed: " << elf_errmsg(-1) << "\n";
        elf_end(elf);
        close(fd);
        return 1;
    }

    std::cout << "\nSections:\n";
    Elf_Scn *scn = nullptr;
    size_t index = 0;
    while ((scn = elf_nextscn(elf, scn)) != nullptr) {
        GElf_Shdr shdr;
        if (gelf_getshdr(scn, &shdr) == nullptr) {
            std::cerr << "gelf_getshdr failed: " << elf_errmsg(-1) << "\n";
            elf_end(elf);
            close(fd);
            return 1;
        }

        const char *name = elf_strptr(elf, shstrndx, shdr.sh_name);
        std::cout << "[" << index << "] " << (name != nullptr ? name : "<null>") << "\n";
        ++index;
    }

    std::cout << "\nSymbols:\n";
    scn = nullptr;
    while ((scn = elf_nextscn(elf, scn)) != nullptr) {
        GElf_Shdr shdr;
        if (gelf_getshdr(scn, &shdr) == nullptr) {
            continue;
        }
        if (shdr.sh_type != SHT_SYMTAB) {
            continue;
        }

        Elf_Data *data = elf_getdata(scn, nullptr);
        if (data == nullptr) {
            continue;
        }

        size_t count = shdr.sh_size / shdr.sh_entsize;
        for (size_t i = 0; i < count; ++i) {
            GElf_Sym sym;
            if (gelf_getsym(data, i, &sym) == nullptr) {
                std::cerr << "gelf_getsym failed: " << elf_errmsg(-1) << "\n";
                elf_end(elf);
                close(fd);
                return 1;
            }

            const char *name = elf_strptr(elf, shdr.sh_link, sym.st_name);
            if (name == nullptr || name[0] == '\0') {
                continue;
            }
            std::cout << "[" << i << "] " << name << "\n";
        }
        break;
    }

    elf_end(elf);
    close(fd);
    return 0;
}
