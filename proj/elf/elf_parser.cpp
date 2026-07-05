#include <elf.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<unsigned char> read_file(const std::string &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open file: " + path);
    }

    std::vector<unsigned char> data((std::istreambuf_iterator<char>(input)),
                                    std::istreambuf_iterator<char>());
    return data;
}

const char *section_name(const std::vector<unsigned char> &data, const Elf64_Shdr &shdr,
                         const Elf64_Shdr &shstrtab) {
    const char *strtab = reinterpret_cast<const char *>(data.data() + shstrtab.sh_offset);
    return strtab + shdr.sh_name;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <elf-file>\n";
        return 1;
    }

    try {
        std::vector<unsigned char> data = read_file(argv[1]);
        if (data.size() < sizeof(Elf64_Ehdr)) {
            throw std::runtime_error("file is too small to be an ELF file");
        }

        const Elf64_Ehdr *ehdr = reinterpret_cast<const Elf64_Ehdr *>(data.data());
        if (memcmp(ehdr->e_ident, ELFMAG, SELFMAG) != 0) {
            throw std::runtime_error("not an ELF file");
        }

        std::cout << "ELF class: " << (ehdr->e_ident[EI_CLASS] == ELFCLASS64 ? "ELF64" : "ELF32") << "\n";
        std::cout << "Data encoding: " << (ehdr->e_ident[EI_DATA] == ELFDATA2LSB ? "little-endian" : "big-endian") << "\n";
        std::cout << "Entry point: 0x" << std::hex << ehdr->e_entry << std::dec << "\n";
        std::cout << "Section header offset: 0x" << std::hex << ehdr->e_shoff << std::dec << "\n";
        std::cout << "Number of section headers: " << ehdr->e_shnum << "\n";

        if (ehdr->e_shoff == 0 || ehdr->e_shnum == 0) {
            std::cout << "No section headers found\n";
            return 0;
        }

        const Elf64_Shdr *shdr = reinterpret_cast<const Elf64_Shdr *>(data.data() + ehdr->e_shoff);
        const Elf64_Shdr &shstrtab = shdr[ehdr->e_shstrndx];

        std::cout << "\nSections:\n";
        for (unsigned int i = 0; i < ehdr->e_shnum; ++i) {
            std::cout << "[" << i << "] " << section_name(data, shdr[i], shstrtab) << "\n";
        }

        std::cout << "\nSymbols:\n";
        for (unsigned int i = 0; i < ehdr->e_shnum; ++i) {
            if (shdr[i].sh_type != SHT_SYMTAB) {
                continue;
            }

            const Elf64_Sym *symtab = reinterpret_cast<const Elf64_Sym *>(data.data() + shdr[i].sh_offset);
            std::size_t count = shdr[i].sh_size / shdr[i].sh_entsize;
            const Elf64_Shdr &strtab = shdr[shdr[i].sh_link];
            const char *symstrtab = reinterpret_cast<const char *>(data.data() + strtab.sh_offset);

            for (std::size_t j = 0; j < count; ++j) {
                if (symtab[j].st_name == 0) {
                    continue;
                }
                std::cout << j << ": " << (symstrtab + symtab[j].st_name) << "\n";
            }
            break;
        }
    } catch (const std::exception &ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }

    return 0;
}
