#include <elf.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s <elf-file>\n", prog);
}

static void parse_elf(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        perror("open");
        exit(EXIT_FAILURE);
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        perror("fstat");
        close(fd);
        exit(EXIT_FAILURE);
    }

    unsigned char *data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) {
        perror("mmap");
        close(fd);
        exit(EXIT_FAILURE);
    }

    if (st.st_size < (off_t)sizeof(Elf64_Ehdr)) {
        fprintf(stderr, "File too small to be an ELF file\n");
        munmap(data, st.st_size);
        close(fd);
        exit(EXIT_FAILURE);
    }

    Elf64_Ehdr *ehdr = (Elf64_Ehdr *)data;
    if (memcmp(ehdr->e_ident, ELFMAG, SELFMAG) != 0) {
        fprintf(stderr, "Not an ELF file\n");
        munmap(data, st.st_size);
        close(fd);
        exit(EXIT_FAILURE);
    }

    printf("ELF class: %s\n", ehdr->e_ident[EI_CLASS] == ELFCLASS64 ? "ELF64" : "ELF32");
    printf("Data encoding: %s\n", ehdr->e_ident[EI_DATA] == ELFDATA2LSB ? "little-endian" : "big-endian");
    printf("Entry point: 0x%lx\n", (unsigned long)ehdr->e_entry);
    printf("Section header offset: 0x%lx\n", (unsigned long)ehdr->e_shoff);
    printf("Number of section headers: %u\n", ehdr->e_shnum);

    if (ehdr->e_shoff == 0 || ehdr->e_shnum == 0) {
        printf("No section headers found\n");
        munmap(data, st.st_size);
        close(fd);
        return;
    }

    Elf64_Shdr *shdr = (Elf64_Shdr *)(data + ehdr->e_shoff);
    const char *shstrtab = (const char *)(data + shdr[ehdr->e_shstrndx].sh_offset);

    printf("\nSections:\n");
    for (unsigned int i = 0; i < ehdr->e_shnum; ++i) {
        const char *name = shstrtab + shdr[i].sh_name;
        printf("[%u] %s\n", i, name);
    }

    printf("\nSymbols:\n");
    for (unsigned int i = 0; i < ehdr->e_shnum; ++i) {
        if (shdr[i].sh_type != SHT_SYMTAB) {
            continue;
        }

        Elf64_Sym *symtab = (Elf64_Sym *)(data + shdr[i].sh_offset);
        size_t count = shdr[i].sh_size / shdr[i].sh_entsize;
        for (size_t j = 0; j < count; ++j) {
            const char *name = (const char *)(data + shdr[shdr[i].sh_link].sh_offset + symtab[j].st_name);
            if (name[0] == '\0') {
                continue;
            }
            printf("%zu: %s\n", j, name);
        }
        break;
    }

    munmap(data, st.st_size);
    close(fd);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    parse_elf(argv[1]);
    return EXIT_SUCCESS;
}
