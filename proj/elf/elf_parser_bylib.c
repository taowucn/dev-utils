#include <elf.h>
#include <fcntl.h>
#include <gelf.h>
#include <libelf.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s <elf-file>\n", prog);
}

int main(int argc, char **argv) {
    Elf *elf = NULL;
    int fd = -1;
    int ret = EXIT_FAILURE;

    if (argc != 2) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    if (elf_version(EV_CURRENT) == EV_NONE) {
        fprintf(stderr, "libelf initialization failed: %s\n", elf_errmsg(-1));
        return EXIT_FAILURE;
    }

    fd = open(argv[1], O_RDONLY);
    if (fd < 0) {
        perror("open");
        return EXIT_FAILURE;
    }

    elf = elf_begin(fd, ELF_C_READ, NULL);
    if (elf == NULL) {
        fprintf(stderr, "elf_begin failed: %s\n", elf_errmsg(-1));
        goto out;
    }

    if (elf_kind(elf) != ELF_K_ELF) {
        fprintf(stderr, "%s is not an ELF file\n", argv[1]);
        goto out;
    }

    GElf_Ehdr ehdr;
    if (gelf_getehdr(elf, &ehdr) == NULL) {
        fprintf(stderr, "gelf_getehdr failed: %s\n", elf_errmsg(-1));
        goto out;
    }

    printf("ELF class: %s\n", gelf_getclass(elf) == ELFCLASS64 ? "ELF64" : "ELF32");
    printf("Data encoding: %s\n", ehdr.e_ident[EI_DATA] == ELFDATA2LSB ? "little-endian" : "big-endian");
    printf("Entry point: 0x%llx\n", (unsigned long long)ehdr.e_entry);
    printf("Section header offset: 0x%llx\n", (unsigned long long)ehdr.e_shoff);
    printf("Number of section headers: %u\n", ehdr.e_shnum);

    size_t shstrndx = 0;
    if (elf_getshdrstrndx(elf, &shstrndx) != 0) {
        fprintf(stderr, "elf_getshdrstrndx failed: %s\n", elf_errmsg(-1));
        goto out;
    }

    printf("\nSections:\n");
    Elf_Scn *scn = NULL;
    size_t index = 0;
    while ((scn = elf_nextscn(elf, scn)) != NULL) {
        GElf_Shdr shdr;
        if (gelf_getshdr(scn, &shdr) == NULL) {
            fprintf(stderr, "gelf_getshdr failed: %s\n", elf_errmsg(-1));
            goto out;
        }

        char *name = elf_strptr(elf, shstrndx, shdr.sh_name);
        printf("[%zu] %s\n", index, name != NULL ? name : "<null>");
        index++;
    }

    printf("\nSymbols:\n");
    scn = NULL;
    while ((scn = elf_nextscn(elf, scn)) != NULL) {
        GElf_Shdr shdr;
        if (gelf_getshdr(scn, &shdr) == NULL) {
            continue;
        }

        if (shdr.sh_type != SHT_SYMTAB) {
            continue;
        }

        Elf_Data *data = elf_getdata(scn, NULL);
        if (data == NULL) {
            continue;
        }

        size_t count = shdr.sh_size / shdr.sh_entsize;
        for (size_t i = 0; i < count; ++i) {
            GElf_Sym sym;
            if (gelf_getsym(data, i, &sym) == NULL) {
                fprintf(stderr, "gelf_getsym failed: %s\n", elf_errmsg(-1));
                goto out;
            }

            char *name = elf_strptr(elf, shdr.sh_link, sym.st_name);
            if (name == NULL || name[0] == '\0') {
                continue;
            }
            printf("[%zu] %s\n", i, name);
        }
        break;
    }

    ret = EXIT_SUCCESS;

out:
    if (elf != NULL) {
        elf_end(elf);
    }
    if (fd >= 0) {
        close(fd);
    }
    return ret;
}
