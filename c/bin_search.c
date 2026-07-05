#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>

static long get_file_size(const char *path)
{
	struct stat statbuff;

	if (stat(path, &statbuff) < 0) {
		perror(path);
		return 0;
	} else {
		return statbuff.st_size;
	}
}

int main(int argc, char *argv[])
{
	const char *in_filename;
	void *virt_addr = NULL;
	uint8_t *virt_u8 = 0;
	uint16_t *virt_u16 = 0;
	uint32_t *virt_u32 = 0;
	uint64_t *virt_u64 = 0;
	uint8_t val_u8;
	uint16_t val_u16;
	uint32_t val_u32;
	uint64_t val_u64;

	long fsize = 0;
	long done_size = 0, done_num = 0;
	long elem_value = 0;
	long elem_size = 0, elem_num = 0;
	long elem_bitmask = 0, bitno = 0;
	int fd = -1;
	int rval = 0;

	if ((argc != 4) && (argc != 5)) {
		printf("Usage: %s <input_file> <element_size> <element_value> [bitno]\n", argv[0]);
		printf("       input_file is treated as binary, search binary with (1/2/4/8) byte per element. "
			"Return element_offset, byte_offset and bit_offset\n");

		printf("\nExample: \n");
		printf("    Search bin 0xff with 4 byte per element: \n");
		printf("    # %s input.bin 4 0xff\n\n", argv[0]);
		return -1;
	}
	in_filename = argv[1];
	elem_size = strtoul(argv[2], NULL, 0);
	elem_value = strtoul(argv[3], NULL, 0);
	if (argc == 5) {
		bitno = strtoul(argv[4], NULL, 0);
	}

	do {
		fd = open(in_filename, O_RDONLY);
		if (fd < 0) {
			perror(in_filename);
			rval = -1;
			break;
		}
		fsize = get_file_size(in_filename);
		if (fsize < 0) {
			perror(in_filename);
			rval = -1;
			break;
		}

		virt_addr = mmap(NULL, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
		if (virt_addr == MAP_FAILED) {
			perror(in_filename);
			rval = -1;
			break;
		}

		switch (elem_size) {
		case 1:
			virt_u8 = (uint8_t *)virt_addr;
			val_u8 = elem_value;
			elem_bitmask = bitno ? ((1UL << bitno) - 1) : 0xff;
			printf("Input file: %s, one elem_size: %lu, elem_value: %lu (0x%lx), bitno: %lu, bitmask: 0x%lx\n",
				in_filename, elem_size, elem_value, elem_value, bitno, elem_bitmask);
			while (done_size < fsize) {
				if (((*virt_u8) & elem_bitmask) == val_u8) {
					printf("Found U8 Value: %u (0x%x) in file data (0x%x & 0x%lx) at offset in [Element/Byte/Bit]:  "
						"[%lu, %lu, %lu]  Hex: [0x%lx, 0x%lx, 0x%lx] \n",
						val_u8, val_u8, *virt_u8, elem_bitmask,
						done_size / elem_size, done_size, done_size << 3,
						done_size / elem_size, done_size, done_size << 3);
				}
				virt_u8++;
				done_size += elem_size;
			}
			break;
		case 2:
			virt_u16 = (uint16_t *)virt_addr;
			val_u16 = elem_value;
			elem_bitmask = bitno ? ((1UL << bitno) - 1) : 0xffff;
			printf("Input file: %s, one elem_size: %lu, elem_value: %lu (0x%lx), bitno: %lu, bitmask: 0x%lx\n",
				in_filename, elem_size, elem_value, elem_value, bitno, elem_bitmask);
			while (done_size < fsize) {
				if (((*virt_u16) & elem_bitmask) == val_u16) {
					printf("Found U16 Value: %u (0x%x) in file data (0x%x & 0x%lx) at offset in [Element/Byte/Bit]:  "
						"[%lu, %lu, %lu]  Hex: [0x%lx, 0x%lx, 0x%lx] \n",
						val_u16, val_u16, *virt_u16, elem_bitmask,
						done_size / elem_size, done_size, done_size << 3,
						done_size / elem_size, done_size, done_size << 3);
				}
				virt_u16++;
				done_size += elem_size;
			}
			break;
		case 4:
			virt_u32 = (uint32_t *)virt_addr;
			val_u32 = elem_value;
			elem_bitmask = bitno ? ((1UL << bitno) - 1) : 0xffffffff;
			printf("Input file: %s, one elem_size: %lu, elem_value: %lu (0x%lx), bitno: %lu, bitmask: 0x%lx\n",
				in_filename, elem_size, elem_value, elem_value, bitno, elem_bitmask);
			while (done_size < fsize) {
				if (((*virt_u32) & elem_bitmask) == val_u32) {
					printf("Found U32 Value: %u (0x%x) in file data (0x%x & 0x%lx) at offset in [Element/Byte/Bit]:  "
						"[%lu, %lu, %lu]  Hex: [0x%lx, 0x%lx, 0x%lx] \n",
						val_u32, val_u32, *virt_u32, elem_bitmask,
						done_size / elem_size, done_size, done_size << 3,
						done_size / elem_size, done_size, done_size << 3);
				}
				virt_u32++;
				done_size += elem_size;
			}
			break;
		case 8:
			virt_u64 = (uint64_t *)virt_addr;
			val_u64 = elem_value;
			elem_bitmask = bitno ? ((1UL << bitno) -1) : 0xffffffffffffffff;
			printf("Input file: %s, one elem_size: %lu, elem_value: %lu (0x%lx), bitno: %lu, bitmask: 0x%lx\n",
				in_filename, elem_size, elem_value, elem_value, bitno, elem_bitmask);
			while (done_size < fsize) {
				if (((*virt_u64) & elem_bitmask) == val_u64) {
					printf("Found U64 Value: %lu (0x%lx) in file data (0x%lx & 0x%lx) at offset in [Element/Byte/Bit]:  "
						"[%lu, %lu, %lu]  Hex: [0x%lx, 0x%lx, 0x%lx] \n",
						val_u64, val_u64, *virt_u64, elem_bitmask,
						done_size / elem_size, done_size, done_size << 3,
						done_size / elem_size, done_size, done_size << 3);
				}
				virt_u64++;
				done_size += elem_size;
			}
			break;
		default:
			printf("Unsupport byte size per element: %lu\n", elem_size);
			rval = -1;
			break;
		}
	} while (0);

	if (virt_addr) {
		munmap(virt_addr, fsize);
		virt_addr = NULL;
	}
	if (fd >= 0 ) {
		close(fd);
		fd = -1;
	}

	return rval;
}
