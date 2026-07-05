#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>

#define BITSIZE_MASK(_BSIZE_) ((1U << (_BSIZE_)) -1)

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
	void *virt_addr = NULL, *ptr = NULL;
	uint8_t *virt_u8 = 0;
	uint16_t *virt_u16 = 0;
	uint32_t *virt_u32 = 0;
	uint64_t *virt_u64 = 0;
	uint8_t val_u8;
	uint32_t bit_value = 0;
	uint16_t val_u16;
	uint32_t val_u32;
	uint64_t val_u64;

	long fsize = 0;
	long done_size = 0, done_num = 0;
	long byte_offset = 0, bit_offset = 0;
	long word_offset = 0;
	long elem_bitsize = 0, elem_num = 0;
	long elem_bitmask = 0;
	long remain_bit_cnt = 0;
	long write_value = 0, bit_write_val = 0, bitmask = 0;
	long map_attr = 0, map_right = 0;
	int is_write_flag = 0;

	int fd = -1;
	int rval = 0;

	if ((argc != 4) && (argc != 5)) {
		printf("Usage: %s <input_file> <bit_offset> <element_bitsize> <write_value>\n\n", argv[0]);
		printf("       Read or Write data with (1/1/2/4/8) bitsize per element at bit_offset\n");
		printf("       write_value is optional. If not specified, just read value.\n");

		printf("\nExample: \n");
		printf("    1. Read bin at bit_offset 0x16 with 32 bitsize: \n");
		printf("    # %s data.bin 0x16 32 \n\n", argv[0]);

		printf("    2. Read bin at bit_offset 0x16 with 1 bitsize: \n");
		printf("    # %s data.bin 0x16 1 \n\n", argv[0]);

		printf("    1. Write bin at bit_offset 0x16 with 32 bitsize, write value 0x1234: \n");
		printf("    # %s data.bin 0x16 32 0x1234 \n\n", argv[0]);

		printf("    2. Write bin at bit_offset 0x16 with 1 bitsize, wirte value 1: \n");
		printf("    # %s data.bin 0x16 1 1 \n\n", argv[0]);

		return -1;
	}
	in_filename = argv[1];
	bit_offset = strtoul(argv[2], NULL, 0);
	elem_bitsize = strtoul(argv[3], NULL, 0);
	if (elem_bitsize > 32) {
		if (elem_bitsize != 64) {
			printf("Element bitsize: %ld out of range, [1, 32] or 64 \n", elem_bitsize);
			return -1;
		}
	}
	printf("Input file: %s, elem_bitsize: %lu, bit_offset: %lu\n", in_filename, elem_bitsize, bit_offset);
	if (argc == 5) {
		write_value = strtoul(argv[4], NULL, 0);
		is_write_flag = 1;
		printf("Write value: %lu (0x%lx)\n", write_value, write_value);
	}

	do {
		fd = open(in_filename, (is_write_flag ? O_RDWR : O_RDONLY));
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
		if (byte_offset > fsize) {
			printf("Input file: %s, byte_offset: %lu larger total byte size: %lu\n",
				in_filename, byte_offset, fsize);
			rval = -1;
			break;
		}

		map_right = is_write_flag ? (PROT_READ | PROT_WRITE) : (PROT_READ);
		map_attr = is_write_flag ? MAP_SHARED : MAP_PRIVATE;
		virt_addr = mmap(NULL, fsize, PROT_READ | PROT_WRITE, map_attr, fd, 0);
		if (virt_addr == MAP_FAILED) {
			perror(in_filename);
			rval = -1;
			break;
		}
		byte_offset = bit_offset / 8;
		remain_bit_cnt = bit_offset - (byte_offset * 8);
		printf("Byte_offset: %lu (0x%lx). Offset Position (byte, bit): (%lu, %lu)\n\n",
			byte_offset, byte_offset, byte_offset, remain_bit_cnt);

		if ((elem_bitsize + remain_bit_cnt) <= 8) {
			ptr = virt_addr + byte_offset;
			virt_u8 = (uint8_t *)ptr;
			val_u8 = *virt_u8;

			if (elem_bitsize == 8) {
				bitmask = 0xff;
			} else {
				bitmask = BITSIZE_MASK(elem_bitsize);
			}
			bit_value = (val_u8 >> remain_bit_cnt) & bitmask;
			printf("Read U8 value: %u (0x%x), mask: 0x%lx, bit value: %u (0x%x)\n",
				val_u8, val_u8, bitmask, bit_value, bit_value);

			if (is_write_flag) {
				bitmask = bitmask << remain_bit_cnt;
				bit_write_val = val_u8 & (~bitmask);  // clear
				bit_write_val = bit_write_val | (write_value << remain_bit_cnt); // set
				*virt_u8 = bit_write_val;
				printf("Write value: %u (0x%x)\n", *virt_u8, *virt_u8);
			}
		} else if ((elem_bitsize + remain_bit_cnt) <= 32) {
			ptr = virt_addr + byte_offset;
			virt_u32 = (uint32_t *)ptr;
			val_u32 = *virt_u32;

			if (elem_bitsize == 32) {
				bitmask = 0xffffffff;
			} else {
				bitmask = BITSIZE_MASK(elem_bitsize);
			}
			bit_value = (val_u32 >> remain_bit_cnt) & bitmask;
			printf("Read U32 value: %u (0x%x), mask: 0x%lx, bit value: %u (0x%x)\n",
				val_u32, val_u32, bitmask, bit_value, bit_value);

			if (is_write_flag) {
				bitmask = bitmask << remain_bit_cnt;
				bit_write_val = val_u32 & (~bitmask);  // clear
				bit_write_val = bit_write_val | (write_value << remain_bit_cnt); // set
				*virt_u32 = bit_write_val;
				printf("Write value: %u (0x%x)\n", *virt_u32, *virt_u32);
			}
		} else {
			if ((remain_bit_cnt + elem_bitsize) > 32) {
				printf("remain_bit: %lu + bitsize: %lu > 32 bit\n", remain_bit_cnt, elem_bitsize);
				rval = -1;
			}
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
