#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

static uint64_t reverse_bitmap(uint64_t bitmap, uint64_t bit_no)
{
	uint64_t in_bitmap = bitmap >> 1;
	uint64_t out_bitmap = bitmap & 0x01; // record the lowest bit firstly
	uint64_t i;

	for (i = 1; i < bit_no; i++) {
		out_bitmap <<= 1;
		out_bitmap |= (in_bitmap & 0x01);
		in_bitmap >>= 1;
	}

	return out_bitmap;
}

int main(int argc, char *argv[])
{
	uint64_t in_bitmap, out_bitmap, bit_no;

	if (argc != 3) {
		printf("Usage: %s <in_bitmap> <bit_no> \n\n", argv[0]);
		printf("       Reverse bitmap, bit_no <= 64 \n");
		return -1;
	}

	in_bitmap = strtoul(argv[1], NULL, 0);
	bit_no = strtoul(argv[2], NULL, 0);

	out_bitmap = reverse_bitmap(in_bitmap, bit_no);
	printf("In_bitmap: 0x%lx, no: %lu. Reversed_bitmap: 0x%lx\n", in_bitmap, bit_no, out_bitmap);

	return 0;
}

