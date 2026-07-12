#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>

#ifndef ROUND_UP
#define ROUND_UP(size, align) (((size) + ((align) - 1)) & ~((align) - 1))
#endif

int main(int argc, char *argv[])
{
	uint32_t in_sz = 0;
	uint32_t align_sz = 0;
	uint32_t out_sz = 0;

	if (argc != 3) {
		printf("Usage: %s <input_size> <align_size>\n", argv[0]);
		printf("Return round up size according input and align rule\n");
		return -1;
	}

	in_sz = atoi(argv[1]);
	align_sz = atoi(argv[2]);
	out_sz = ROUND_UP(in_sz, align_sz);
	printf("ROUND_UP (%u, %u) = %u\n", in_sz, align_sz, out_sz);

	return 0;
}
