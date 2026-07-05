#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>

static uint32_t get_file_size(const char *path)
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
	FILE *left_fp = NULL;
	FILE *right_fp = NULL;
	char *left_file = NULL;
	char *right_file = NULL;
	uint32_t left_size = 0;
	uint32_t right_size = 0;
	uint32_t diff_count = 0, max_diff_cnt = 0, do_count = 0;
	uint8_t left_data, right_data;
	int rval = 0;

	if ((argc != 4) && ((argc != 3))) {
		printf("Usage: %s <Left filename> <Right filename>\n", argv[0]);
		printf("           %s <Left filename> <Right filename> <Max different byte size>\n", argv[0]);
		return -1;
	}

	left_file = argv[1];
	right_file = argv[2];
	left_size = get_file_size(left_file);
	right_size = get_file_size(right_file);

	if (argc == 4) {
		max_diff_cnt = atoi(argv[3]);
	} else {
		max_diff_cnt = left_size;
	}

	if ((left_size == 0) || (right_size == 0)) {
		printf("File size is zero, %s: %u,  %s: %u\n", left_file, left_size, right_file, right_size);
		return -1;
	}
	if (left_size != right_size) {
		printf("File size mismatch, %s: %u,  %s: %u\n", left_file, left_size, right_file, right_size);
		return -1;
	}

	do {
		left_fp = fopen(left_file, "rb");
		if (left_fp == NULL) {
			perror("fopen");
			rval = -1;
			break;
		}

		right_fp = fopen(right_file, "rb");
		if (right_fp == NULL) {
			perror("fopen");
			rval = -1;
			break;
		}

		while ((diff_count < max_diff_cnt) && (do_count++ < left_size)) {
			if (fread(&left_data, 1, 1, left_fp) != 1) {
				perror("fread");
				rval = -1;
				break;
			}
			if (fread(&right_data, 1, 1, right_fp) != 1) {
				perror("fread");
				rval = -1;
				break;
			}
			if (left_data != right_data) {
				printf("[%08u] byte_offset at: 0x%08x, expect: 0x%02x, real: 0x%02x\n",
					diff_count, do_count, left_data, right_data);
				diff_count++;
			}
		}
	}while (0);

	if (right_fp != NULL) {
		fclose(right_fp);
		right_fp = NULL;
	}
	if (left_fp != NULL) {
		fclose(left_fp);
		left_fp = NULL;
	}

	if (diff_count == 0) {
		printf("Same: %s = %s\n", left_file, right_file);
	} else {
		printf("Different: %s != %s\n", left_file, right_file);
	}

	return rval;
}

