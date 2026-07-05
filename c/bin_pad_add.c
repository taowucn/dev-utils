#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>

static uint32_t get_file_size(const char *path)
{
	struct stat statbuff;

	if (stat(path, &statbuff) < 0) {
		return 0;
	} else {
		return statbuff.st_size;
	}
}

static int check_file_size(uint32_t in_size, uint32_t algin_sz)
{
	if ((in_size % algin_sz) == 0) {
		return 0;
	} else {
		return -1;
	}
}

int main(int argc, char *argv[])
{
	//struct stat in_state;
	FILE *in_fp = NULL;
	FILE *out_fp = NULL;
	char *in_file = NULL;
	char *out_file = NULL;
	char *data = NULL, *data_pad = NULL;
	uint32_t in_size = 0;
	//uint32_t out_size = 0;
	uint32_t read_sz = 0;
	uint32_t pad_sz = 0;
	int rval = 0;

	if (argc != 5) {
		printf("Usage: %s <Input filename> <Output filename> <read size> <pad size>\n", argv[0]);
		printf("\nNOTE: adding pad data is zero\n");
		return -1;
	}

	in_file = argv[1];
	out_file = argv[2];
	read_sz = atoi(argv[3]);
	pad_sz = atoi(argv[4]);

	in_size = get_file_size(in_file);
	if (in_size == 0) {
		perror("state");
		return -1;
	}
	if (check_file_size(in_size, read_sz) < 0) {
		printf("Check file size err: total size: %u, read size: %u\n", in_size, read_sz);
		return -1;
	}

	data = (char *)malloc(read_sz);
	if (data == NULL) {
		perror("malloc");
		return -1;
	}

	data_pad = (char *)malloc(pad_sz);
	if (data_pad == NULL) {
		perror("malloc");
		return -1;
	}
	memset(data_pad, 0, pad_sz);

	do {
		in_fp = fopen(in_file, "rb");
		if (in_fp == NULL) {
			perror("fopen");
			rval = -1;
			break;
		}

		out_fp = fopen(out_file, "w+b");
		if (out_fp == NULL) {
			perror("fopen");
			rval = -1;
			break;
		}
		while (in_size > 0) {
			if (fread(data, 1, read_sz, in_fp) != read_sz) {
				perror("fread in");
				rval = -1;
				break;
			}
			if (fwrite(data, 1, read_sz, out_fp) != read_sz) {
				perror("fwrite out");
				rval = -1;
				break;
			}
			if (fwrite(data_pad, 1, pad_sz, out_fp) != pad_sz) {
				perror("fwrite pad");
				rval = -1;
				break;
			}
			in_size -= read_sz;
		}
	}while (0);

	if (out_fp != NULL) {
		fclose(out_fp);
		out_fp = NULL;
	}
	if (in_fp != NULL) {
		fclose(in_fp);
		in_fp = NULL;
	}
	if (data_pad != NULL) {
		free(data_pad);
		data_pad = NULL;
	}
	if (data != NULL) {
		free(data);
		data = NULL;
	}

	if (rval == 0) {
		printf("Write %s done\n", out_file);
	}

	return 0;
}
