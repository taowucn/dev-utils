/*
 * bin_diff - compare two binary files byte by byte and summarize the result.
 *
 * The whole common area is always scanned, so the summary counts every
 * difference. <Max print lines> only limits how many diff lines are printed.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <sys/stat.h>

#define BUF_SIZE	(64 * 1024)
#define PAIR_TABLE_SIZE	(256 * 256)	/* one slot per (left, right) byte pair */
#define PAIR_INDEX(l, r) (((uint32_t)(l) << 8) | (uint32_t)(r))
#define MAX_PAIR_LINES	32		/* top-N pairs listed in the summary */

struct diff_stat {
	uint64_t bytes_compared;	/* bytes present in both files */
	uint64_t diff_bytes;		/* bytes that differ */
	uint64_t diff_bits;		/* bits that differ */
	uint64_t diff_runs;		/* contiguous runs of differing bytes */
	uint64_t first_offset;		/* offset of the first differing byte */
	uint64_t last_offset;		/* offset of the last differing byte */
	uint64_t printed;		/* diff lines already printed */
	uint64_t max_print;		/* print limit, 0 means unlimited */
	int truncated;			/* print limit has been reached */
	uint64_t *pair_count;		/* occurrences per (left, right) byte pair */
	uint32_t pair_kinds;		/* distinct (left, right) byte pairs seen */
};

static int get_file_size(const char *path, uint64_t *size)
{
	struct stat statbuff;

	if (stat(path, &statbuff) < 0) {
		perror(path);
		return -1;
	}

	*size = (uint64_t)statbuff.st_size;

	return 0;
}

static uint32_t bit_diff_count(uint8_t left_data, uint8_t right_data)
{
	uint8_t bits = (uint8_t)(left_data ^ right_data);
	uint32_t count = 0;

	while (bits != 0) {
		bits &= (uint8_t)(bits - 1);
		count++;
	}

	return count;
}
static void record_diff(struct diff_stat *stat, uint64_t offset,
			uint8_t left_data, uint8_t right_data, int in_run)
{
	if (stat->diff_bytes == 0) {
		stat->first_offset = offset;
	}
	stat->last_offset = offset;
	stat->diff_bytes++;
	stat->diff_bits += bit_diff_count(left_data, right_data);
	if (!in_run) {
		stat->diff_runs++;
	}

	if (stat->pair_count != NULL) {
		uint64_t *slot = &stat->pair_count[PAIR_INDEX(left_data, right_data)];

		if (*slot == 0) {
			stat->pair_kinds++;
		}
		(*slot)++;
	}

	if ((stat->max_print != 0) && (stat->printed >= stat->max_print)) {
		if (!stat->truncated) {
			stat->truncated = 1;
			printf("... more differences are counted but not printed\n");
		}
		return;
	}

	printf("[%08" PRIu64 "] byte_offset at: 0x%08" PRIx64 ", left: 0x%02x, right: 0x%02x\n",
		stat->diff_bytes - 1, offset, left_data, right_data);
	stat->printed++;
}

static int compare_files(const char *left_file, const char *right_file,
			uint64_t common_size, struct diff_stat *stat)
{
	FILE *left_fp = NULL;
	FILE *right_fp = NULL;
	uint8_t *left_buf = NULL;
	uint8_t *right_buf = NULL;
	uint64_t offset = 0;
	int in_run = 0;
	int rval = 0;

	left_buf = malloc(BUF_SIZE);
	right_buf = malloc(BUF_SIZE);
	if ((left_buf == NULL) || (right_buf == NULL)) {
		fprintf(stderr, "Out of memory\n");
		rval = -1;
		goto out;
	}

	left_fp = fopen(left_file, "rb");
	if (left_fp == NULL) {
		perror(left_file);
		rval = -1;
		goto out;
	}

	right_fp = fopen(right_file, "rb");
	if (right_fp == NULL) {
		perror(right_file);
		rval = -1;
		goto out;
	}

	while (offset < common_size) {
		uint64_t remain = common_size - offset;
		size_t want = (remain < BUF_SIZE) ? (size_t)remain : BUF_SIZE;
		size_t i;

		if (fread(left_buf, 1, want, left_fp) != want) {
			fprintf(stderr, "%s: read failed at offset 0x%08" PRIx64 "\n",
				left_file, offset);
			rval = -1;
			goto out;
		}
		if (fread(right_buf, 1, want, right_fp) != want) {
			fprintf(stderr, "%s: read failed at offset 0x%08" PRIx64 "\n",
				right_file, offset);
			rval = -1;
			goto out;
		}

		if (memcmp(left_buf, right_buf, want) != 0) {
			for (i = 0; i < want; i++) {
				if (left_buf[i] != right_buf[i]) {
					record_diff(stat, offset + i, left_buf[i],
						right_buf[i], in_run);
					in_run = 1;
				} else {
					in_run = 0;
				}
			}
		} else {
			in_run = 0;
		}

		offset += want;
		stat->bytes_compared = offset;
	}

out:
	if (right_fp != NULL) {
		fclose(right_fp);
	}
	if (left_fp != NULL) {
		fclose(left_fp);
	}
	free(right_buf);
	free(left_buf);

	return rval;
}
/*
 * List each distinct byte difference and how often it occurs, most frequent
 * first. Selection sort over the used slots only, so the cost is bounded by
 * the number of distinct pairs (<= 65536) rather than the file size.
 */
static void print_pair_table(const struct diff_stat *stat)
{
	uint32_t *index = NULL;
	uint32_t used = 0;
	uint32_t i;
	uint32_t limit;

	if ((stat->pair_count == NULL) || (stat->pair_kinds == 0)) {
		return;
	}

	index = malloc(sizeof(*index) * stat->pair_kinds);
	if (index == NULL) {
		fprintf(stderr, "Out of memory, skip the byte difference table\n");
		return;
	}

	for (i = 0; i < PAIR_TABLE_SIZE; i++) {
		if (stat->pair_count[i] != 0) {
			index[used++] = i;
		}
	}

	limit = (used < MAX_PAIR_LINES) ? used : MAX_PAIR_LINES;
	for (i = 0; i < limit; i++) {
		uint32_t best = i;
		uint32_t j;
		uint32_t tmp;

		for (j = i + 1; j < used; j++) {
			if (stat->pair_count[index[j]] > stat->pair_count[index[best]]) {
				best = j;
			}
		}
		tmp = index[i];
		index[i] = index[best];
		index[best] = tmp;
	}

	printf("---------- Byte difference count ----------\n");
	printf("  left -> right :      count   bits  percent\n");
	for (i = 0; i < limit; i++) {
		uint8_t left_data = (uint8_t)(index[i] >> 8);
		uint8_t right_data = (uint8_t)(index[i] & 0xff);
		uint64_t count = stat->pair_count[index[i]];
		double percent = (double)count * 100.0 / (double)stat->diff_bytes;

		printf("  0x%02x -> 0x%02x : %10" PRIu64 "   %4" PRIu32 "  %6.2f%%\n",
			left_data, right_data, count,
			bit_diff_count(left_data, right_data), percent);
	}
	if (used > limit) {
		printf("  ... %" PRIu32 " more kinds not listed\n", used - limit);
	}

	free(index);
}

static void print_summary(const char *left_file, const char *right_file,
			uint64_t left_size, uint64_t right_size,
			const struct diff_stat *stat)
{
	double percent = 0.0;

	if (stat->bytes_compared != 0) {
		percent = (double)stat->diff_bytes * 100.0 / (double)stat->bytes_compared;
	}

	printf("---------------- Summary ----------------\n");
	printf("Left  : %s (%" PRIu64 " bytes)\n", left_file, left_size);
	printf("Right : %s (%" PRIu64 " bytes)\n", right_file, right_size);
	printf("Bytes compared : %" PRIu64 "\n", stat->bytes_compared);
	printf("Bytes different: %" PRIu64 " (%.4f%%)\n", stat->diff_bytes, percent);
	printf("Bits different : %" PRIu64 "\n", stat->diff_bits);
	// printf("Diff regions   : %" PRIu64 "\n", stat->diff_runs);
	printf("Diff byte kinds: %" PRIu32 "\n", stat->pair_kinds);

	if (stat->diff_bytes != 0) {
		printf("First diff at  : 0x%08" PRIx64 " (%" PRIu64 ")\n",
			stat->first_offset, stat->first_offset);
		printf("Last diff at   : 0x%08" PRIx64 " (%" PRIu64 ")\n",
			stat->last_offset, stat->last_offset);
	}
	if (left_size != right_size) {
		uint64_t extra = (left_size > right_size) ?
			(left_size - right_size) : (right_size - left_size);

		printf("Size mismatch  : %s has %" PRIu64 " extra bytes\n",
			(left_size > right_size) ? left_file : right_file, extra);
	}
	if (stat->truncated) {
		printf("Printed        : %" PRIu64 " of %" PRIu64 " diff lines\n",
			stat->printed, stat->diff_bytes);
	}
	printf("-----------------------------------------\n");

	print_pair_table(stat);

	if ((stat->diff_bytes == 0) && (left_size == right_size)) {
		printf("Match\n");
	} else {
		printf("Mismatch\n");
	}
}

int main(int argc, char *argv[])
{
	const char *left_file = NULL;
	const char *right_file = NULL;
	uint64_t left_size = 0;
	uint64_t right_size = 0;
	uint64_t common_size = 0;
	struct diff_stat stat;

	if ((argc != 3) && (argc != 4)) {
		printf("Version: 0.0.2\n");
		printf("Usage: %s <Left filename> <Right filename>\n", argv[0]);
		printf("       %s <Left filename> <Right filename> <Max print lines>\n", argv[0]);
		return 2;
	}

	left_file = argv[1];
	right_file = argv[2];

	if (get_file_size(left_file, &left_size) < 0) {
		return 2;
	}
	if (get_file_size(right_file, &right_size) < 0) {
		return 2;
	}

	memset(&stat, 0, sizeof(stat));
	if (argc == 4) {
		stat.max_print = strtoull(argv[3], NULL, 0);
	}

	if ((left_size == 0) || (right_size == 0)) {
		printf("File size is zero, %s: %" PRIu64 ",  %s: %" PRIu64 "\n",
			left_file, left_size, right_file, right_size);
		return 2;
	}

	/* 512 KiB table; counting stays enabled only if the allocation works */
	stat.pair_count = calloc(PAIR_TABLE_SIZE, sizeof(*stat.pair_count));
	if (stat.pair_count == NULL) {
		fprintf(stderr, "Out of memory, byte difference counting is disabled\n");
	}
	if (left_size != right_size) {
		printf("File size mismatch, %s: %" PRIu64 ",  %s: %" PRIu64 "\n",
			left_file, left_size, right_file, right_size);
		printf("Comparing the common %" PRIu64 " bytes ...\n",
			(left_size < right_size) ? left_size : right_size);
	}

	common_size = (left_size < right_size) ? left_size : right_size;

	if (compare_files(left_file, right_file, common_size, &stat) < 0) {
		free(stat.pair_count);
		return 2;
	}

	print_summary(left_file, right_file, left_size, right_size, &stat);
	free(stat.pair_count);

	return ((stat.diff_bytes != 0) || (left_size != right_size)) ? 1 : 0;
}
