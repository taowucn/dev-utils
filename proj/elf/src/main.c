#include <stdio.h>

int sum(int a, int b);

int g_array[2] = {1, 2};

int main(void) {

    int result = 0;
    result = sum(g_array[0], g_array[1]);
    printf("sum_result: %d\n", result);

    return 0;
}
