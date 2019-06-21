#include "mbed.h"
#include "trained.hpp"
#include "trained_weight.hpp"            // keep the weights in ROM for now, we have plenty of internal flash

static DigitalOut led(LED1);
static EventQueue queue;

void print_memory_info() {
    // Grab the heap statistics
    mbed_stats_heap_t heap_stats;
    mbed_stats_heap_get(&heap_stats);
    printf("Heap size: %lu / %lu bytes\r\n", heap_stats.current_size, heap_stats.reserved_size);
}

void run_nn() {
    print_memory_info();

    float window_class_1[186] = { 0 };

    float window_class_2[186] = {
        78, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22, 17, 0, -22, -17, 0, 16, -10, 0, -82, 38, 0, 84, 69, 0, 0, 0, 0, -19, 0, 0, 0, 0, 0, 0, 18, -1, 0, -18, 1, -69, -80, 0, -2, 80, 0, 0, 0, 0, -3, 18, 0, 103, -18, 0, -3, 27, 0, -24, -97, 0, 0, -8, 0, 0, 78, 0, 0, 0, 0, 22, 0, 0, 9, 0, 0, -32, 18, 0, 0, -18, 0, 0, -76, 0, 0, 8, 0, 0, 0, 0, 19, 68, 0, -19, 0, 0, -78, -97, 0, 10, 0, 0, 68, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22, 0, 0, 2, 0, 16, -25, -1, -16, -2, 1, 0, -5, 0, 23, -25, -1, 0, 0, 0, -23, -15, 1, -97, 15, 0, -32, 58, 0, 32, 71, 0, 24, -98, 0, 0, 0, 0, 0, 0, 0, 74, 22, 0, 0, 75, 0, -73, 0, 0, -4, 0, 0, 0, 0, 0, -20, 0, 0, 0, 0, 0, 0, 17, 0, 0, -17, 0, 0, 0, 0
    };

    float window_class_3[186] = {
        0, 0, 0, 8, -17, 0, 0, 22, -6, 4, 75, -25, -30, 0, 0, 31, 0, 0, -31, 0, 0, 0, 0, 0, 0, 16, 0, 0, -16, 0, 29, 0, 16, -1, -71, 16, 0, 1, 26, -3, 70, 5, 0, 0, 0, -25, 0, 22, 0, 0, 9, 0, 0, 24, 18, 0, 8, 4, 0, 0, -5, 0, 32, 0, 0, 0, -18, 0, 32, 0, 0, 18, 0, -67, 13, 0, 1, 0, 0, -4, 17, 0, -2, -17, 0, 0, 0, 0, 2, 0, -74, 0, 0, -23, -7, 29, 0, -20, 67, 17, 19, -78, 6, -3, 1, 0, 0, 0, -6, -16, -3, -17, 0, -16, 17, 0, 0, 11, 0, -11, -1, 17, -21, 0, 0, 0, -4, -17, 0, 1, 0, 0, 7, 25, -8, 0, 72, -23, 0, 0, -8, -66, -70, 7, 0, 0, 0, 66, -4, 1, 17, 6, -6, 0, 68, -9, -17, -80, -16, 0, -17, 20, 23, 17, -21, 0, 0, 0, -23, 6, -13, 0, 2, -17, 21, 70, -16, -21, -65, -16, 0, 65, -7, 29, 17, -25
    };

    // which one you want?
    float *windows[] = {
        window_class_1,
        window_class_2,
        window_class_3
    };

    printf("Created WrappedRamTensor\n");
    print_memory_info();

    for (uint8_t ix = 0; ix < 3; ix++) {
        printf("Classifying example %u\n", ix);

        float *window = windows[ix];

        // so my board hardfaults when removing this later, I guess the memory is managed by uTensor
        // this is not good practice I'd say
        RamTensor<float> *input_x = new RamTensor<float>({ 1, 186 });
        float *buff = (float*)input_x->write(0, 0);
        for (int ix = 0; ix < 186; ix++) {
            buff[ix] = window[ix];
        }

        Context ctx;
        get_trained_ctx(ctx, input_x);
        ctx.eval();

        printf("Predictions:\n");

        S_TENSOR pred_tensor = ctx.get("y_pred/Softmax:0");  // getting a reference to the output tensor
        printf("Got pred_tensor\n");
        print_memory_info();

        const float* ptr_pred = pred_tensor->read<float>(0, 0);
        printf("0: %f\r\n", *ptr_pred);
        printf("1: %f\r\n", *(ptr_pred + 1));
        printf("2: %f\r\n", *(ptr_pred + 2));

        printf("\n");

        // delete input_x;
    }
}

int main() {
    printf("\n\nHello world\n\n");

    run_nn();
}
