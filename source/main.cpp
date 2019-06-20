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

    float window[189] = { 0 };

    WrappedRamTensor<float> *input_x = new WrappedRamTensor<float>({1, sizeof(window) / sizeof(window[0])}, window);

    printf("Created WrappedRamTensor\n");
    print_memory_info();

    Context ctx;
    get_trained_ctx(ctx, input_x);

    printf("Got trained context\n");
    print_memory_info();

    S_TENSOR pred_tensor = ctx.get("y_pred/Softmax:0");  // getting a reference to the output tensor

    printf("Got pred_tensor\n");
    print_memory_info();

    ctx.eval();
    printf("Called ctx_eval\n");
    print_memory_info();

    int pred_label = *(pred_tensor->read<int>(0, 0));  //getting the result back
    printf("Predicted label: %d\r\n", pred_label);
}

int main() {
    printf("\n\nHello world\n\n");

    run_nn();
}
