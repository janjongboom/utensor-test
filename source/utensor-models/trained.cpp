// Auto generated by utensor-cli

#include "trained.hpp"
#include "uTensor/ops/MathOps.hpp"
#include "uTensor/core/tensor.hpp"
#include "uTensor/core/context.hpp"
#include "uTensor/ops/NnOps.hpp"
#include "uTensor/ops/ArrayOps.hpp"
#include "uTensor/ops/MatrixOps.hpp"
#include "trained_weight.hpp"


void get_trained_ctx(Context& ctx, Tensor* input_0) {

{ // add tensor for placeholders
    ctx.add(input_0, "x_input:0", 2);
}
{
    ctx.add(new BinaryTensor<int>({1}, inline_x_MatMul_eightbit_x_input__port__0_reshape_dims_0),
            "x/MatMul_eightbit/x_input__port__0/reshape_dims:0",
            1);
}
{
    ctx.add(new RamTensor<float>(), "x/MatMul_eightbit/x_input__port__0/reshape:0", 2);
    ctx.push(new ReshapeOp(),
             { "x_input:0", "x/MatMul_eightbit/x_input__port__0/reshape_dims:0" },
             { "x/MatMul_eightbit/x_input__port__0/reshape:0" });
    ctx.eval();
}
{
    ctx.add(new BinaryTensor<int>({1}, inline_x_MatMul_eightbit_x_input__port__0_reduction_dims_0),
            "x/MatMul_eightbit/x_input__port__0/reduction_dims:0",
            2);
}
{
    RamTensor<float>* out_tensor;
    out_tensor = new RamTensor<float>({ 1 });
    ctx.add(out_tensor, "x/MatMul_eightbit/x_input__port__0/min:0", 1);
    ctx.push(new MinOp(),
             { "x/MatMul_eightbit/x_input__port__0/reshape:0", "x/MatMul_eightbit/x_input__port__0/reduction_dims:0" },
             { "x/MatMul_eightbit/x_input__port__0/min:0" });
    ctx.eval();
}
{
    RamTensor<float>* out_tensor;
    out_tensor = new RamTensor<float>({ 1 });
    ctx.add(out_tensor, "x/MatMul_eightbit/x_input__port__0/max:0", 1);
    ctx.push(new MaxOp(),
             { "x/MatMul_eightbit/x_input__port__0/reshape:0", "x/MatMul_eightbit/x_input__port__0/reduction_dims:0" },
             { "x/MatMul_eightbit/x_input__port__0/max:0" });
    ctx.eval();
}
{
    ctx.add(new RamTensor<uint8_t>(), "x/MatMul_eightbit/x_input__port__0/quantize:0", 1);
    ctx.add(new RamTensor<float>({1}), "x/MatMul_eightbit/x_input__port__0/quantize:1", 1);
    ctx.add(new RamTensor<float>({1}), "x/MatMul_eightbit/x_input__port__0/quantize:2", 1);
    ctx.push(new QuantizeV2Op(),
             {  "x_input:0",  "x/MatMul_eightbit/x_input__port__0/min:0", "x/MatMul_eightbit/x_input__port__0/max:0" },
             {  "x/MatMul_eightbit/x_input__port__0/quantize:0",  "x/MatMul_eightbit/x_input__port__0/quantize:1", "x/MatMul_eightbit/x_input__port__0/quantize:2" });
    ctx.eval();
}
{
    ctx.add(new BinaryTensor<uint8_t>({186,40}, inline_x_kernel_quantized_const_0),
            "x/kernel_quantized_const:0",
            1);
}
{
    ctx.add(new BinaryTensor<float>({1}, inline_x_kernel_quantized_min_0),
            "x/kernel_quantized_min:0",
            1);
}
{
    ctx.add(new BinaryTensor<float>({1}, inline_x_kernel_quantized_max_0),
            "x/kernel_quantized_max:0",
            1);
}
{
    ctx.add(new RamTensor<float>(), "x/kernel:0", 2);
    ctx.push(new DequantizeOp(),
             { "x/kernel_quantized_const:0", "x/kernel_quantized_min:0", "x/kernel_quantized_max:0" },
             { "x/kernel:0" });
    ctx.eval();
}
{
    ctx.add(new BinaryTensor<int>({1}, inline_x_MatMul_eightbit_x_kernel_read__port__0_reshape_dims_0),
            "x/MatMul_eightbit/x/kernel/read__port__0/reshape_dims:0",
            1);
}
{
    ctx.add(new RamTensor<float>(), "x/MatMul_eightbit/x/kernel/read__port__0/reshape:0", 2);
    ctx.push(new ReshapeOp(),
             { "x/kernel:0", "x/MatMul_eightbit/x/kernel/read__port__0/reshape_dims:0" },
             { "x/MatMul_eightbit/x/kernel/read__port__0/reshape:0" });
    ctx.eval();
}
{
    ctx.add(new BinaryTensor<int>({1}, inline_x_MatMul_eightbit_x_kernel_read__port__0_reduction_dims_0),
            "x/MatMul_eightbit/x/kernel/read__port__0/reduction_dims:0",
            2);
}
{
    RamTensor<float>* out_tensor;
    out_tensor = new RamTensor<float>({ 1 });
    ctx.add(out_tensor, "x/MatMul_eightbit/x/kernel/read__port__0/min:0", 1);
    ctx.push(new MinOp(),
             { "x/MatMul_eightbit/x/kernel/read__port__0/reshape:0", "x/MatMul_eightbit/x/kernel/read__port__0/reduction_dims:0" },
             { "x/MatMul_eightbit/x/kernel/read__port__0/min:0" });
    ctx.eval();
}
{
    RamTensor<float>* out_tensor;
    out_tensor = new RamTensor<float>({ 1 });
    ctx.add(out_tensor, "x/MatMul_eightbit/x/kernel/read__port__0/max:0", 1);
    ctx.push(new MaxOp(),
             { "x/MatMul_eightbit/x/kernel/read__port__0/reshape:0", "x/MatMul_eightbit/x/kernel/read__port__0/reduction_dims:0" },
             { "x/MatMul_eightbit/x/kernel/read__port__0/max:0" });
    ctx.eval();
}
{
    ctx.add(new RamTensor<uint8_t>(), "x/MatMul_eightbit/x/kernel/read__port__0/quantize:0", 1);
    ctx.add(new RamTensor<float>({1}), "x/MatMul_eightbit/x/kernel/read__port__0/quantize:1", 1);
    ctx.add(new RamTensor<float>({1}), "x/MatMul_eightbit/x/kernel/read__port__0/quantize:2", 1);
    ctx.push(new QuantizeV2Op(),
             {  "x/kernel:0",  "x/MatMul_eightbit/x/kernel/read__port__0/min:0", "x/MatMul_eightbit/x/kernel/read__port__0/max:0" },
             {  "x/MatMul_eightbit/x/kernel/read__port__0/quantize:0",  "x/MatMul_eightbit/x/kernel/read__port__0/quantize:1", "x/MatMul_eightbit/x/kernel/read__port__0/quantize:2" });
    ctx.eval();
}
{
    ctx.add(new RamTensor<int>(), "x/MatMul/eightbit:0", 2);
    ctx.add(new RamTensor<float>({1}), "x/MatMul/eightbit:1", 2);
    ctx.add(new RamTensor<float>({1}), "x/MatMul/eightbit:2", 2);
    ctx.push(new QntMatMulOp<uint8_t, uint8_t, int>(),
             { "x/MatMul_eightbit/x_input__port__0/quantize:0", "x/MatMul_eightbit/x_input__port__0/quantize:1", "x/MatMul_eightbit/x_input__port__0/quantize:2", "x/MatMul_eightbit/x/kernel/read__port__0/quantize:0", "x/MatMul_eightbit/x/kernel/read__port__0/quantize:1",  "x/MatMul_eightbit/x/kernel/read__port__0/quantize:2" },
             { "x/MatMul/eightbit:0", "x/MatMul/eightbit:1",  "x/MatMul/eightbit:2" });
    ctx.eval();
}
{
    ctx.add(new RamTensor<float>({1}), "x/MatMul/eightbit/requant_range:0", 1);
    ctx.add(new RamTensor<float>({1}), "x/MatMul/eightbit/requant_range:1", 1);
    ctx.push(new Requantization_RangeOp(),
             { "x/MatMul/eightbit:0", "x/MatMul/eightbit:1", "x/MatMul/eightbit:2" },
             { "x/MatMul/eightbit/requant_range:0", "x/MatMul/eightbit/requant_range:1" });
    ctx.eval();
}
{
    ctx.add(new RamTensor<uint8_t>(), "x/MatMul/eightbit/requantize:0", 1);
    ctx.add(new RamTensor<float>({1}), "x/MatMul/eightbit/requantize:1", 1);
    ctx.add(new RamTensor<float>({1}), "x/MatMul/eightbit/requantize:2", 1);
    ctx.push(new RequantizeOp(),
             { "x/MatMul/eightbit:0", "x/MatMul/eightbit:1", "x/MatMul/eightbit:2", "x/MatMul/eightbit/requant_range:0", "x/MatMul/eightbit/requant_range:1" },
             { "x/MatMul/eightbit/requantize:0", "x/MatMul/eightbit/requantize:1", "x/MatMul/eightbit/requantize:2" });
    ctx.eval();
}
{
    ctx.add(new BinaryTensor<float>({40}, inline_x_bias_0),
            "x/bias:0",
            2);
}
{
    ctx.add(new BinaryTensor<int>({1}, inline_x_BiasAdd_eightbit_x_bias_read__port__0_reshape_dims_0),
            "x/BiasAdd_eightbit/x/bias/read__port__0/reshape_dims:0",
            1);
}
{
    ctx.add(new RamTensor<float>(), "x/BiasAdd_eightbit/x/bias/read__port__0/reshape:0", 2);
    ctx.push(new ReshapeOp(),
             { "x/bias:0", "x/BiasAdd_eightbit/x/bias/read__port__0/reshape_dims:0" },
             { "x/BiasAdd_eightbit/x/bias/read__port__0/reshape:0" });
    ctx.eval();
}
{
    ctx.add(new BinaryTensor<int>({1}, inline_x_BiasAdd_eightbit_x_bias_read__port__0_reduction_dims_0),
            "x/BiasAdd_eightbit/x/bias/read__port__0/reduction_dims:0",
            2);
}
{
    RamTensor<float>* out_tensor;
    out_tensor = new RamTensor<float>({ 1 });
    ctx.add(out_tensor, "x/BiasAdd_eightbit/x/bias/read__port__0/min:0", 1);
    ctx.push(new MinOp(),
             { "x/BiasAdd_eightbit/x/bias/read__port__0/reshape:0", "x/BiasAdd_eightbit/x/bias/read__port__0/reduction_dims:0" },
             { "x/BiasAdd_eightbit/x/bias/read__port__0/min:0" });
    ctx.eval();
}
{
    RamTensor<float>* out_tensor;
    out_tensor = new RamTensor<float>({ 1 });
    ctx.add(out_tensor, "x/BiasAdd_eightbit/x/bias/read__port__0/max:0", 1);
    ctx.push(new MaxOp(),
             { "x/BiasAdd_eightbit/x/bias/read__port__0/reshape:0", "x/BiasAdd_eightbit/x/bias/read__port__0/reduction_dims:0" },
             { "x/BiasAdd_eightbit/x/bias/read__port__0/max:0" });
    ctx.eval();
}
{
    ctx.add(new RamTensor<uint8_t>(), "x/BiasAdd_eightbit/x/bias/read__port__0/quantize:0", 1);
    ctx.add(new RamTensor<float>({1}), "x/BiasAdd_eightbit/x/bias/read__port__0/quantize:1", 1);
    ctx.add(new RamTensor<float>({1}), "x/BiasAdd_eightbit/x/bias/read__port__0/quantize:2", 1);
    ctx.push(new QuantizeV2Op(),
             {  "x/bias:0",  "x/BiasAdd_eightbit/x/bias/read__port__0/min:0", "x/BiasAdd_eightbit/x/bias/read__port__0/max:0" },
             {  "x/BiasAdd_eightbit/x/bias/read__port__0/quantize:0",  "x/BiasAdd_eightbit/x/bias/read__port__0/quantize:1", "x/BiasAdd_eightbit/x/bias/read__port__0/quantize:2" });
    ctx.eval();
}
{
    ctx.add(new RamTensor<int>(), "x/BiasAdd/eightbit:0", 2);
    ctx.add(new RamTensor<float>({1}), "x/BiasAdd/eightbit:1", 2);
    ctx.add(new RamTensor<float>({1}), "x/BiasAdd/eightbit:2", 2);
    ctx.push(new QuantizedAddOp<uint8_t, uint8_t, int>(),
             { "x/MatMul/eightbit/requantize:0", "x/MatMul/eightbit/requantize:1", "x/MatMul/eightbit/requantize:2", "x/BiasAdd_eightbit/x/bias/read__port__0/quantize:0", "x/BiasAdd_eightbit/x/bias/read__port__0/quantize:1",  "x/BiasAdd_eightbit/x/bias/read__port__0/quantize:2" },
             { "x/BiasAdd/eightbit:0", "x/BiasAdd/eightbit:1",  "x/BiasAdd/eightbit:2" });
    ctx.eval();
}
{
    ctx.add(new RamTensor<float>({1}), "x/BiasAdd/eightbit/requant_range:0", 1);
    ctx.add(new RamTensor<float>({1}), "x/BiasAdd/eightbit/requant_range:1", 1);
    ctx.push(new Requantization_RangeOp(),
             { "x/BiasAdd/eightbit:0", "x/BiasAdd/eightbit:1", "x/BiasAdd/eightbit:2" },
             { "x/BiasAdd/eightbit/requant_range:0", "x/BiasAdd/eightbit/requant_range:1" });
    ctx.eval();
}
{
    ctx.add(new RamTensor<uint8_t>(), "x/BiasAdd/eightbit/requantize:0", 1);
    ctx.add(new RamTensor<float>({1}), "x/BiasAdd/eightbit/requantize:1", 1);
    ctx.add(new RamTensor<float>({1}), "x/BiasAdd/eightbit/requantize:2", 1);
    ctx.push(new RequantizeOp(),
             { "x/BiasAdd/eightbit:0", "x/BiasAdd/eightbit:1", "x/BiasAdd/eightbit:2", "x/BiasAdd/eightbit/requant_range:0", "x/BiasAdd/eightbit/requant_range:1" },
             { "x/BiasAdd/eightbit/requantize:0", "x/BiasAdd/eightbit/requantize:1", "x/BiasAdd/eightbit/requantize:2" });
    ctx.eval();
}
{
    ctx.add(new RamTensor<uint8_t>(), "x/Relu/eightbit:0", 1);
    ctx.add(new RamTensor<float>({1}), "x/Relu/eightbit:1", 1);
    ctx.add(new RamTensor<float>({1}), "x/Relu/eightbit:2", 1);
    ctx.push(new QuantizedReluOp<uint8_t, float, uint8_t>(),
             { "x/BiasAdd/eightbit/requantize:0", "x/BiasAdd/eightbit/requantize:1", "x/BiasAdd/eightbit/requantize:2" },
             { "x/Relu/eightbit:0", "x/Relu/eightbit:1", "x/Relu/eightbit:2" });
    ctx.eval();
}
{
    ctx.add(new BinaryTensor<float>({40,3}, inline_y_pred_kernel_0),
            "y_pred/kernel:0",
            2);
}
{
    ctx.add(new BinaryTensor<int>({1}, inline_y_pred_MatMul_eightbit_y_pred_kernel_read__port__0_reshape_dims_0),
            "y_pred/MatMul_eightbit/y_pred/kernel/read__port__0/reshape_dims:0",
            1);
}
{
    ctx.add(new RamTensor<float>(), "y_pred/MatMul_eightbit/y_pred/kernel/read__port__0/reshape:0", 2);
    ctx.push(new ReshapeOp(),
             { "y_pred/kernel:0", "y_pred/MatMul_eightbit/y_pred/kernel/read__port__0/reshape_dims:0" },
             { "y_pred/MatMul_eightbit/y_pred/kernel/read__port__0/reshape:0" });
    ctx.eval();
}
{
    ctx.add(new BinaryTensor<int>({1}, inline_y_pred_MatMul_eightbit_y_pred_kernel_read__port__0_reduction_dims_0),
            "y_pred/MatMul_eightbit/y_pred/kernel/read__port__0/reduction_dims:0",
            2);
}
{
    RamTensor<float>* out_tensor;
    out_tensor = new RamTensor<float>({ 1 });
    ctx.add(out_tensor, "y_pred/MatMul_eightbit/y_pred/kernel/read__port__0/min:0", 1);
    ctx.push(new MinOp(),
             { "y_pred/MatMul_eightbit/y_pred/kernel/read__port__0/reshape:0", "y_pred/MatMul_eightbit/y_pred/kernel/read__port__0/reduction_dims:0" },
             { "y_pred/MatMul_eightbit/y_pred/kernel/read__port__0/min:0" });
    ctx.eval();
}
{
    RamTensor<float>* out_tensor;
    out_tensor = new RamTensor<float>({ 1 });
    ctx.add(out_tensor, "y_pred/MatMul_eightbit/y_pred/kernel/read__port__0/max:0", 1);
    ctx.push(new MaxOp(),
             { "y_pred/MatMul_eightbit/y_pred/kernel/read__port__0/reshape:0", "y_pred/MatMul_eightbit/y_pred/kernel/read__port__0/reduction_dims:0" },
             { "y_pred/MatMul_eightbit/y_pred/kernel/read__port__0/max:0" });
    ctx.eval();
}
{
    ctx.add(new RamTensor<uint8_t>(), "y_pred/MatMul_eightbit/y_pred/kernel/read__port__0/quantize:0", 1);
    ctx.add(new RamTensor<float>({1}), "y_pred/MatMul_eightbit/y_pred/kernel/read__port__0/quantize:1", 1);
    ctx.add(new RamTensor<float>({1}), "y_pred/MatMul_eightbit/y_pred/kernel/read__port__0/quantize:2", 1);
    ctx.push(new QuantizeV2Op(),
             {  "y_pred/kernel:0",  "y_pred/MatMul_eightbit/y_pred/kernel/read__port__0/min:0", "y_pred/MatMul_eightbit/y_pred/kernel/read__port__0/max:0" },
             {  "y_pred/MatMul_eightbit/y_pred/kernel/read__port__0/quantize:0",  "y_pred/MatMul_eightbit/y_pred/kernel/read__port__0/quantize:1", "y_pred/MatMul_eightbit/y_pred/kernel/read__port__0/quantize:2" });
    ctx.eval();
}
{
    ctx.add(new RamTensor<int>(), "y_pred/MatMul/eightbit:0", 2);
    ctx.add(new RamTensor<float>({1}), "y_pred/MatMul/eightbit:1", 2);
    ctx.add(new RamTensor<float>({1}), "y_pred/MatMul/eightbit:2", 2);
    ctx.push(new QntMatMulOp<uint8_t, uint8_t, int>(),
             { "x/Relu/eightbit:0", "x/Relu/eightbit:1", "x/Relu/eightbit:2", "y_pred/MatMul_eightbit/y_pred/kernel/read__port__0/quantize:0", "y_pred/MatMul_eightbit/y_pred/kernel/read__port__0/quantize:1",  "y_pred/MatMul_eightbit/y_pred/kernel/read__port__0/quantize:2" },
             { "y_pred/MatMul/eightbit:0", "y_pred/MatMul/eightbit:1",  "y_pred/MatMul/eightbit:2" });
    ctx.eval();
}
{
    ctx.add(new RamTensor<float>({1}), "y_pred/MatMul/eightbit/requant_range:0", 1);
    ctx.add(new RamTensor<float>({1}), "y_pred/MatMul/eightbit/requant_range:1", 1);
    ctx.push(new Requantization_RangeOp(),
             { "y_pred/MatMul/eightbit:0", "y_pred/MatMul/eightbit:1", "y_pred/MatMul/eightbit:2" },
             { "y_pred/MatMul/eightbit/requant_range:0", "y_pred/MatMul/eightbit/requant_range:1" });
    ctx.eval();
}
{
    ctx.add(new RamTensor<uint8_t>(), "y_pred/MatMul/eightbit/requantize:0", 1);
    ctx.add(new RamTensor<float>({1}), "y_pred/MatMul/eightbit/requantize:1", 1);
    ctx.add(new RamTensor<float>({1}), "y_pred/MatMul/eightbit/requantize:2", 1);
    ctx.push(new RequantizeOp(),
             { "y_pred/MatMul/eightbit:0", "y_pred/MatMul/eightbit:1", "y_pred/MatMul/eightbit:2", "y_pred/MatMul/eightbit/requant_range:0", "y_pred/MatMul/eightbit/requant_range:1" },
             { "y_pred/MatMul/eightbit/requantize:0", "y_pred/MatMul/eightbit/requantize:1", "y_pred/MatMul/eightbit/requantize:2" });
    ctx.eval();
}
{
    ctx.add(new BinaryTensor<float>({3}, inline_y_pred_bias_0),
            "y_pred/bias:0",
            2);
}
{
    ctx.add(new BinaryTensor<int>({1}, inline_y_pred_BiasAdd_eightbit_y_pred_bias_read__port__0_reshape_dims_0),
            "y_pred/BiasAdd_eightbit/y_pred/bias/read__port__0/reshape_dims:0",
            1);
}
{
    ctx.add(new RamTensor<float>(), "y_pred/BiasAdd_eightbit/y_pred/bias/read__port__0/reshape:0", 2);
    ctx.push(new ReshapeOp(),
             { "y_pred/bias:0", "y_pred/BiasAdd_eightbit/y_pred/bias/read__port__0/reshape_dims:0" },
             { "y_pred/BiasAdd_eightbit/y_pred/bias/read__port__0/reshape:0" });
    ctx.eval();
}
{
    ctx.add(new BinaryTensor<int>({1}, inline_y_pred_BiasAdd_eightbit_y_pred_bias_read__port__0_reduction_dims_0),
            "y_pred/BiasAdd_eightbit/y_pred/bias/read__port__0/reduction_dims:0",
            2);
}
{
    RamTensor<float>* out_tensor;
    out_tensor = new RamTensor<float>({ 1 });
    ctx.add(out_tensor, "y_pred/BiasAdd_eightbit/y_pred/bias/read__port__0/min:0", 1);
    ctx.push(new MinOp(),
             { "y_pred/BiasAdd_eightbit/y_pred/bias/read__port__0/reshape:0", "y_pred/BiasAdd_eightbit/y_pred/bias/read__port__0/reduction_dims:0" },
             { "y_pred/BiasAdd_eightbit/y_pred/bias/read__port__0/min:0" });
    ctx.eval();
}
{
    RamTensor<float>* out_tensor;
    out_tensor = new RamTensor<float>({ 1 });
    ctx.add(out_tensor, "y_pred/BiasAdd_eightbit/y_pred/bias/read__port__0/max:0", 1);
    ctx.push(new MaxOp(),
             { "y_pred/BiasAdd_eightbit/y_pred/bias/read__port__0/reshape:0", "y_pred/BiasAdd_eightbit/y_pred/bias/read__port__0/reduction_dims:0" },
             { "y_pred/BiasAdd_eightbit/y_pred/bias/read__port__0/max:0" });
    ctx.eval();
}
{
    ctx.add(new RamTensor<uint8_t>(), "y_pred/BiasAdd_eightbit/y_pred/bias/read__port__0/quantize:0", 1);
    ctx.add(new RamTensor<float>({1}), "y_pred/BiasAdd_eightbit/y_pred/bias/read__port__0/quantize:1", 1);
    ctx.add(new RamTensor<float>({1}), "y_pred/BiasAdd_eightbit/y_pred/bias/read__port__0/quantize:2", 1);
    ctx.push(new QuantizeV2Op(),
             {  "y_pred/bias:0",  "y_pred/BiasAdd_eightbit/y_pred/bias/read__port__0/min:0", "y_pred/BiasAdd_eightbit/y_pred/bias/read__port__0/max:0" },
             {  "y_pred/BiasAdd_eightbit/y_pred/bias/read__port__0/quantize:0",  "y_pred/BiasAdd_eightbit/y_pred/bias/read__port__0/quantize:1", "y_pred/BiasAdd_eightbit/y_pred/bias/read__port__0/quantize:2" });
    ctx.eval();
}
{
    ctx.add(new RamTensor<int>(), "y_pred/BiasAdd/eightbit:0", 2);
    ctx.add(new RamTensor<float>({1}), "y_pred/BiasAdd/eightbit:1", 2);
    ctx.add(new RamTensor<float>({1}), "y_pred/BiasAdd/eightbit:2", 2);
    ctx.push(new QuantizedAddOp<uint8_t, uint8_t, int>(),
             { "y_pred/MatMul/eightbit/requantize:0", "y_pred/MatMul/eightbit/requantize:1", "y_pred/MatMul/eightbit/requantize:2", "y_pred/BiasAdd_eightbit/y_pred/bias/read__port__0/quantize:0", "y_pred/BiasAdd_eightbit/y_pred/bias/read__port__0/quantize:1",  "y_pred/BiasAdd_eightbit/y_pred/bias/read__port__0/quantize:2" },
             { "y_pred/BiasAdd/eightbit:0", "y_pred/BiasAdd/eightbit:1",  "y_pred/BiasAdd/eightbit:2" });
    ctx.eval();
}
{
    ctx.add(new RamTensor<float>({1}), "y_pred/BiasAdd/eightbit/requant_range:0", 1);
    ctx.add(new RamTensor<float>({1}), "y_pred/BiasAdd/eightbit/requant_range:1", 1);
    ctx.push(new Requantization_RangeOp(),
             { "y_pred/BiasAdd/eightbit:0", "y_pred/BiasAdd/eightbit:1", "y_pred/BiasAdd/eightbit:2" },
             { "y_pred/BiasAdd/eightbit/requant_range:0", "y_pred/BiasAdd/eightbit/requant_range:1" });
    ctx.eval();
}
{
    ctx.add(new RamTensor<uint8_t>(), "y_pred/BiasAdd/eightbit/requantize:0", 1);
    ctx.add(new RamTensor<float>({1}), "y_pred/BiasAdd/eightbit/requantize:1", 1);
    ctx.add(new RamTensor<float>({1}), "y_pred/BiasAdd/eightbit/requantize:2", 1);
    ctx.push(new RequantizeOp(),
             { "y_pred/BiasAdd/eightbit:0", "y_pred/BiasAdd/eightbit:1", "y_pred/BiasAdd/eightbit:2", "y_pred/BiasAdd/eightbit/requant_range:0", "y_pred/BiasAdd/eightbit/requant_range:1" },
             { "y_pred/BiasAdd/eightbit/requantize:0", "y_pred/BiasAdd/eightbit/requantize:1", "y_pred/BiasAdd/eightbit/requantize:2" });
    ctx.eval();
}
{
    ctx.add(new RamTensor<float>(), "y_pred/BiasAdd:0", 1);
    ctx.push(new DequantizeOp(),
             { "y_pred/BiasAdd/eightbit/requantize:0", "y_pred/BiasAdd/eightbit/requantize:1", "y_pred/BiasAdd/eightbit/requantize:2" },
             { "y_pred/BiasAdd:0" });
    ctx.eval();
}
{
    ctx.add(new RamTensor<float>(), "y_pred/Softmax:0");
    ctx.push(new SoftmaxOp(),
             { "y_pred/BiasAdd:0" },
             { "y_pred/Softmax:0" });
}
}