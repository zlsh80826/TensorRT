import ctypes
import argparse
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action="store_true")
    args = parser.parse_args()

    dtype = trt.float32
    nptype = np.float32
    if args.fp16:
        #dtype = trt.float16
        #nptype = np.half
        print('Precision mode: FP16')
    else:
        print('Precision mode: FP32')

    ctypes.CDLL("libnvinfer_plugin.so", mode=ctypes.RTLD_GLOBAL)

    with open('test.engine', 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime, \
        runtime.deserialize_cuda_engine(f.read()) as engine, \
        engine.create_execution_context() as context:

        # We always use batch size 1.
        assert context.all_binding_shapes_specified
        input_shape = context.get_binding_shape(0)
        mask_shape = context.get_binding_shape(1)
        cu_seq_len_shape = context.get_binding_shape(2)
        dummy_shape = context.get_binding_shape(3)

        # Create a stream in which to copy inputs/outputs and run inference.
        stream = cuda.Stream()

        # Allocate device memory for inputs.
        input_nbytes = trt.volume(input_shape) * dtype.itemsize
        input_mask_nbytes = trt.volume(mask_shape) * dtype.itemsize
        cu_seq_len_nbytes = trt.volume(cu_seq_len_shape) * dtype.itemsize
        dummy_len_nbytes = trt.volume(dummy_shape) * dtype.itemsize
        d_inputs = [cuda.mem_alloc(input_nbytes), cuda.mem_alloc(input_mask_nbytes),
                    cuda.mem_alloc(cu_seq_len_nbytes), cuda.mem_alloc(dummy_len_nbytes)]

        np.random.seed(0)
        sample_input = ((np.random.rand(input_shape[0], input_shape[1], input_shape[2], input_shape[3]) - 0.5) * 1e-3).astype(nptype)
        sample_mask_input = np.ones((mask_shape[0], int(mask_shape[1]))).astype(nptype)
        sample_cu_seq_len = np.array([0, 31, 31 + 33, 31 + 33 + 93, 31 + 33 + 93 + 67], dtype=np.int32)
        sample_dummy = np.zeros((93,), dtype=np.float32)

        # Allocate output buffer by querying the size from the context. This may be different for different input shapes.
        h_output = cuda.pagelocked_empty(tuple(context.get_binding_shape(4)), dtype=nptype)
        d_output = cuda.mem_alloc(h_output.nbytes)

        h_sample_input = cuda.register_host_memory(np.ascontiguousarray(sample_input.ravel()))
        h_sample_mask_input = cuda.register_host_memory(np.ascontiguousarray(sample_mask_input.ravel()))
        h_sample_cu_seq_len = cuda.register_host_memory(np.ascontiguousarray(sample_cu_seq_len.ravel()))
        h_sample_dummy = cuda.register_host_memory(np.ascontiguousarray(sample_dummy.ravel()))

        cuda.memcpy_htod_async(d_inputs[0], h_sample_input, stream)
        cuda.memcpy_htod_async(d_inputs[1], h_sample_mask_input, stream)
        cuda.memcpy_htod_async(d_inputs[2], h_sample_cu_seq_len, stream)
        cuda.memcpy_htod_async(d_inputs[3], h_sample_dummy, stream)

        context.execute_async_v2(bindings=[int(i) for i in d_inputs] + [int(d_output)], stream_handle=stream.handle)
        stream.synchronize()
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()

        print(h_sample_input[: 10])
        print(h_output[:, 0, 0, 0])
