import tensorrt as trt
import numpy as np
import argparse

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

trt.init_libnvinfer_plugins(TRT_LOGGER, '')
PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list


def get_trt_plugin(plugin_name, input_tensor, dtype):
        plugin = None
        for plugin_creator in PLUGIN_CREATORS:
            print(plugin_creator.name)
            if plugin_creator.name == plugin_name:
                B, S, hidden_size, _, _ = input_tensor.shape
                hidden_size = int(hidden_size / 3)
                num_heads = 12
                head_size = int(hidden_size / num_heads)
                pf_type = trt.PluginField("type_id", np.array([int(dtype)], np.int32), trt.PluginFieldType.INT32)
                pf_hidden_size = trt.PluginField("hidden_size", np.array([hidden_size], np.int32), trt.PluginFieldType.INT32)
                pf_num_heads = trt.PluginField("num_heads", np.array([num_heads], np.int32), trt.PluginFieldType.INT32)
                pf_has_mask = trt.PluginField("has_mask", np.array([1], np.int32), trt.PluginFieldType.INT32)

                field_collection = trt.PluginFieldCollection([pf_hidden_size, pf_num_heads, pf_has_mask, pf_type])
                plugin = plugin_creator.create_plugin(name=plugin_name, field_collection=field_collection)
        return plugin


def main(args):
    explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(explicit_batch_flag) as network, \
         builder.create_builder_config() as builder_config:

        builder_config.max_workspace_size = 128 * (1024 * 1024)
        if args.fp16:
            builder_config.set_flag(trt.BuilderFlag.FP16)

        dtype = trt.float32
        if args.fp16:
            dtype = trt.float16

        input_shape = (128, 1, 768 * 3, 1, 1)
        mask_shape = (input_shape[1], int(input_shape[2]/3))
        input_tensor = network.add_input(name="input_layer", dtype=dtype, shape=input_shape)
        input_mask_tensor = network.add_input(name="input_mask_layer", dtype=dtype, shape=mask_shape)
        qkv = network.add_plugin_v2(inputs=[input_tensor, input_mask_tensor], 
                                    plugin=get_trt_plugin("CustomQKVToContextPluginDynamic", 
                                                          input_tensor, dtype))
        qkv.get_output(0).name = "output"
        network.mark_output(qkv.get_output(0))
        engine = builder.build_engine(network, builder_config)
        serialized_engine = engine.serialize()
        TRT_LOGGER.log(TRT_LOGGER.INFO, "Saving Engine to {:}".format("test.engine"))
        with open("test.engine", "wb") as fout:
            fout.write(serialized_engine)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action="store_true")
    args = parser.parse_args()
    if args.fp16:
        print('Precision mode: FP16')
    else:
        print('Precision mode: FP32')
    main(args)
