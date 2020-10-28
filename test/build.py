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
            if plugin_creator.name == plugin_name and plugin_creator.plugin_version == "2":
                _, hidden_size, _, _ = input_tensor.shape
                hidden_size = int(hidden_size / 3)
                num_heads = 12
                head_size = int(hidden_size / num_heads)
                pf_type = trt.PluginField("type_id", np.array([int(dtype)], np.int32), trt.PluginFieldType.INT32)
                pf_hidden_size = trt.PluginField("hidden_size", np.array([hidden_size], np.int32), trt.PluginFieldType.INT32)
                pf_num_heads = trt.PluginField("num_heads", np.array([num_heads], np.int32), trt.PluginFieldType.INT32)
                pf_has_mask = trt.PluginField("has_mask", np.array([1], np.int32), trt.PluginFieldType.INT32)
                pf_var_seqlen = trt.PluginField("var_seqlen", np.array([1], np.int32), trt.PluginFieldType.INT32)

                field_collection = trt.PluginFieldCollection([pf_hidden_size, pf_num_heads, pf_has_mask, pf_type, pf_var_seqlen])
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


        # assumn the input is 4 sequence, lens are 31,33,93,67
        # input len = [31,33,93,67]
        # mask = [num_seq, fp16maskSize] 
        # cu_seq_len = [num_seq + 1]
        # dummy_tensor = [max_seq_len]

        input_shape = (224, 768 * 3, 1, 1)
        mask_shape = (4, 1024)
        cu_shape = (5,)
        dummy_shape = (93,)

        i0 = network.add_input(name='i0', dtype=dtype, shape=input_shape)
        i1 = network.add_input(name='i1', dtype=dtype, shape=mask_shape)
        i2 = network.add_input(name='i2', dtype=trt.int32, shape=cu_shape)
        i3 = network.add_input(name='i3', dtype=trt.float32, shape=dummy_shape)

        qkv = network.add_plugin_v2(inputs=[i0, i1, i2, i3], 
                                    plugin=get_trt_plugin("CustomQKVToContextPluginDynamic", 
                                                          i0, dtype))
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
