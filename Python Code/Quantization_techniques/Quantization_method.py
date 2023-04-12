import tensorflow as tf

def quantized(model,mod_name):
    df1=pd.DataFrame(columns=['dataset', 'score', 'size', 'type'])
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_models_dir = pathlib.Path("your_path")
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    direc=str(mod_name  + "your_model_name_model.tflite")
    tflite_model_file = tflite_models_dir/direc
    tflite_model_file.write_bytes(tflite_model)

    ###Post-training float16 quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    direc=str(mod_name  + "your_model_name_quant_f16.tflite")
    tflite_fp16_model = converter.convert()
    tflite_model_fp16_file = tflite_models_dir/direc
    tflite_model_fp16_file.write_bytes(tflite_fp16_model)

    #Post-training dynamic range quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()
    direc=str(mod_name  + "your_model_name_quant.tflite")
    tflite_model_quant_file = tflite_models_dir/direc
    tflite_model_quant_file.write_bytes(tflite_quant_model)

    #Post-training integer quantization
    #direc=str(mod_name + var + "effi_model_quant_float.tflite")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen()
    tflite_model_quant_float = converter.convert()
    direc=str(mod_name  + "your_model_name_quant_float.tflite")
    tflite_model_quant_float_file = tflite_models_dir/direc
    tflite_model_quant_float_file.write_bytes(tflite_model_quant_float)

    #Convert using integer-only quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen()
    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Set the input and output tensors to uint8 (APIs added in r2.3)
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    direc=str(mod_name  + "your_model_name_quant_int.tflite")
    tflite_model_quant_int = converter.convert()
    tflite_model_quant_int_file = tflite_models_dir/direc
    tflite_model_quant_int_file.write_bytes(tflite_model_quant_int)

    #Load the model into the interpreters

    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
    interpreter.allocate_tensors()

    interpreter_fp16 = tf.lite.Interpreter(model_path=str(tflite_model_fp16_file))
    interpreter_fp16.allocate_tensors()

    interpreter_quant = tf.lite.Interpreter(model_path=str(tflite_model_quant_file))
    interpreter_quant.allocate_tensors()

    interpreter_quant_float = tf.lite.Interpreter(model_path=str(tflite_model_quant_float_file))
    interpreter_quant_float.allocate_tensors()

   interpreter_quant_int = tf.lite.Interpreter(model_path=str(tflite_model_quant_int_file))
    interpreter_quant_int.allocate_tensors()

    models=[interpreter, interpreter_fp16, interpreter_quant]
    tam = [tflite_model_file, tflite_model_fp16_file, tflite_model_quant_file]
    x = 0
    for test_model in models:
        score = evaluate_model(test_model)
        print(score)
        print("######################################")
        size_tf = os.path.getsize(tam[x])
        print(size_tf)
        x += 1
        df1=df1.append({'dataset':num, 'score':score, 'size':size_tf, 'type':mod_name}, ignore_index=True)
    return df1 #return a DataFrame

  
  
