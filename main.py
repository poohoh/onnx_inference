import time

import onnxruntime as ort
import numpy as np

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# model load
# session = ort.InferenceSession("checkpoint/45.onnx")

# fire detection
# session = ort.InferenceSession("../model/fire_detection/ONNX/45.onnx", providers=["CPUExecutionProvider"])
# session = ort.InferenceSession("../model/fire_detection/ONNX/45.onnx", providers=["CUDAExecutionProvider"])

# Vehicle Detection
# session = ort.InferenceSession("../model/230914_vd/ONNX/vd.onnx", providers=["CPUExecutionProvider"])
session = ort.InferenceSession("../model/230914_vd/ONNX/vd.onnx", providers=["CUDAExecutionProvider"])

# License Plate Detection
# session = ort.InferenceSession("../model/230911_lpd/ONNX/lpd.onnx", providers=["CPUExecutionProvider"])
# session = ort.InferenceSession("../model/230911_lpd/ONNX/lpd.onnx", providers=["CUDAExecutionProvider"])

# License Plate Recognition
# session = ort.InferenceSession("../model/230918_lpr/ONNX/lpr.onnx", providers=["CPUExecutionProvider"])
# session = ort.InferenceSession("../model/230918_lpr/ONNX/lpr.onnx", providers=["CUDAExecutionProvider"])



# run model
sum = 0
for i in range(10000):

    # input data
    input_data = np.random.randn(1, 3, 416, 416).astype(np.float32)
    input_name = session.get_inputs()[0].name

    start = time.time()
    results = session.run(None, {input_name: input_data})
    end = time.time()
    sum += end - start

    if i % 100 == 0:
        print(f'i: {i}, sum: {sum}')
avg_elapsed = sum / 10000

# result
print(results)
print(f'elapsed time: {avg_elapsed}')