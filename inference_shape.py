import onnx
from onnx import shape_inference
path = "checkpoint/45.onnx"
onnx.save(onnx.shape_inference.infer_shapes(onnx.load(path)), path)