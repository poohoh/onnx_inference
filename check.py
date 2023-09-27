import onnx

onnx_model = onnx.load("checkpoint/45.onnx")
onnx.checker.check_model(onnx_model)