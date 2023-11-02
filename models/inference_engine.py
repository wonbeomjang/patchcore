import numpy as np
import onnxruntime
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt
import torch

from config import PatchCoreConfig, InferenceEngine


class PatchCoreEngine:
    def __init__(
        self, patch_core_config: PatchCoreConfig = PatchCoreConfig(), *args, **kwargs
    ):
        if patch_core_config.backbone.inference_engin == InferenceEngine.onnx:
            self.load_onnx(patch_core_config)
            self._predict = self._predict_onnx
        elif patch_core_config.backbone.inference_engin == InferenceEngine.trt:
            self.load_tenosrrt(patch_core_config)
            self.alloc_buf()
            self._predict = self._predict_trt
        else:
            raise ValueError("Unsupported Inference Engine")

    def __call__(
        self, x: torch.Tensor | np.ndarray, *args, **kwargs
    ) -> dict[str, np.ndarray]:
        if isinstance(x, torch.Tensor):
            x: np.ndarray = x.detach().cpu().numpy()
        assert isinstance(x, np.ndarray)

        score = self._predict(x)

        return {"score": score}

    def _predict_onnx(self, batch: np.ndarray):
        ort_inputs = {self.onnx.get_inputs()[0].name: batch}
        return self.onnx.run(None, ort_inputs)[0]

    def _predict_trt(self, image: np.ndarray):  # result gets copied into output
        cuda.memcpy_htod(self.inputs[0]["allocation"], image)
        self.context.execute_v2(self.allocations)
        for o in range(len(self.outputs)):
            cuda.memcpy_dtoh(
                self.outputs[0]["host_allocation"], self.outputs[o]["allocation"]
            )

        return self.outputs[0]["host_allocation"]

    def load_onnx(self, patch_core_config):
        self.onnx = onnxruntime.InferenceSession(patch_core_config.backbone.onnx_path)

    def load_tenosrrt(self, patch_core_config):
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        trt.init_libnvinfer_plugins(None, "")
        with open(patch_core_config.backbone.trt_path, "rb") as f:
            self.engine: trt.ICudaEngine = runtime.deserialize_cuda_engine(f.read())
        self.context: trt.IExecutionContext = self.engine.create_execution_context()

        assert self.engine
        assert self.context
        self.bindings = None

    def alloc_buf(self):
        self.inputs = []
        self.outputs = []
        self.allocations = []

        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True

            name = self.engine.get_binding_name(i)
            dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(i)))
            shape = self.context.get_binding_shape(i)

            if is_input and shape[0] < 0:
                assert self.engine.num_optimization_profiles > 0
                profile_shape = self.engine.get_profile_shape(0, name)
                assert len(profile_shape) == 3

                self.context.set_binding_shape(i, profile_shape[2])
                shape = self.context.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = dtype.itemsize
            for s in shape:
                size *= s

            allocation = cuda.mem_alloc(size)
            host_allocation = None if is_input else np.zeros(shape, dtype)
            binding = {
                "index": i,
                "name": name,
                "dtype": dtype,
                "shape": list(shape),
                "allocation": allocation,
                "host_allocation": host_allocation,
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

    def eval(self):
        pass

    def train(self):
        pass
