// registration.cpp
#include <torch/library.h>
#include <torch/extension.h>
#include "function.h"
TORCH_LIBRARY(myops, m) {
    m.def("de_disp(Tensor self, Tensor other) -> Tensor");       // 注册正向接口
    m.def("de_disp_backward(Tensor self) -> (Tensor, Tensor)");  // 注册反向接口
}

// 通过pybind将c++接口和python接口绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("de_disp", &de_disp_autograd, "disp"); // 其中de_disp为python侧调用的名称
}
