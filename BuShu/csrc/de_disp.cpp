// de_disp.cpp
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
using torch::autograd::Function;
using torch::autograd::AutogradContext;
using variable_list = std::vector<at::Tensor>;
// 前向实现
at::Tensor de_disp_impl_npu(const at::Tensor& self, const at::Tensor& other) {
    // 创建输出内存
    at::Tensor result = at::empty_like(self, other);
    // 调用aclnn接口计算
    EXEC_NPU_CMD(aclnnDeDisp, self, other, result);
    return result;
}
// 反向实现
std::tuple<at::Tensor> de_disp_backward_impl_npu(const at::Tensor& grad) {
    at::Tensor result = grad; // 创建输出内存
    return {result};
}
// 通过继承torch::autograd::Function类实现前反向绑定
class DeDispFunction : public torch::autograd::Function<DeDispFunction> {
    public:
        static at::Tensor forward(AutogradContext *ctx, at::Tensor self, at::Tensor other) {
            at::AutoDispatchBelowADInplaceOrView guard;
            static auto op = torch::Dispatcher::singleton()
                            .findSchemaOrThrow("myops::de_disp", "")
                            .typed<decltype(de_disp_impl_npu)>();
            auto result = op.call(self, other);
            return result;
        }
        static variable_list backward(AutogradContext *ctx, variable_list grad_outputs) {
            auto grad_output = grad_outputs[0];
            static auto op = torch::Dispatcher::singleton()
                          .findSchemaOrThrow("myops::de_disp_backward", "")
                          .typed<decltype(de_disp_backward_impl_npu)>();
            auto result = op.call(grad_output);
            return {std::get<0>(result)};
        }
};
// 使用的时候调用apply()方法
at::Tensor de_disp_autograd(const at::Tensor& self, const at::Tensor& other) {
    return DeDispFunction::apply(self);
}
// 为NPU设备注册前反向实现
// NPU设备在pytorch 2.1及以上版本使用的设备名称是PrivateUse1，在2.1以下版本用的是XLA，如果是2.1以下版本PrivateUse1需要改成XLA
TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("de_disp", &de_disp_impl_npu);
    m.impl("de_disp_backward", &de_disp_backward_impl_npu);
}
// 给op绑定NPU的自动求导实现
// 如果是pytorch 2.1以下的版本，AutogradPrivateUse1需要改成AutogradXLA
TORCH_LIBRARY_IMPL(myops, AutogradPrivateUse1, m) {
    m.impl("de_disp", &de_disp_autograd);
}
