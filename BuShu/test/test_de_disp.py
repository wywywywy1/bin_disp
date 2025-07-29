import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import custom_ops
import copy
import inspect
import os
import numpy as np
import threading

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)

def get_function_source(func):
    module = inspect.getmodule(func)
    file_path = inspect.getfile(func)
    source_lines = inspect.getsourcelines(func)
    print(f"模块: {module.__name__}")
    print(f"文件路径: {file_path}")
    print(f"行号: {source_lines[1]}")
    return module, file_path, source_lines

class TestDeDisp(TestCase):
    def test_de_disp(self):
        time_reso = 4.9152e-05
        down_time_rate = 4.0
        freq = 1.0
        DM = 1.0
        y = 1.0

        x_cpu = torch.randn([512, 1], dtype=torch.float32)
        xTeamy_cpu = torch.tensor([[4150.0], [1.0]])
        x_npu = copy.deepcopy(x_cpu).npu()
        xTeamy_npu = copy.deepcopy(xTeamy_cpu).npu()
        x_cpu.requires_grad = True
        x_npu.requires_grad = True
        xTeamy_cpu.requires_grad = True
        xTeamy_npu.requires_grad = True

        # calculate on npu
        # get_function_source(custom_ops.de_disp)
        # print(dir(custom_ops))
        # callable_items = [item for item in dir(custom_ops) if callable(getattr(custom_ops, item))]
        # print("custom_ops模块中的可调用对象:", callable_items)
        output = custom_ops.de_disp(x_npu, xTeamy_npu)
        # output.backward(output)

        # calculate on cpu
        # get_function_source(torch.de_disp)
        # print(dir(torch))
        # callable_items = [item for item in dir(torch) if callable(getattr(torch, item))]
        # print("torch模块中的可调用对象:", callable_items)
        # cpuout = torch.de_disp(x_cpu)
        # cpuout.backward(cpuout)
        cpuout = torch.randn([512, 1], dtype=torch.float32)
        for i in range(512):
            x = x_cpu[i, 0]
            cpuout[i, 0] = 4.15 * DM * (x - freq**(-2)) * 1e3 / time_reso / down_time_rate + y
        # cpuout.backward(cpuout)

        # compare result
        self.assertRtolEqual(output, cpuout)
        # self.assertRtolEqual(x_npu.grad, x_cpu.grad)

    # def test_de_disp_meta(self):
    #     input1 = torch.randn([512, 1], dtype=torch.float32)

    #     x_input1 = input1.to("meta")
    #     x_input1.requires_grad = True
    #     custom_out = custom_ops.de_disp(x_input1)
    #     custom_out.backward(custom_out)

    #     x_input2 = input1.to("meta")
    #     x_input2.requires_grad = True
    #     cpuout = torch.dedisp(x_input2)
    #     cpuout.backward(cpuout)

    #     self.assertTrue(custom_out.is_meta)
    #     self.assertRtolEqual(custom_out.size(), cpuout.size())
    #     self.assertRtolEqual(x_input1.grad.size())

if __name__ == "__main__":
    run_tests()

