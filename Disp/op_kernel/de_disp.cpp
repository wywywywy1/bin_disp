#include "kernel_operator.h"
#include <cstdio>

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;
template<typename DTYPE>
class KernelDedisp {
public:
    __aicore__ inline KernelDedisp() {
    }
    __aicore__ inline void Init(GM_ADDR freq, GM_ADDR xTeamy, GM_ADDR outfreq, uint32_t totalLength, uint32_t tileNum, 
                                float time_reso, float down_time_rate, float xTeam, float y, float freq1) {

        this->time_reso = time_reso;
        this->down_time_rate = down_time_rate;
        this->xTeam = xTeam;
        this->y = y;
        this->freq1 = freq1;
        ASSERT(totalLength % GetBlockNum() == 0);
        // 使用获取到的TilingData计算得到singleCoreSize(每个核上总计算数据大小)、tileNum（每个核上分块个数）、singleTileLength（每个分块大小）等变量
        this->blockLength = totalLength / GetBlockNum(); //总长度除以分块数目可以得到每一块的长度，即每个核要计算的数据的大小
        this->tileNum = tileNum;

        ASSERT(this->blockLength % (tileNum * BUFFER_NUM) == 0);
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM; //每个核计算数据大小除以每个核的分块数除以buffer数目，即每个核中每个buffer的大小
        
        // 获取当前核的起始索引
        ASSERT((blockLength * (GetBlockIdx() + 1)) <= totalLength);
        freqGm.SetGlobalBuffer((__gm__ DTYPE*)freq + this->blockLength * GetBlockIdx(), this->blockLength);
        xTeamyGm.SetGlobalBuffer((__gm__ DTYPE*)xTeamy + this->blockLength * GetBlockIdx(), this->blockLength);
        outfreqGm.SetGlobalBuffer((__gm__ DTYPE*)outfreq + this->blockLength * GetBlockIdx(), this->blockLength);
        // 通过Pipe内存管理对象为输入输出Queue分配内存
        pipe.InitBuffer(inQueuefreq, BUFFER_NUM, this->tileLength * sizeof(DTYPE));
        pipe.InitBuffer(inQueuexTeamy, BUFFER_NUM, this->tileLength * sizeof(DTYPE));
        pipe.InitBuffer(outQueueoutfreq, BUFFER_NUM, this->tileLength * sizeof(DTYPE));
        ASSERT(tmpBuffer1.GetSize() >= tileLength * sizeof(DTYPE));
        ASSERT(tmpBuffer2.GetSize() >= tileLength * sizeof(DTYPE));
        ASSERT(tmpBuffer3.GetSize() >= tileLength * sizeof(DTYPE));
        ASSERT(tmpBuffer4.GetSize() >= tileLength * sizeof(DTYPE));
        ASSERT(tmpBuffer5.GetSize() >= tileLength * sizeof(DTYPE));
        pipe.InitBuffer(tmpBuffer1, this->tileLength * sizeof(DTYPE));
        pipe.InitBuffer(tmpBuffer2, this->tileLength * sizeof(DTYPE));
        pipe.InitBuffer(tmpBuffer3, this->tileLength * sizeof(DTYPE));
        pipe.InitBuffer(tmpBuffer4, this->tileLength * sizeof(DTYPE));
        pipe.InitBuffer(tmpBuffer5, this->tileLength * sizeof(DTYPE));
    }
    // 核心处理函数，实现算子逻辑，调用私有成员函数CopyIn、Compute、CopyOut完成矢量算子的三级流水操作
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    // 搬入函数，完成CopyIn阶段的处理，被核心Process函数调用
    __aicore__ inline void CopyIn(int32_t progress)
    {
        // 从Queue中分配输入Tensor
        LocalTensor<DTYPE> freqLocal = inQueuefreq.AllocTensor<DTYPE>();
        LocalTensor<DTYPE> xTeamyLocal = inQueuexTeamy.AllocTensor<DTYPE>();
        // 将GlobalTensor数据拷贝到LocalTensor
        DataCopy(freqLocal, freqGm[progress * this->tileLength], this->tileLength);
        DataCopy(xTeamyLocal, xTeamyGm[progress * this->tileLength], this->tileLength);
        // 将LocalTesor放入VECIN（代表矢量编程中搬入数据的逻辑存放位置）的Queue中
        inQueuefreq.EnQue(freqLocal);
        inQueuexTeamy.EnQue(xTeamyLocal);
    }
    // 计算函数，完成Compute阶段的处理，被核心Process函数调用
    __aicore__ inline void Compute(int32_t progress)
    {
        // 将Tensor从队列中取出，用于后续计算
        LocalTensor<DTYPE> freqLocal = inQueuefreq.DeQue<DTYPE>();
        LocalTensor<DTYPE> xTeamyLocal = inQueuexTeamy.DeQue<DTYPE>();
        // 从Queue中分配输出Tensor
        LocalTensor<DTYPE> outfreqLocal = outQueueoutfreq.AllocTensor<DTYPE>();
        // 调用接口进行计算        
        // 核心计算逻辑
        LocalTensor<DTYPE> tmpTensor1 = tmpBuffer1.Get<DTYPE>();
        LocalTensor<DTYPE> tmpTensor2 = tmpBuffer2.Get<DTYPE>();
        LocalTensor<DTYPE> tmpTensor3 = tmpBuffer3.Get<DTYPE>();
        LocalTensor<DTYPE> tmpTensor4 = tmpBuffer4.Get<DTYPE>();
        LocalTensor<DTYPE> tmpTensor5 = tmpBuffer5.Get<DTYPE>();
        //*
        ASSERT(time_reso != 0.0f);
        ASSERT(down_time_rate != 0);

        float inputVal1 = -this->freq1;
        float inputVal0 = xTeamyLocal.GetValue(0);
        float inputVal2 = 1 / this->time_reso;
        float inputVal3 = 1 / this->down_time_rate;
        float inputValy = xTeamyLocal.GetValue(1);

        // AscendC::DumpTensor(freqLocal,5, this->tileLength);
        Adds(tmpTensor1, freqLocal, inputVal1, this->tileLength);
        Muls(tmpTensor2, tmpTensor1, inputVal0, this->tileLength);
        Muls(tmpTensor3, tmpTensor2, inputVal2, this->tileLength);
        Muls(tmpTensor4, tmpTensor3, inputVal3, this->tileLength);
        Adds(outfreqLocal, tmpTensor4, inputValy, tileLength);
    
        //打印最后计算结果的tensor信息，维度以及tensor的内容
        // AscendC::DumpTensor(outfreqLocal,5, this->tileLength); 
        
        uint32_t offset = progress * this->tileLength;

        ASSERT(offset + this->tileLength <= this->blockLength);

        // DataCopy(outfreqLocal, freqLocal, this->tileLength);
        // 将计算结果LocalTensor放入到VecOut的Queue中
        outQueueoutfreq.EnQue<DTYPE>(outfreqLocal);
        // 释放输入Tensor
        inQueuefreq.FreeTensor(freqLocal);
    }
    // 搬出函数，完成CopyOut阶段的处理，被核心Process函数调用
    __aicore__ inline void CopyOut(int32_t progress)
    {
        // 从VecOut的Queue中取出输出Tensor
        LocalTensor<DTYPE> outfreqLocal = outQueueoutfreq.DeQue<DTYPE>();
        // 将输出Tensor拷贝到GlobalTensor中
        DataCopy(outfreqGm[progress * this->tileLength], outfreqLocal, this->tileLength);
        // 将不再使用的LocalTensor释放
        outQueueoutfreq.FreeTensor(outfreqLocal);
    }

private:
    //Pipe内存管理对象
    TPipe pipe;
    //输入数据Queue队列管理对象，QuePosition为VECIN
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueuefreq; 
    //输出数据Queue队列管理对象，QuePosition为VECOUT
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueoutfreq;
    //管理输入输出Global Memory内存地址的对象，其中xGm, yGm为输入，zGm为输出
    GlobalTensor<DTYPE> freqGm;
    GlobalTensor<DTYPE> outfreqGm;
    TBuf<QuePosition::VECCALC> tmpBuffer1, tmpBuffer2, tmpBuffer3, tmpBuffer4, tmpBuffer5;
    // 每个核上总计算数据大小
    uint32_t blockLength;
    // 每个核上总计算数据分块个数
    uint32_t tileNum;
    // 每个分块大小
    uint32_t tileLength;
    float time_reso;
    float down_time_rate;
    float xTeam;
    float y;
    float freq1;
};

extern "C" __global__ __aicore__ void de_disp(GM_ADDR freq, GM_ADDR outfreq, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelDedisp<float> op;
    op.Init(freq, outfreq, tiling_data.totalLength, tiling_data.tileNum,
         tiling_data.time_reso, tiling_data.down_time_rate, tiling_data.xTeam, tiling_data.y, tiling_data.freq1);
    op.Process();
    // for (int i = 0; i < 100000; i++) {
    //     op.Process();
    // }
}