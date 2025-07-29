
#include "de_disp_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  DeDispTilingData tiling;
  tiling.set_freq1(1.0);
  tiling.set_y(1.0);
  tiling.set_xTeam(4150.0);
  tiling.set_time_reso(4.9152e-05);
  tiling.set_down_time_rate(4.0);
  const uint32_t BLOCK_DIM = 8;
  const uint32_t TILE_NUM = 8; //送进AIcore的数据分多少小块
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize(); //根据上下文获取输入输出shape信息
  int32_t data_sz = 1;
  for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
    data_sz *= x1_shape->GetStorageShape().GetDim(i);
  // tiling.set_size(data_sz);
  tiling.set_totalLength(totalLength);
  tiling.set_tileNum(TILE_NUM);
  context->SetBlockDim(BLOCK_DIM);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class DeDisp : public OpDef {
public:
    explicit DeDisp(const char* name) : OpDef(name)
    {
        this->Input("freq")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("xTeamy")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("outfreq")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310p");

    }
};

OP_ADD(DeDisp);
}
