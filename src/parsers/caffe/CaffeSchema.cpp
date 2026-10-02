// SPDX-License-Identifier: Apache-2.0
// parsers/caffe/CaffeSchema.cpp — constexpr tables for the caffe.proto subset
// NetVis decodes. Every number below is verbatim from upstream BVLC/caffe
// src/caffe/proto/caffe.proto (master); see CaffeSchema.h for how the tables
// are used. Order inside a table is the .proto declaration order where that
// matters for readability only — lookups never depend on it.
#include "parsers/caffe/CaffeSchema.h"

#include <iterator>

namespace netvis::caffe {
namespace {

using K = FieldKind;

constexpr FieldSpec S(uint32_t n, const char* name, FieldKind k, bool rep = false) {
  return FieldSpec{n, name, k, rep, nullptr, nullptr};
}
constexpr FieldSpec E(uint32_t n, const char* name, const EnumSpec* e, bool rep = false) {
  return FieldSpec{n, name, FieldKind::Enum, rep, e, nullptr};
}
constexpr FieldSpec M(uint32_t n, const char* name, const MessageSpec* m, bool rep = false) {
  return FieldSpec{n, name, FieldKind::Message, rep, nullptr, m};
}

// ---- enums -------------------------------------------------------------------
constexpr EnumValue kPhaseV[] = {{0, "TRAIN"}, {1, "TEST"}};
constexpr EnumSpec kPhase = {"Phase", kPhaseV, std::size(kPhaseV)};

constexpr EnumValue kPoolMethodV[] = {{0, "MAX"}, {1, "AVE"}, {2, "STOCHASTIC"}};
constexpr EnumSpec kPoolMethod = {"PoolMethod", kPoolMethodV, std::size(kPoolMethodV)};

constexpr EnumValue kRoundModeV[] = {{0, "CEIL"}, {1, "FLOOR"}};
constexpr EnumSpec kRoundMode = {"RoundMode", kRoundModeV, std::size(kRoundModeV)};

constexpr EnumValue kEngineV[] = {{0, "DEFAULT"}, {1, "CAFFE"}, {2, "CUDNN"}};
constexpr EnumSpec kEngine = {"Engine", kEngineV, std::size(kEngineV)};

constexpr EnumValue kEltwiseOpV[] = {{0, "PROD"}, {1, "SUM"}, {2, "MAX"}};
constexpr EnumSpec kEltwiseOp = {"EltwiseOp", kEltwiseOpV, std::size(kEltwiseOpV)};

constexpr EnumValue kNormRegionV[] = {{0, "ACROSS_CHANNELS"}, {1, "WITHIN_CHANNEL"}};
constexpr EnumSpec kNormRegion = {"NormRegion", kNormRegionV, std::size(kNormRegionV)};

constexpr EnumValue kReductionOpV[] = {{1, "SUM"}, {2, "ASUM"}, {3, "SUMSQ"}, {4, "MEAN"}};
constexpr EnumSpec kReductionOp = {"ReductionOp", kReductionOpV, std::size(kReductionOpV)};

constexpr EnumValue kVarianceNormV[] = {{0, "FAN_IN"}, {1, "FAN_OUT"}, {2, "AVERAGE"}};
constexpr EnumSpec kVarianceNorm = {"VarianceNorm", kVarianceNormV, std::size(kVarianceNormV)};

constexpr EnumValue kDimCheckModeV[] = {{0, "STRICT"}, {1, "PERMISSIVE"}};
constexpr EnumSpec kDimCheckMode = {"DimCheckMode", kDimCheckModeV, std::size(kDimCheckModeV)};

// V1LayerParameter.LayerType (numbers from caffe.proto; modern strings from
// UpgradeV1LayerType). The two arrays are index-aligned with the enum number.
constexpr EnumValue kV1TypeV[] = {
    {0, "NONE"}, {1, "ACCURACY"}, {2, "BNLL"}, {3, "CONCAT"}, {4, "CONVOLUTION"},
    {5, "DATA"}, {6, "DROPOUT"}, {7, "EUCLIDEAN_LOSS"}, {8, "FLATTEN"},
    {9, "HDF5_DATA"}, {10, "HDF5_OUTPUT"}, {11, "IM2COL"}, {12, "IMAGE_DATA"},
    {13, "INFOGAIN_LOSS"}, {14, "INNER_PRODUCT"}, {15, "LRN"},
    {16, "MULTINOMIAL_LOGISTIC_LOSS"}, {17, "POOLING"}, {18, "RELU"},
    {19, "SIGMOID"}, {20, "SOFTMAX"}, {21, "SOFTMAX_LOSS"}, {22, "SPLIT"},
    {23, "TANH"}, {24, "WINDOW_DATA"}, {25, "ELTWISE"}, {26, "POWER"},
    {27, "SIGMOID_CROSS_ENTROPY_LOSS"}, {28, "HINGE_LOSS"}, {29, "MEMORY_DATA"},
    {30, "ARGMAX"}, {31, "THRESHOLD"}, {32, "DUMMY_DATA"}, {33, "SLICE"},
    {34, "MVN"}, {35, "ABSVAL"}, {36, "SILENCE"}, {37, "CONTRASTIVE_LOSS"},
    {38, "EXP"}, {39, "DECONVOLUTION"},
};
constexpr EnumSpec kV1Type = {"LayerType", kV1TypeV, std::size(kV1TypeV)};

constexpr const char* kV1Modern[] = {
    "",                        // NONE
    "Accuracy", "BNLL", "Concat", "Convolution", "Data", "Dropout",
    "EuclideanLoss", "Flatten", "HDF5Data", "HDF5Output", "Im2col", "ImageData",
    "InfogainLoss", "InnerProduct", "LRN", "MultinomialLogisticLoss", "Pooling",
    "ReLU", "Sigmoid", "Softmax", "SoftmaxWithLoss", "Split", "TanH",
    "WindowData", "Eltwise", "Power", "SigmoidCrossEntropyLoss", "HingeLoss",
    "MemoryData", "ArgMax", "Threshold", "DummyData", "Slice", "MVN", "AbsVal",
    "Silence", "ContrastiveLoss", "Exp", "Deconvolution",
};
static_assert(std::size(kV1Modern) == std::size(kV1TypeV), "V1 tables out of step");
static_assert(std::size(kV1Modern) == static_cast<size_t>(kMaxV1LayerType) + 1,
              "kMaxV1LayerType out of step");

// ---- shared messages -------------------------------------------------------------
constexpr FieldSpec kBlobShapeF[] = {S(1, "dim", K::Int64, true)};
constexpr MessageSpec kBlobShape = {"BlobShape", kBlobShapeF, std::size(kBlobShapeF)};

constexpr FieldSpec kNetStateF[] = {
    E(1, "phase", &kPhase), S(2, "level", K::Int32), S(3, "stage", K::String, true)};
constexpr MessageSpec kNetState = {"NetState", kNetStateF, std::size(kNetStateF)};

constexpr FieldSpec kNetStateRuleF[] = {
    E(1, "phase", &kPhase), S(2, "min_level", K::Int32), S(3, "max_level", K::Int32),
    S(4, "stage", K::String, true), S(5, "not_stage", K::String, true)};
constexpr MessageSpec kNetStateRule = {"NetStateRule", kNetStateRuleF,
                                       std::size(kNetStateRuleF)};

constexpr FieldSpec kParamSpecF[] = {
    S(1, "name", K::String), E(2, "share_mode", &kDimCheckMode),
    S(3, "lr_mult", K::Float), S(4, "decay_mult", K::Float)};
constexpr MessageSpec kParamSpec = {"ParamSpec", kParamSpecF, std::size(kParamSpecF)};

constexpr FieldSpec kFillerF[] = {
    S(1, "type", K::String), S(2, "value", K::Float), S(3, "min", K::Float),
    S(4, "max", K::Float), S(5, "mean", K::Float), S(6, "std", K::Float),
    S(7, "sparse", K::Int32), E(8, "variance_norm", &kVarianceNorm)};
constexpr MessageSpec kFiller = {"FillerParameter", kFillerF, std::size(kFillerF)};

// ---- *Parameter messages ---------------------------------------------------------
constexpr FieldSpec kArgMaxF[] = {
    S(1, "out_max_val", K::Bool), S(2, "top_k", K::UInt32), S(3, "axis", K::Int32)};
constexpr MessageSpec kArgMax = {"ArgMaxParameter", kArgMaxF, std::size(kArgMaxF)};

constexpr FieldSpec kBatchNormF[] = {
    S(1, "use_global_stats", K::Bool), S(2, "moving_average_fraction", K::Float),
    S(3, "eps", K::Float)};
constexpr MessageSpec kBatchNorm = {"BatchNormParameter", kBatchNormF, std::size(kBatchNormF)};

constexpr FieldSpec kBiasF[] = {
    S(1, "axis", K::Int32), S(2, "num_axes", K::Int32), M(3, "filler", &kFiller)};
constexpr MessageSpec kBias = {"BiasParameter", kBiasF, std::size(kBiasF)};

constexpr FieldSpec kClipF[] = {S(1, "min", K::Float), S(2, "max", K::Float)};
constexpr MessageSpec kClip = {"ClipParameter", kClipF, std::size(kClipF)};

constexpr FieldSpec kConcatF[] = {S(1, "concat_dim", K::UInt32), S(2, "axis", K::Int32)};
constexpr MessageSpec kConcat = {"ConcatParameter", kConcatF, std::size(kConcatF)};

constexpr FieldSpec kConvolutionF[] = {
    S(1, "num_output", K::UInt32),     S(2, "bias_term", K::Bool),
    S(3, "pad", K::UInt32, true),      S(4, "kernel_size", K::UInt32, true),
    S(5, "group", K::UInt32),          S(6, "stride", K::UInt32, true),
    M(7, "weight_filler", &kFiller),   M(8, "bias_filler", &kFiller),
    S(9, "pad_h", K::UInt32),          S(10, "pad_w", K::UInt32),
    S(11, "kernel_h", K::UInt32),      S(12, "kernel_w", K::UInt32),
    S(13, "stride_h", K::UInt32),      S(14, "stride_w", K::UInt32),
    E(15, "engine", &kEngine),         S(16, "axis", K::Int32),
    S(17, "force_nd_im2col", K::Bool), S(18, "dilation", K::UInt32, true)};
constexpr MessageSpec kConvolution = {"ConvolutionParameter", kConvolutionF,
                                      std::size(kConvolutionF)};

constexpr FieldSpec kCropF[] = {S(1, "axis", K::Int32), S(2, "offset", K::UInt32, true)};
constexpr MessageSpec kCrop = {"CropParameter", kCropF, std::size(kCropF)};

constexpr FieldSpec kDummyDataF[] = {
    M(1, "data_filler", &kFiller, true), S(2, "num", K::UInt32, true),
    S(3, "channels", K::UInt32, true),   S(4, "height", K::UInt32, true),
    S(5, "width", K::UInt32, true),      M(6, "shape", &kBlobShape, true)};
constexpr MessageSpec kDummyData = {"DummyDataParameter", kDummyDataF, std::size(kDummyDataF)};

constexpr FieldSpec kDropoutF[] = {S(1, "dropout_ratio", K::Float)};
constexpr MessageSpec kDropout = {"DropoutParameter", kDropoutF, std::size(kDropoutF)};

constexpr FieldSpec kEltwiseF[] = {
    E(1, "operation", &kEltwiseOp), S(2, "coeff", K::Float, true),
    S(3, "stable_prod_grad", K::Bool)};
constexpr MessageSpec kEltwise = {"EltwiseParameter", kEltwiseF, std::size(kEltwiseF)};

constexpr FieldSpec kELUF[] = {S(1, "alpha", K::Float)};
constexpr MessageSpec kELU = {"ELUParameter", kELUF, std::size(kELUF)};

constexpr FieldSpec kEmbedF[] = {
    S(1, "num_output", K::UInt32), S(2, "input_dim", K::UInt32), S(3, "bias_term", K::Bool),
    M(4, "weight_filler", &kFiller), M(5, "bias_filler", &kFiller)};
constexpr MessageSpec kEmbed = {"EmbedParameter", kEmbedF, std::size(kEmbedF)};

constexpr FieldSpec kExpF[] = {
    S(1, "base", K::Float), S(2, "scale", K::Float), S(3, "shift", K::Float)};
constexpr MessageSpec kExp = {"ExpParameter", kExpF, std::size(kExpF)};

constexpr FieldSpec kFlattenF[] = {S(1, "axis", K::Int32), S(2, "end_axis", K::Int32)};
constexpr MessageSpec kFlatten = {"FlattenParameter", kFlattenF, std::size(kFlattenF)};

constexpr FieldSpec kInnerProductF[] = {
    S(1, "num_output", K::UInt32), S(2, "bias_term", K::Bool),
    M(3, "weight_filler", &kFiller), M(4, "bias_filler", &kFiller),
    S(5, "axis", K::Int32), S(6, "transpose", K::Bool)};
constexpr MessageSpec kInnerProduct = {"InnerProductParameter", kInnerProductF,
                                       std::size(kInnerProductF)};

constexpr FieldSpec kInputF[] = {M(1, "shape", &kBlobShape, true)};
constexpr MessageSpec kInput = {"InputParameter", kInputF, std::size(kInputF)};

constexpr FieldSpec kLogF[] = {
    S(1, "base", K::Float), S(2, "scale", K::Float), S(3, "shift", K::Float)};
constexpr MessageSpec kLog = {"LogParameter", kLogF, std::size(kLogF)};

constexpr FieldSpec kLRNF[] = {
    S(1, "local_size", K::UInt32), S(2, "alpha", K::Float), S(3, "beta", K::Float),
    E(4, "norm_region", &kNormRegion), S(5, "k", K::Float), E(6, "engine", &kEngine)};
constexpr MessageSpec kLRN = {"LRNParameter", kLRNF, std::size(kLRNF)};

constexpr FieldSpec kMVNF[] = {
    S(1, "normalize_variance", K::Bool), S(2, "across_channels", K::Bool),
    S(3, "eps", K::Float)};
constexpr MessageSpec kMVN = {"MVNParameter", kMVNF, std::size(kMVNF)};

constexpr FieldSpec kParameterF[] = {M(1, "shape", &kBlobShape)};
constexpr MessageSpec kParameter = {"ParameterParameter", kParameterF, std::size(kParameterF)};

constexpr FieldSpec kPoolingF[] = {
    E(1, "pool", &kPoolMethod),     S(2, "kernel_size", K::UInt32),
    S(3, "stride", K::UInt32),      S(4, "pad", K::UInt32),
    S(5, "kernel_h", K::UInt32),    S(6, "kernel_w", K::UInt32),
    S(7, "stride_h", K::UInt32),    S(8, "stride_w", K::UInt32),
    S(9, "pad_h", K::UInt32),       S(10, "pad_w", K::UInt32),
    E(11, "engine", &kEngine),      S(12, "global_pooling", K::Bool),
    E(13, "round_mode", &kRoundMode)};
constexpr MessageSpec kPooling = {"PoolingParameter", kPoolingF, std::size(kPoolingF)};

constexpr FieldSpec kPowerF[] = {
    S(1, "power", K::Float), S(2, "scale", K::Float), S(3, "shift", K::Float)};
constexpr MessageSpec kPower = {"PowerParameter", kPowerF, std::size(kPowerF)};

constexpr FieldSpec kPReLUF[] = {M(1, "filler", &kFiller), S(2, "channel_shared", K::Bool)};
constexpr MessageSpec kPReLU = {"PReLUParameter", kPReLUF, std::size(kPReLUF)};

constexpr FieldSpec kRecurrentF[] = {
    S(1, "num_output", K::UInt32), M(2, "weight_filler", &kFiller),
    M(3, "bias_filler", &kFiller), S(4, "debug_info", K::Bool),
    S(5, "expose_hidden", K::Bool)};
constexpr MessageSpec kRecurrent = {"RecurrentParameter", kRecurrentF, std::size(kRecurrentF)};

constexpr FieldSpec kReductionF[] = {
    E(1, "operation", &kReductionOp), S(2, "axis", K::Int32), S(3, "coeff", K::Float)};
constexpr MessageSpec kReduction = {"ReductionParameter", kReductionF, std::size(kReductionF)};

constexpr FieldSpec kReLUF[] = {S(1, "negative_slope", K::Float), E(2, "engine", &kEngine)};
constexpr MessageSpec kReLU = {"ReLUParameter", kReLUF, std::size(kReLUF)};

constexpr FieldSpec kReshapeF[] = {
    M(1, "shape", &kBlobShape), S(2, "axis", K::Int32), S(3, "num_axes", K::Int32)};
constexpr MessageSpec kReshape = {"ReshapeParameter", kReshapeF, std::size(kReshapeF)};

constexpr FieldSpec kScaleF[] = {
    S(1, "axis", K::Int32), S(2, "num_axes", K::Int32), M(3, "filler", &kFiller),
    S(4, "bias_term", K::Bool), M(5, "bias_filler", &kFiller)};
constexpr MessageSpec kScale = {"ScaleParameter", kScaleF, std::size(kScaleF)};

constexpr FieldSpec kSigmoidF[] = {E(1, "engine", &kEngine)};
constexpr MessageSpec kSigmoid = {"SigmoidParameter", kSigmoidF, std::size(kSigmoidF)};

constexpr FieldSpec kSliceF[] = {
    S(1, "slice_dim", K::UInt32), S(2, "slice_point", K::UInt32, true), S(3, "axis", K::Int32)};
constexpr MessageSpec kSlice = {"SliceParameter", kSliceF, std::size(kSliceF)};

constexpr FieldSpec kSoftmaxF[] = {E(1, "engine", &kEngine), S(2, "axis", K::Int32)};
constexpr MessageSpec kSoftmax = {"SoftmaxParameter", kSoftmaxF, std::size(kSoftmaxF)};

constexpr FieldSpec kSPPF[] = {
    S(1, "pyramid_height", K::UInt32), E(2, "pool", &kPoolMethod), E(6, "engine", &kEngine)};
constexpr MessageSpec kSPP = {"SPPParameter", kSPPF, std::size(kSPPF)};

constexpr FieldSpec kSwishF[] = {S(1, "beta", K::Float)};
constexpr MessageSpec kSwish = {"SwishParameter", kSwishF, std::size(kSwishF)};

constexpr FieldSpec kTanHF[] = {E(1, "engine", &kEngine)};
constexpr MessageSpec kTanH = {"TanHParameter", kTanHF, std::size(kTanHF)};

constexpr FieldSpec kThresholdF[] = {S(1, "threshold", K::Float)};
constexpr MessageSpec kThreshold = {"ThresholdParameter", kThresholdF, std::size(kThresholdF)};

constexpr FieldSpec kTileF[] = {S(1, "axis", K::Int32), S(2, "tiles", K::Int32)};
constexpr MessageSpec kTile = {"TileParameter", kTileF, std::size(kTileF)};

// ---- LayerParameter (modern, NetParameter.layer = 100) ----------------------------
// Slots whose message is not decoded (transform, loss, accuracy, data families,
// python, ...) carry message == nullptr: binary shows "(not decoded)", text
// flattens them generically.
constexpr FieldSpec kLayerF[] = {
    S(1, "name", K::String),
    S(2, "type", K::String),
    S(3, "bottom", K::String, true),
    S(4, "top", K::String, true),
    S(5, "loss_weight", K::Float, true),
    M(6, "param", &kParamSpec, true),
    M(7, "blobs", nullptr, true),
    M(8, "include", &kNetStateRule, true),
    M(9, "exclude", &kNetStateRule, true),
    E(10, "phase", &kPhase),
    S(11, "propagate_down", K::Bool, true),
    M(100, "transform_param", nullptr),
    M(101, "loss_param", nullptr),
    M(102, "accuracy_param", nullptr),
    M(103, "argmax_param", &kArgMax),
    M(104, "concat_param", &kConcat),
    M(105, "contrastive_loss_param", nullptr),
    M(106, "convolution_param", &kConvolution),
    M(107, "data_param", nullptr),
    M(108, "dropout_param", &kDropout),
    M(109, "dummy_data_param", &kDummyData),
    M(110, "eltwise_param", &kEltwise),
    M(111, "exp_param", &kExp),
    M(112, "hdf5_data_param", nullptr),
    M(113, "hdf5_output_param", nullptr),
    M(114, "hinge_loss_param", nullptr),
    M(115, "image_data_param", nullptr),
    M(116, "infogain_loss_param", nullptr),
    M(117, "inner_product_param", &kInnerProduct),
    M(118, "lrn_param", &kLRN),
    M(119, "memory_data_param", nullptr),
    M(120, "mvn_param", &kMVN),
    M(121, "pooling_param", &kPooling),
    M(122, "power_param", &kPower),
    M(123, "relu_param", &kReLU),
    M(124, "sigmoid_param", &kSigmoid),
    M(125, "softmax_param", &kSoftmax),
    M(126, "slice_param", &kSlice),
    M(127, "tanh_param", &kTanH),
    M(128, "threshold_param", &kThreshold),
    M(129, "window_data_param", nullptr),
    M(130, "python_param", nullptr),
    M(131, "prelu_param", &kPReLU),
    M(132, "spp_param", &kSPP),
    M(133, "reshape_param", &kReshape),
    M(134, "log_param", &kLog),
    M(135, "flatten_param", &kFlatten),
    M(136, "reduction_param", &kReduction),
    M(137, "embed_param", &kEmbed),
    M(138, "tile_param", &kTile),
    M(139, "batch_norm_param", &kBatchNorm),
    M(140, "elu_param", &kELU),
    M(141, "bias_param", &kBias),
    M(142, "scale_param", &kScale),
    M(143, "input_param", &kInput),
    M(144, "crop_param", &kCrop),
    M(145, "parameter_param", &kParameter),
    M(146, "recurrent_param", &kRecurrent),
    M(147, "swish_param", &kSwish),
    M(148, "clip_param", &kClip),
};
constexpr MessageSpec kLayer = {"LayerParameter", kLayerF, std::size(kLayerF)};

// ---- V1LayerParameter (legacy, NetParameter.layers = 2) ---------------------------
constexpr FieldSpec kV1LayerF[] = {
    M(1, "layer", nullptr),              // V0LayerParameter, structural
    S(2, "bottom", K::String, true),
    S(3, "top", K::String, true),
    S(4, "name", K::String),
    E(5, "type", &kV1Type),
    M(6, "blobs", nullptr, true),
    S(7, "blobs_lr", K::Float, true),
    S(8, "weight_decay", K::Float, true),
    M(9, "concat_param", &kConcat),
    M(10, "convolution_param", &kConvolution),
    M(11, "data_param", nullptr),
    M(12, "dropout_param", &kDropout),
    M(13, "hdf5_data_param", nullptr),
    M(14, "hdf5_output_param", nullptr),
    M(15, "image_data_param", nullptr),
    M(16, "infogain_loss_param", nullptr),
    M(17, "inner_product_param", &kInnerProduct),
    M(18, "lrn_param", &kLRN),
    M(19, "pooling_param", &kPooling),
    M(20, "window_data_param", nullptr),
    M(21, "power_param", &kPower),
    M(22, "memory_data_param", nullptr),
    M(23, "argmax_param", &kArgMax),
    M(24, "eltwise_param", &kEltwise),
    M(25, "threshold_param", &kThreshold),
    M(26, "dummy_data_param", &kDummyData),
    M(27, "accuracy_param", nullptr),
    M(29, "hinge_loss_param", nullptr),
    M(30, "relu_param", &kReLU),
    M(31, "slice_param", &kSlice),
    M(32, "include", &kNetStateRule, true),
    M(33, "exclude", &kNetStateRule, true),
    M(34, "mvn_param", &kMVN),
    S(35, "loss_weight", K::Float, true),
    M(36, "transform_param", nullptr),
    M(37, "tanh_param", &kTanH),
    M(38, "sigmoid_param", &kSigmoid),
    M(39, "softmax_param", &kSoftmax),
    M(40, "contrastive_loss_param", nullptr),
    M(41, "exp_param", &kExp),
    M(42, "loss_param", nullptr),
    S(1001, "param", K::String, true),
    E(1002, "blob_share_mode", &kDimCheckMode, true),
};
constexpr MessageSpec kV1Layer = {"V1LayerParameter", kV1LayerF, std::size(kV1LayerF)};

// ---- NetParameter (top level; the parser handles each field structurally) ----------
constexpr FieldSpec kNetF[] = {
    S(1, "name", K::String),
    M(2, "layers", &kV1Layer, true),
    S(3, "input", K::String, true),
    S(4, "input_dim", K::Int32, true),
    S(5, "force_backward", K::Bool),
    M(6, "state", &kNetState),
    S(7, "debug_info", K::Bool),
    M(8, "input_shape", &kBlobShape, true),
    M(100, "layer", &kLayer, true),
};
constexpr MessageSpec kNet = {"NetParameter", kNetF, std::size(kNetF)};

}  // namespace

const FieldSpec* find_field(const MessageSpec& m, uint32_t number) {
  for (size_t i = 0; i < m.count; ++i)
    if (m.fields[i].number == number) return &m.fields[i];
  return nullptr;
}

const FieldSpec* find_field(const MessageSpec& m, std::string_view name) {
  for (size_t i = 0; i < m.count; ++i)
    if (name == m.fields[i].name) return &m.fields[i];
  return nullptr;
}

const char* enum_name(const EnumSpec& e, int64_t number) {
  for (size_t i = 0; i < e.count; ++i)
    if (e.values[i].number == number) return e.values[i].name;
  return nullptr;
}

std::optional<int32_t> enum_number(const EnumSpec& e, std::string_view name) {
  for (size_t i = 0; i < e.count; ++i)
    if (name == e.values[i].name) return e.values[i].number;
  return std::nullopt;
}

const MessageSpec& net_parameter_spec() { return kNet; }
const MessageSpec& layer_parameter_spec() { return kLayer; }
const MessageSpec& v1_layer_parameter_spec() { return kV1Layer; }
const MessageSpec& blob_shape_spec() { return kBlobShape; }
const MessageSpec& net_state_spec() { return kNetState; }
const MessageSpec& net_state_rule_spec() { return kNetStateRule; }
const EnumSpec& phase_enum() { return kPhase; }
const EnumSpec& v1_layer_type_enum() { return kV1Type; }

const char* v1_modern_type(int64_t number) {
  if (number < 0 || number > kMaxV1LayerType) return "";
  return kV1Modern[static_cast<size_t>(number)];
}

}  // namespace netvis::caffe
