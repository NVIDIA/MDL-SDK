/***************************************************************************************************
* Copyright 2022 NVIDIA Corporation. All rights reserved.
**************************************************************************************************/

#include "pch.h"

#include "dds_types.h"

namespace MI {

namespace DDS {

std::string get_dxgi_format_string( Dxgi_format value)
{
#define CASE(format) case DXGI_FORMAT_##format: return #format;

    switch( value) {
        CASE(UNKNOWN);
        CASE(R32G32B32A32_TYPELESS);
        CASE(R32G32B32A32_FLOAT);
        CASE(R32G32B32A32_UINT);
        CASE(R32G32B32A32_SINT);
        CASE(R32G32B32_TYPELESS);
        CASE(R32G32B32_FLOAT);
        CASE(R32G32B32_UINT);
        CASE(R32G32B32_SINT);
        CASE(R16G16B16A16_TYPELESS);
        CASE(R16G16B16A16_FLOAT);
        CASE(R16G16B16A16_UNORM);
        CASE(R16G16B16A16_UINT);
        CASE(R16G16B16A16_SNORM);
        CASE(R16G16B16A16_SINT);
        CASE(R32G32_TYPELESS);
        CASE(R32G32_FLOAT);
        CASE(R32G32_UINT);
        CASE(R32G32_SINT);
        CASE(R32G8X24_TYPELESS);
        CASE(D32_FLOAT_S8X24_UINT);
        CASE(R32_FLOAT_X8X24_TYPELESS);
        CASE(X32_TYPELESS_G8X24_UINT);
        CASE(R10G10B10A2_TYPELESS);
        CASE(R10G10B10A2_UNORM);
        CASE(R10G10B10A2_UINT);
        CASE(R11G11B10_FLOAT);
        CASE(R8G8B8A8_TYPELESS);
        CASE(R8G8B8A8_UNORM);
        CASE(R8G8B8A8_UNORM_SRGB);
        CASE(R8G8B8A8_UINT);
        CASE(R8G8B8A8_SNORM);
        CASE(R8G8B8A8_SINT);
        CASE(R16G16_TYPELESS);
        CASE(R16G16_FLOAT);
        CASE(R16G16_UNORM);
        CASE(R16G16_UINT);
        CASE(R16G16_SNORM);
        CASE(R16G16_SINT);
        CASE(R32_TYPELESS);
        CASE(D32_FLOAT);
        CASE(R32_FLOAT);
        CASE(R32_UINT);
        CASE(R32_SINT);
        CASE(R24G8_TYPELESS);
        CASE(D24_UNORM_S8_UINT);
        CASE(R24_UNORM_X8_TYPELESS);
        CASE(X24_TYPELESS_G8_UINT);
        CASE(R8G8_TYPELESS);
        CASE(R8G8_UNORM);
        CASE(R8G8_UINT);
        CASE(R8G8_SNORM);
        CASE(R8G8_SINT);
        CASE(R16_TYPELESS);
        CASE(R16_FLOAT);
        CASE(D16_UNORM);
        CASE(R16_UNORM);
        CASE(R16_UINT);
        CASE(R16_SNORM);
        CASE(R16_SINT);
        CASE(R8_TYPELESS);
        CASE(R8_UNORM);
        CASE(R8_UINT);
        CASE(R8_SNORM);
        CASE(R8_SINT);
        CASE(A8_UNORM);
        CASE(R1_UNORM);
        CASE(R9G9B9E5_SHAREDEXP);
        CASE(R8G8_B8G8_UNORM);
        CASE(G8R8_G8B8_UNORM);
        CASE(BC1_TYPELESS);
        CASE(BC1_UNORM);
        CASE(BC1_UNORM_SRGB);
        CASE(BC2_TYPELESS);
        CASE(BC2_UNORM);
        CASE(BC2_UNORM_SRGB);
        CASE(BC3_TYPELESS);
        CASE(BC3_UNORM);
        CASE(BC3_UNORM_SRGB);
        CASE(BC4_TYPELESS);
        CASE(BC4_UNORM);
        CASE(BC4_SNORM);
        CASE(BC5_TYPELESS);
        CASE(BC5_UNORM);
        CASE(BC5_SNORM);
        CASE(B5G6R5_UNORM);
        CASE(B5G5R5A1_UNORM);
        CASE(B8G8R8A8_UNORM);
        CASE(B8G8R8X8_UNORM);
        CASE(R10G10B10_XR_BIAS_A2_UNORM);
        CASE(B8G8R8A8_TYPELESS);
        CASE(B8G8R8A8_UNORM_SRGB);
        CASE(B8G8R8X8_TYPELESS);
        CASE(B8G8R8X8_UNORM_SRGB);
        CASE(BC6H_TYPELESS);
        CASE(BC6H_UF16);
        CASE(BC6H_SF16);
        CASE(BC7_TYPELESS);
        CASE(BC7_UNORM);
        CASE(BC7_UNORM_SRGB);
        CASE(AYUV);
        CASE(Y410);
        CASE(Y416);
        CASE(NV12);
        CASE(P010);
        CASE(P016);
        CASE(420_OPAQUE);
        CASE(YUY2);
        CASE(Y210);
        CASE(Y216);
        CASE(NV11);
        CASE(AI44);
        CASE(IA44);
        CASE(P8);
        CASE(A8P8);
        CASE(B4G4R4A4_UNORM);
        CASE(P208);
        CASE(V208);
        CASE(V408);
        CASE(SAMPLER_FEEDBACK_MIN_MIP_OPAQUE);
        CASE(SAMPLER_FEEDBACK_MIP_REGION_USED_OPAQUE);

        case DXGI_FORMAT_FORCE_UINT:
        default:
            return std::to_string( value);
    }
}

#undef CASE

}  // namespace DDS

}  // namespace MI
