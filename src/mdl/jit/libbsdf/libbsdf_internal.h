/***************************************************************************************************
 * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************************************/

#ifndef MDL_LIBBSDF_INTERNAL_H
#define MDL_LIBBSDF_INTERNAL_H

#define BSDF_API extern "C"
#define BSDF_PARAM extern "C"
#define BSDF_INLINE __attribute__((always_inline))
#define __align__(n) __attribute__((aligned(n)))

#define M_PI            3.14159265358979323846   // pi
#define M_ONE_OVER_PI   0.318309886183790671538  // pi

enum scatter_mode {
    scatter_reflect,
    scatter_transmit,
    scatter_reflect_transmit
};

// define vector types CUDA-like
struct bool2
{
    bool x, y;
};

struct bool3
{
    bool x, y, z;
};

struct bool4
{
    bool x, y, z, w;
};

struct __align__(8) int2
{
    int x, y;
};

struct int3
{
    int x, y, z;
};

struct __align__(16) int4
{
    int x, y, z, w;
};

struct __align__(8) float2
{
    float x, y;
};

struct float3
{
    float x, y, z;
};

struct __align__(16) float4
{
    float x, y, z, w;
};

struct __align__(16) double2
{
    double x, y;
};

struct double3
{
    double x, y, z;
};

struct __align__(16) double4
{
    double x, y, z, w;
};

typedef char const *string;

struct color
{
    float r, g, b;
};

BSDF_INLINE float2 make_float2(float x, float y)
{
    float2 t; t.x = x; t.y = y; return t;
}

BSDF_INLINE float3 make_float3(float x, float y, float z)
{
    float3 t; t.x = x; t.y = y; t.z = z; return t;
}

BSDF_INLINE float4 make_float4(float x, float y, float z, float w)
{
    float4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

BSDF_INLINE float2 operator+(const float2& a, const float2& b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}
BSDF_INLINE float2 operator-(const float2& a, const float2& b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}
BSDF_INLINE float3 operator-(const float3& a)
{
    return make_float3(-a.x, -a.y, -a.z);
}
BSDF_INLINE float3 operator+(const float3& a, const float3& b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
BSDF_INLINE void operator+=(float3& a, const float3& b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}
BSDF_INLINE float3 operator-(const float3& a, const float3& b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
BSDF_INLINE float3 operator-(const float3& a, const float b)
{
    return make_float3(a.x - b, a.y - b, a.z - b);
}
BSDF_INLINE float3 operator-(const float a, const float3& b)
{
    return make_float3(a - b.x, a - b.y, a - b.z);
}
BSDF_INLINE void operator-=(float3& a, const float3& b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}
BSDF_INLINE float3 operator*(const float3& a, const float3& b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
BSDF_INLINE float3 operator*(const float3& a, const float s)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}
BSDF_INLINE float3 operator*(const float s, const float3& a)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}
BSDF_INLINE void operator*=(float3& a, const float3& s)
{
    a.x *= s.x; a.y *= s.y; a.z *= s.z;
}
BSDF_INLINE void operator*=(float3& a, const float s)
{
    a.x *= s; a.y *= s; a.z *= s;
}
BSDF_INLINE float3 operator/(const float3& a, const float3& b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
BSDF_INLINE float3 operator/(const float3& a, const float s)
{
    float inv = 1.0f / s;
    return a * inv;
}
BSDF_INLINE float3 operator/(const float s, const float3& a)
{
    return make_float3( s/a.x, s/a.y, s/a.z );
}
BSDF_INLINE void operator/=(float3& a, const float s)
{
    float inv = 1.0f / s;
    a *= inv;
}
BSDF_INLINE float copysignf(const float dst, const float src)
{
    union {
        float f;
        unsigned int i;
    } v1, v2, v3;
    v1.f = src;
    v2.f = dst;
    v3.i = (v2.i & 0x7fffffff) | (v1.i & 0x80000000);
    
    return v3.f;
}


class State
{
public:
    float3 normal() const;
    float3 geometry_normal() const;
    float3 texture_tangent_u(int index) const;
    float3 texture_tangent_v(int index) const;
};

#include "libbsdf_runtime.h"
#include "libbsdf.h"

struct BSDF
{
    void (*sample)(BSDF_sample_data *data, State *state, float3 const &inherited_normal);
    void (*evaluate)(BSDF_evaluate_data *data, State *state, float3 const &inherited_normal);
    void (*pdf)(BSDF_pdf_data *data, State *state, float3 const &inherited_normal);

    // returns true, if the attached BSDF is "bsdf()".
    // note: this is currently unsupported for BSDFs in BSDF_component
    bool (*is_black)();
};

struct BSDF_component
{
    float weight;
    BSDF component;
};

struct color_BSDF_component
{
    float3 weight;
    BSDF component;
};

struct EDF
{
    void(*sample)(EDF_sample_data *data, State *state, float3 const &inherited_normal);
    void(*evaluate)(EDF_evaluate_data *data, State *state, float3 const &inherited_normal);
    void(*pdf)(EDF_pdf_data *data, State *state, float3 const &inherited_normal);

    // returns true, if the attached BSDF is "edf()".
    // note: this is currently unsupported for EDFs in EDF_component
    bool(*is_black)();
};

struct EDF_component
{
    float weight;
    EDF component;
};

struct color_EDF_component
{
    float3 weight;
    EDF component;
};


#endif // MDL_LIBBSDF_INTERNAL_H
