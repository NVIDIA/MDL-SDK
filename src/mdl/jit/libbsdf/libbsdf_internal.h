/***************************************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "../../../mdl/compiler/stdmodule/enums.h"

using namespace mi::mdl::df;

#define BSDF_API extern "C"
#define BSDF_PARAM extern "C"

#ifdef _MSC_VER
    #define __align__(n) __declspec(align(n))
    #define BSDF_INLINE __forceinline
    #define BSDF_NOINLINE __declspec(noinline)
#else
    #define __align__(n) __attribute__((aligned(n)))
    #define BSDF_INLINE __attribute__((always_inline)) inline
    #define BSDF_NOINLINE __attribute__((noinline))
#endif

#ifndef M_PI
    #define M_PI            3.14159265358979323846
#endif
#define M_ONE_OVER_PI       0.318309886183790671538

//-----------------------------------------------------------------------------
// define vector types CUDA-like
//-----------------------------------------------------------------------------



struct __align__(8) bool2
{
    bool x, y;
};

struct bool3
{
    bool x, y, z;
};

struct __align__(16) bool4
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


struct __align__(8) uint2
{
    unsigned x, y;
};

struct uint3
{
    unsigned x, y, z;
};

struct __align__(16) uint4
{
    unsigned x, y, z, w;
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

struct float3x3
{
    float3 col0, col1, col2;
};

//-----------------------------------------------------------------------------
// CUDA-like make functions
//-----------------------------------------------------------------------------

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

BSDF_INLINE uint2 make_uint2(unsigned x, unsigned y)
{
    uint2 t; t.x = x; t.y = y; return t;
}
BSDF_INLINE uint3 make_uint3(unsigned x, unsigned y, unsigned z)
{
    uint3 t; t.x = x; t.y = y; t.z = z; return t;
}
BSDF_INLINE uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w)
{
    uint4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

BSDF_INLINE int2 make_int2(int x, int y)
{
    int2 t; t.x = x; t.y = y; return t;
}
BSDF_INLINE int3 make_int3(int x, int y, int z)
{
    int3 t; t.x = x; t.y = y; t.z = z; return t;
}
BSDF_INLINE int4 make_int4(int x, int y, int z, int w)
{
    int4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

BSDF_INLINE bool2 make_bool2(bool x, bool y)
{
    bool2 t; t.x = x; t.y = y; return t;
}
BSDF_INLINE bool3 make_bool3(bool x, bool y, bool z)
{
    bool3 t; t.x = x; t.y = y; t.z = z; return t;
}
BSDF_INLINE bool4 make_bool4(bool x, bool y, bool z, bool w)
{
    bool4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

//-----------------------------------------------------------------------------
// Generic vector creation and conversion.
// Generate type traits for build-in types.
//-----------------------------------------------------------------------------

namespace
{
    // Generate type traits for build-in types
    template<typename T> struct vector_trait {};
    
    #define VECTOR_TRAIT(T, S)                          \
    template<>                                          \
    struct vector_trait<typename vector<T,S>::TYPE >    \
    {                                                   \
        typedef T ELEMENT_TYPE;                         \
        enum { SIZE = S };                              \
    }

    template<typename T, int S> struct vector {};
    template<> struct vector<float, 1>      { typedef float TYPE; };    VECTOR_TRAIT(float, 1);
    template<> struct vector<float, 2>      { typedef float2 TYPE; };   VECTOR_TRAIT(float, 2);
    template<> struct vector<float, 3>      { typedef float3 TYPE; };   VECTOR_TRAIT(float, 3);
    template<> struct vector<float, 4>      { typedef float4 TYPE; };   VECTOR_TRAIT(float, 4);
    template<> struct vector<unsigned, 1>   { typedef unsigned TYPE; }; VECTOR_TRAIT(unsigned, 1);
    template<> struct vector<unsigned, 2>   { typedef uint2 TYPE; };    VECTOR_TRAIT(unsigned, 2);
    template<> struct vector<unsigned, 3>   { typedef uint3 TYPE; };    VECTOR_TRAIT(unsigned, 3);
    template<> struct vector<unsigned, 4>   { typedef uint4 TYPE; };    VECTOR_TRAIT(unsigned, 4);
    template<> struct vector<int, 1>        { typedef int TYPE; };      VECTOR_TRAIT(int, 1);
    template<> struct vector<int, 2>        { typedef int2 TYPE; };     VECTOR_TRAIT(int, 2);
    template<> struct vector<int, 3>        { typedef int3 TYPE; };     VECTOR_TRAIT(int, 3);
    template<> struct vector<int, 4>        { typedef int4 TYPE; };     VECTOR_TRAIT(int, 4);
    template<> struct vector<bool, 1>       { typedef bool TYPE; };     VECTOR_TRAIT(bool, 1);
    template<> struct vector<bool, 2>       { typedef bool2 TYPE; };    VECTOR_TRAIT(bool, 2);
    template<> struct vector<bool, 3>       { typedef bool3 TYPE; };    VECTOR_TRAIT(bool, 3);
    template<> struct vector<bool, 4>       { typedef bool4 TYPE; };    VECTOR_TRAIT(bool, 4);

    // convert vector to type to vector of same size but different element type
    template<typename TInput, typename TTargetElement>
    struct convert_base
    {
        typedef typename vector<TTargetElement, vector_trait<TInput>::SIZE>::TYPE TARGET_TYPE;
    };
}

//-----------------------------------------------------------------------------
// Convert vector to type to vector of same size but different element type
//-----------------------------------------------------------------------------

template<typename TTargetElement, typename TInput>
BSDF_INLINE typename convert_base<TInput, TTargetElement>::TARGET_TYPE to(const TInput& a)
{
    const unsigned vector_size = vector_trait<TInput>::SIZE;            // number of components
    typedef typename convert_base<TInput, TTargetElement>::TARGET_TYPE TTarget;  // target vec type
    typedef typename vector_trait<TInput>::ELEMENT_TYPE TInputElement;        // input element type

    TTarget res; // convert vector element-wise
    const TInputElement* src_ptr = reinterpret_cast<const TInputElement*>(&a);
    TTargetElement* dst_ptr = reinterpret_cast<TTargetElement*>(&res);
    for (unsigned i = 0; i < vector_size; ++i)
        dst_ptr[i] = TTargetElement(src_ptr[i]);
    return res;
}


//-----------------------------------------------------------------------------
// Generic creation of a vector while specifying values for each component.
//-----------------------------------------------------------------------------

namespace
{
    template<typename TTarget, typename TComponent>
    BSDF_INLINE void set_component(
        TTarget& target,
        unsigned target_index,
        const TComponent& arg)
    {
        typedef typename vector_trait<TTarget>::ELEMENT_TYPE TTargetElement; // output element type
        TTargetElement* dst_ptr = reinterpret_cast<TTargetElement*>(&target);
        dst_ptr[target_index] = TTargetElement(arg);
    }

    template<typename TTarget, typename TComponentFirst, typename... TComponentRest>
    BSDF_INLINE void set_component(
        TTarget& target,
        unsigned target_index,
        const TComponentFirst& first,
        const TComponentRest& ... args)
    {
        typedef typename vector_trait<TTarget>::ELEMENT_TYPE TTargetElement; // output element type
        TTargetElement* dst_ptr = reinterpret_cast<TTargetElement*>(&target);
        dst_ptr[target_index] = TTargetElement(first);

        set_component(target, target_index + 1u, args...);
    }
}

template<typename TTarget, typename... TComponent>
BSDF_INLINE TTarget make(const TComponent& ... args)
{
    typedef typename vector_trait<TTarget>::ELEMENT_TYPE TTargetElement; // output element type
    const unsigned vector_size = vector_trait<TTarget>::SIZE;            // number of components
    const unsigned arg_size = sizeof...(TComponent);                     // number of arguments
    static_assert(arg_size == vector_size || arg_size == 1, 
                  "Vector size and argument count do not match. "
                  "They have to be equal, or argument count has to be one.");

    TTarget res;
    if (arg_size == 1)
        for (unsigned i = 0; i < vector_size; ++i)
            set_component(res, i, args...);
    else
        set_component(res, 0u, args...);

    return res;
}


//-----------------------------------------------------------------------------
// Swizzle replacement functions
//-----------------------------------------------------------------------------
BSDF_INLINE float3 xyz(float4 v) { return  make<float3>(v.x, v.y, v.z); }


//-----------------------------------------------------------------------------
// Function to mimic a ternary operator (?:) 
//-----------------------------------------------------------------------------

namespace
{
    template<typename TA, typename TB>
    struct is_same_type { enum { VALUE = 0 }; };

    template<typename T>
    struct is_same_type<T, T> { enum { VALUE = 1 }; };
}


template<typename TVector, typename TBool>
BSDF_INLINE TVector ternary(const TBool& condition,
                            const TVector& expr_true,
                            const TVector& expr_false)
{
    const unsigned vector_size = vector_trait<TVector>::SIZE;            // number of components
    typedef typename vector_trait<TVector>::ELEMENT_TYPE TElement;       // element type
    static_assert(is_same_type<typename vector_trait<TBool>::ELEMENT_TYPE, bool>::VALUE,
                  "ternary<T(Condition,T,T)>: Condition has to be a boolean type.");

    TVector res;
    TElement* res_ptr = reinterpret_cast<TElement*>(&res);
    const bool* condition_ptr = reinterpret_cast<const bool*>(&condition);
    const TElement* expr_true_ptr = reinterpret_cast<const TElement*>(&expr_true);
    const TElement* expr_false_ptr = reinterpret_cast<const TElement*>(&expr_false);

    for (unsigned i = 0; i < vector_size; ++i)
        res_ptr[i] = condition_ptr[i] ? expr_true_ptr[i] : expr_false_ptr[i];

    return res;
}



//-----------------------------------------------------------------------------
// Operators
//-----------------------------------------------------------------------------

BSDF_INLINE bool3 operator!(const bool3& a)
{
    return make_bool3(!a.x, !a.y, !a.z);
}
BSDF_INLINE bool3 operator&&(const bool3& a, const bool3& b)
{
    return make_bool3(a.x && b.x, a.y && b.y, a.z && b.z);
}
BSDF_INLINE bool3 operator||(const bool3& a, const bool3& b)
{
    return make_bool3(a.x || b.x, a.y || b.y, a.z || b.z);
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
BSDF_INLINE void operator+=(uint3& a, const uint3& b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}
BSDF_INLINE void operator+=(int3& a, const int3& b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
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

BSDF_INLINE bool operator==(const bool3& a, const bool3& b)
{
    return (a.x == b.x && a.y == b.y && a.z == b.z);
}

BSDF_INLINE bool operator!= (const bool3& a, const bool3& b)
{
    return !(a == b);
}

BSDF_INLINE bool equal(unsigned a, unsigned b)
{
    return a == b;
}
BSDF_INLINE bool equal(int a, int b)
{
    return a == b;
}
BSDF_INLINE bool equal(bool a, bool b)
{
    return a == b;
}
BSDF_INLINE bool3 equal(const uint3& a, const uint3& b)
{
    bool3 res;
    res.x = (a.x == b.x);
    res.y = (a.y == b.y);
    res.z = (a.z == b.z);
    return res;
}
BSDF_INLINE bool3 equal(const int3& a, const int3& b)
{
    bool3 res;
    res.x = (a.x == b.x);
    res.y = (a.y == b.y);
    res.z = (a.z == b.z);
    return res;
}
BSDF_INLINE bool3 equal(const bool3& a, const bool3& b)
{
    bool3 res;
    res.x = (a.x == b.x);
    res.y = (a.y == b.y);
    res.z = (a.z == b.z);
    return res;
}

BSDF_INLINE bool less_than(const float a, const float b)
{
    return a < b;
}
BSDF_INLINE bool3 less_than(const float3& a, const float3& b)
{
    bool3 res;
    res.x = a.x < b.x;
    res.y = a.y < b.y;
    res.z = a.z < b.z;
    return res;
}

BSDF_INLINE bool less_than_equal(const float a, const float b)
{
    return !(a > b);
}
BSDF_INLINE bool3 less_than_equal(const float3& a, const float3& b)
{
    bool3 res;
    res.x = !(a.x > b.x);
    res.y = !(a.y > b.y);
    res.z = !(a.z > b.z);
    return res;
}
BSDF_INLINE bool greater_than(const float a, const float b)
{
    return a > b;
}
BSDF_INLINE bool3 greater_than(const float3& a, const float3& b)
{
    bool3 res;
    res.x = a.x > b.x;
    res.y = a.y > b.y;
    res.z = a.z > b.z;
    return res;
}
BSDF_INLINE bool3 greater_than(const uint3& a, const uint3& b)
{
    bool3 res;
    res.x = a.x > b.x;
    res.y = a.y > b.y;
    res.z = a.z > b.z;
    return res;
}
BSDF_INLINE bool greater_than_equal(const float a, const float b)
{
    return !(a < b);
}

BSDF_INLINE bool3 greater_than_equal(const float3& a, const float3& b)
{
    bool3 res;
    res.x = !(a.x < b.x);
    res.y = !(a.y < b.y);
    res.z = !(a.z < b.z);
    return res;
}


enum Mbsdf_part
{
    mbsdf_data_reflection = 0,
    mbsdf_data_transmission = 1
};

namespace state
{
    enum coordinate_space
    {
        coordinate_internal,
        coordinate_object,
        coordinate_world
    };
}

/// The kind of BSDF data in case of BSDF data textures (otherwise BDK_NONE).
/// Must be in-sync with mi::mdl::IValue_texture::Bsdf_data_kind.
enum Bsdf_data_kind {
    BDK_NONE,
    BDK_SIMPLE_GLOSSY_MULTISCATTER,
    BDK_BACKSCATTERING_GLOSSY_MULTISCATTER,
    BDK_BECKMANN_SMITH_MULTISCATTER,
    BDK_GGX_SMITH_MULTISCATTER,
    BDK_BECKMANN_VC_MULTISCATTER,
    BDK_GGX_VC_MULTISCATTER,
    BDK_WARD_GEISLER_MORODER_MULTISCATTER,
    BDK_SHEEN_MULTISCATTER,
};

class State
{
public:
    float3 normal() const;
    float3 geometry_normal() const;
    float3 texture_tangent_u(int index) const;
    float3 texture_tangent_v(int index) const;

    float3 transform_vector(
        state::coordinate_space from,
        state::coordinate_space to,
        const float3& vector) const;

    char const *get_texture_results() const;
    char const *get_arg_block() const;
    float call_lambda_float(int index) const;
    float3 call_lambda_float3(int index) const;
    unsigned int call_lambda_uint(int index) const;
    float get_arg_block_float(int offset) const;
    float3 get_arg_block_float3(int offset) const;
    unsigned int get_arg_block_uint(int offset) const;
    bool get_arg_block_bool(int offset) const;
    float3 get_material_ior() const;
    float3 get_measured_curve_value(int measured_curve_idx, int value_idx);
    unsigned int get_thin_walled() const;

    uint3 bsdf_measurement_resolution(
        int bsdf_measurement_index,
        int part) const;

    float3 bsdf_measurement_evaluate(
        int bsdf_measurement_index,
        const float2& theta_phi_in,
        const float2& theta_phi_out, 
        int part) const;

    float3 bsdf_measurement_sample(
        int bsdf_measurement_index,
        const float2& theta_phi_out,
        const float3& xi,
        int part) const;

    float bsdf_measurement_pdf(
        int bsdf_measurement_index,
        const float2& theta_phi_in,
        const float2& theta_phi_out, 
        int part) const;

    float4 bsdf_measurement_albedos(
        int bsdf_measurement_index,
        const float2& theta_phi) const;

    float light_profile_evaluate(
        int light_profile_index,
        const float2& theta_phi) const;

    float3 light_profile_sample(
        int light_profile_index,
        const float3& xi) const;

    float light_profile_pdf(
        int light_profile_index,
        const float2& theta_phi) const;

    float3 tex_lookup_float3_2d(
        int texture_index,
        const float2& coord,
        int wrap_u,
        int wrap_v,
        const float2& crop_u,
        const float2& crop_v) const;

    float3 tex_lookup_float3_3d(
        int texture_index,
        const float3& coord,
        int wrap_u,
        int wrap_v,
        int wrap_w,
        const float2& crop_u,
        const float2& crop_v,
        const float2& crop_w) const;

    unsigned get_bsdf_data_texture_id(Bsdf_data_kind bsdf_data_kind) const;

    float2 adapt_microfacet_roughness(const float2& roughness_uv) const;
};

#include "libbsdf_runtime.h"
#include "libbsdf.h"

struct BSDF
{
    void(*sample)(
        BSDF_sample_data *data, 
        State *state, 
        float3 const &inherited_normal);

    void(*evaluate)(
        BSDF_evaluate_data *data,
        State *state,
        float3 const &inherited_normal,
        float3 const &inherited_weight);

    void(*pdf)(
        BSDF_pdf_data *data, 
        State *state, 
        float3 const &inherited_normal);

    void(*auxiliary)(
        BSDF_auxiliary_data *data, 
        State *state, 
        float3 const &inherited_normal,
        float3 const &inherited_weight);

    // returns true, if the attached BSDF is "bsdf()".
    // note: this is currently unsupported for BSDFs in BSDF_component
    bool(*is_black)();
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
    void(*sample)(
        EDF_sample_data *data, 
        State *state, 
        float3 const &inherited_normal);

    void(*evaluate)(
        EDF_evaluate_data *data,
        State *state,
        float3 const &inherited_normal,
        float3 const &inherited_weight);

    void(*pdf)(
        EDF_pdf_data *data, 
        State *state, 
        float3 const &inherited_normal);

    void(*auxiliary)(
        EDF_auxiliary_data *data, 
        State *state, 
        float3 const &inherited_normal,
        float3 const &inherited_weight);

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
