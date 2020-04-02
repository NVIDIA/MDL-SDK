/******************************************************************************
 * Copyright (c) 2012-2020, NVIDIA CORPORATION. All rights reserved.
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
 *****************************************************************************/

#include "pch.h"

#include <cmath>

#include <mi/mdl/mdl_definitions.h>
#include <mi/mdl/mdl_values.h>

#include <mi/math/function.h>
#include <mi/math/vector.h>

#include "mdl/runtime/spectral/i_spectral.h"

#include "compilercore_assert.h"
#include "compilercore_tools.h"

namespace mi {
namespace mdl {

using namespace mi::math;

MDL_CONSTEXPR inline float  max_value(float x)  { return x; }
MDL_CONSTEXPR inline double max_value(double x) { return x; }
MDL_CONSTEXPR inline float  min_value(float x)  { return x; }
MDL_CONSTEXPR inline double min_value(double x) { return x; }

inline float  distance(float a,  float b)  { return abs(a-b); }
inline double distance(double a, double b) { return abs(a-b); }

MDL_CONSTEXPR inline float  normalize(float a)  { return 1.0f; }
MDL_CONSTEXPR inline double normalize(double a) { return 1.0; }

/// Helper for luminance(float3)
static IValue const *do_luminance_sRGB(
    IValue_factory       *value_factory,
    IValue const * const arguments[])
{
    IValue_vector const *a = cast<IValue_vector>(arguments[0]);

    // From MDL spec 20.2 Standard library functions:
    // "The color space of a is implementation dependent ... and assumes the sRGB color
    // space if a is of type float3, i.e., the luminance is then equal to 
    // 0.212671 * a.x + 0.715160 * a.y + 0.072169 * a.z."

    IValue_float const *x = cast<IValue_float>(a->get_value(0));
    IValue_float const *y = cast<IValue_float>(a->get_value(1));
    IValue_float const *z = cast<IValue_float>(a->get_value(2));

    float lum = float(
        0.212671 * x->get_value() + 0.715160 * y->get_value() + 0.072169 * z->get_value());

    return value_factory->create_float(lum);
}

/// Helper for luminance(color)
static IValue const *do_luminance_color(
    IValue_factory       *value_factory,
    IValue const * const arguments[])
{
    // FIXME: The Spec does not specify the color space, assume sRGB for now.
    IValue_rgb_color const *a = cast<IValue_rgb_color>(arguments[0]);

    IValue_float const *x = cast<IValue_float>(a->get_value(0));
    IValue_float const *y = cast<IValue_float>(a->get_value(1));
    IValue_float const *z = cast<IValue_float>(a->get_value(2));

    float lum = float(
        0.212671 * x->get_value() + 0.715160 * y->get_value() + 0.072169 * z->get_value());

    return value_factory->create_float(lum);
}

template <typename T>
struct Value_type_trait {};

template <>
struct Value_type_trait<float> {
    typedef IValue_float Value_type;
    typedef double       IM_type;
};

template <>
struct Value_type_trait<double> {
    typedef IValue_double Value_type;
    typedef double       IM_type;
};

static inline IValue const *create_value(IValue_factory *factory, float value)
{
    return factory->create_float(value);
}

static inline IValue const *create_value(IValue_factory *factory, double value)
{
    return factory->create_double(value);
}

/// Helper for cross.
template <
    typename T
>
static IValue const *do_cross(
    IValue_factory       *value_factory,
    IValue const * const arguments[])
{
    typedef typename Value_type_trait<T>::Value_type VT;

    IValue_vector const *lhs = cast<IValue_vector>(arguments[0]);
    IValue_vector const *rhs = cast<IValue_vector>(arguments[1]);

    typedef math::Vector<T,3> Vector3;

    Vector3 v_lhs(
        cast<VT>(lhs->get_value(0))->get_value(),
        cast<VT>(lhs->get_value(1))->get_value(),
        cast<VT>(lhs->get_value(2))->get_value());
    Vector3 v_rhs(
        cast<VT>(rhs->get_value(0))->get_value(),
        cast<VT>(rhs->get_value(1))->get_value(),
        cast<VT>(rhs->get_value(2))->get_value());

    Vector3 v_res = cross(v_lhs, v_rhs);

    IValue const *res[3] = {
        create_value(value_factory, v_res.x),
        create_value(value_factory, v_res.y),
        create_value(value_factory, v_res.z)
    };

    IType_vector const *v_type = lhs->get_type();

    return value_factory->create_vector(v_type, res, 3);
}

/// Helper for max_value(floatX)
template <
    typename T
>
static IValue const *do_max_value(
    IValue_factory       *value_factory,
    IValue const * const arguments[])
{
    typedef typename Value_type_trait<T>::Value_type VT;

    IValue_vector const *a = cast<IValue_vector>(arguments[0]);
    VT const            *v = cast<VT>(a->get_value(0));

    T res = v->get_value();

    for (int i = 1, n = a->get_component_count(); i < n; ++i) {
        v = cast<VT>(a->get_value(i));
        T t = v->get_value();

        if (t > res)
            res = t;
    }
    return create_value(value_factory, res);
}

/// Helper for max_value(color)
static IValue const *do_max_value_rgb_color(
    IValue_factory       *value_factory,
    IValue const * const arguments[])
{
    IValue_rgb_color const *a = cast<IValue_rgb_color>(arguments[0]);
    IValue_float     const *v = cast<IValue_float>(a->get_value(0));

    float t   = v->get_value();
    float res = t;

    v = cast<IValue_float>(a->get_value(1));
    t   = v->get_value();
    if (t > res)
        res = t;

    v = cast<IValue_float>(a->get_value(2));
    t   = v->get_value();
    if (t > res)
        res = t;

    return create_value(value_factory, res);
}

/// Helper for min_value(floatX)
template <
    typename T
>
static IValue const *do_min_value(
    IValue_factory       *value_factory,
    IValue const * const arguments[])
{
    typedef typename Value_type_trait<T>::Value_type VT;

    IValue_vector const *a = cast<IValue_vector>(arguments[0]);
    VT const            *v = cast<VT>(a->get_value(0));

    T res = v->get_value();

    for (int i = 1, n = a->get_component_count(); i < n; ++i) {
        v = cast<VT>(a->get_value(i));
        T t = v->get_value();

        if (t < res)
            res = t;
    }
    return create_value(value_factory, res);
}

/// Helper for min_value(color)
static IValue const *do_min_value_rgb_color(
    IValue_factory       *value_factory,
    IValue const * const arguments[])
{
    IValue_rgb_color const *a = cast<IValue_rgb_color>(arguments[0]);
    IValue_float     const *v = cast<IValue_float>(a->get_value(0));

    float t   = v->get_value();
    float res = t;

    v = cast<IValue_float>(a->get_value(1));
    t   = v->get_value();
    if (t < res)
        res = t;

    v = cast<IValue_float>(a->get_value(2));
    t   = v->get_value();
    if (t < res)
        res = t;

    return create_value(value_factory, res);
}

/// Helper for distance(floatX)
template <
    typename T
>
static IValue const *do_distance(
    IValue_factory       *value_factory,
    IValue const * const arguments[])
{
    typedef typename Value_type_trait<T>::Value_type VT;
    typedef typename Value_type_trait<T>::IM_type IM;

    IValue_vector const *a = cast<IValue_vector>(arguments[0]);
    IValue_vector const *b = cast<IValue_vector>(arguments[1]);

    IM r = IM(cast<VT>(a->get_value(0))->get_value()) - IM(cast<VT>(b->get_value(0))->get_value());
    r = r * r;
    for (int i = 1, n = a->get_component_count(); i < n; ++i) {
        IM t =
            IM(cast<VT>(a->get_value(i))->get_value()) -
            IM(cast<VT>(b->get_value(i))->get_value());
        r += t * t;
    }
    r = sqrt(r);

    return create_value(value_factory, T(r));
}

/// Helper for length(floatX)
template <
    typename T
>
static IValue const *do_length(
    IValue_factory       *value_factory,
    IValue const * const arguments[])
{
    typedef typename Value_type_trait<T>::Value_type VT;
    typedef typename Value_type_trait<T>::IM_type IM;

    IValue_vector const *a = cast<IValue_vector>(arguments[0]);

    IM r = IM(cast<VT>(a->get_value(0))->get_value());
    r = r * r;
    for (int i = 1, n = a->get_component_count(); i < n; ++i) {
        IM t = IM(cast<VT>(a->get_value(i))->get_value());
        r += t * t;
    }
    r = sqrt(r);

    return create_value(value_factory, T(r));
}

/// Helper for normalize(floatX)
template <
    typename T
>
static IValue const *do_normalize(
    IValue_factory       *value_factory,
    IValue const * const arguments[])
{
    typedef typename Value_type_trait<T>::Value_type VT;
    typedef typename Value_type_trait<T>::IM_type IM;

    IValue_vector const *a = cast<IValue_vector>(arguments[0]);
    int                  n = a->get_component_count();

    MDL_ASSERT(n <= 4);

    IM r = IM(cast<VT>(a->get_value(0))->get_value());
    r = r * r;
    for (int i = 1; i < n; ++i) {
        IM t = IM(cast<VT>(a->get_value(i))->get_value());
        r += t * t;
    }
    r = sqrt(r);

    // MDL spec 20.2 says: "If the length of a is zero the result of normalize(a) is undefined"

    IValue const *res[4];
    for (int i = 0; i < n; ++i) {
        res[i] = create_value(value_factory, T(IM(cast<VT>(a->get_value(i))->get_value()) / r));
    }

    return value_factory->create_vector(a->get_type(), res, n);
}

/// Helper for dot(floatX)
template <
    typename T
>
static IValue const *do_dot(
    IValue_factory       *value_factory,
    IValue const * const arguments[])
{
    typedef typename Value_type_trait<T>::Value_type VT;
    typedef typename Value_type_trait<T>::IM_type IM;

    IValue_vector const *a = cast<IValue_vector>(arguments[0]);
    IValue_vector const *b = cast<IValue_vector>(arguments[1]);

    IM r = IM(cast<VT>(a->get_value(0))->get_value()) * IM(cast<VT>(b->get_value(0))->get_value());
    for (int i = 1, n = a->get_component_count(); i < n; ++i) {
        r += 
            IM(cast<VT>(a->get_value(i))->get_value()) *
            IM(cast<VT>(b->get_value(i))->get_value());
    }
    return create_value(value_factory, T(r));
}

/// Helper for transpose(matrix)
template <
    typename T
>
static IValue const *do_transpose(
    IValue_factory       *value_factory,
    IValue const * const arguments[])
{
    typedef typename Value_type_trait<T>::Value_type VT;

    IValue_matrix const *a = cast<IValue_matrix>(arguments[0]);

    T data[4][4];

    IType_matrix const *m_type = a->get_type();
    IType_vector const *v_type = m_type->get_element_type();
    
    int n = a->get_component_count();
    int m = v_type->get_size();
    for (int i = 0; i < n; ++i) {
        IValue_vector const *row = cast<IValue_vector>(a->get_value(i));
        for (int j = 0; j < m; ++j) {
            VT const *v = cast<VT>(row->get_value(j));

            data[i][j] = v->get_value();
        }
    }

    IType_factory      *type_fact = value_factory->get_type_factory();
    IType_atomic const *e_type    = v_type->get_element_type();
    IType_vector const *nv_type   = type_fact->create_vector(e_type, n);

    IValue const *row[4];
    IValue const *matrix[4];

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            row[j] = create_value(value_factory, data[j][i]);
        }
        matrix[i] = value_factory->create_vector(nv_type, row, n);
    }

    IType_matrix const *nm_type = type_fact->create_matrix(nv_type, m);
    return value_factory->create_matrix(nm_type, matrix, m);
}

#include "compilercore_intrinsic_eval.i"

}  // mdl
}  // mi
