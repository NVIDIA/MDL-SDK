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

#ifndef MDL_LIBBSDF_UTILITIES_H
#define MDL_LIBBSDF_UTILITIES_H


BSDF_INLINE void absorb(BSDF_sample_data *data)
{
    data->pdf        = 0.f;
    data->event_type = BSDF_EVENT_ABSORB;
}

BSDF_INLINE void absorb(BSDF_evaluate_data *data)
{
    data->pdf  = 0.0f;
}

BSDF_INLINE void absorb(BSDF_pdf_data *data)
{
    data->pdf  = 0.0f;
}

BSDF_INLINE void absorb(BSDF_auxiliary_data *data)
{
}

BSDF_INLINE BSDF_pdf_data to_pdf_data(const BSDF_sample_data* sample_data)
{
    BSDF_pdf_data res;
    res.ior1 = sample_data->ior1;
    res.ior2 = sample_data->ior2;
    res.k1 = sample_data->k1;
    res.k2 = sample_data->k2;
    return res;
}

BSDF_INLINE void no_emission(EDF_sample_data *data)
{
    data->k1 = make_float3(0.0f, 0.0f, 0.0f);
    data->pdf = 0.f;
    data->edf_over_pdf = make_float3(0.0f, 0.0f, 0.0f);
    data->event_type = EDF_EVENT_NONE;
}

BSDF_INLINE void no_emission(EDF_evaluate_data *data)
{
    // keep cos if set
    data->pdf = 0.f;
}

BSDF_INLINE void no_emission(EDF_pdf_data *data)
{
    data->pdf = 0.0f;
}

BSDF_INLINE void no_emission(EDF_auxiliary_data *data)
{
}



BSDF_INLINE EDF_evaluate_data to_eval_data(const EDF_sample_data* sample_data)
{
    EDF_evaluate_data res;
    res.k1 = sample_data->k1;
    return res;

}
BSDF_INLINE EDF_pdf_data to_pdf_data(const EDF_sample_data* sample_data)
{
    EDF_pdf_data res;
    res.k1 = sample_data->k1;
    return res;

}

template<typename TDF_data>
BSDF_INLINE void no_contribution(TDF_data* data, const float3& normal);

template<> BSDF_INLINE void no_contribution(BSDF_sample_data* data, const float3& normal) { 
    absorb(data); }
template<> BSDF_INLINE void no_contribution(BSDF_evaluate_data* data, const float3& normal) { 
    absorb(data); }
template<> BSDF_INLINE void no_contribution(BSDF_pdf_data* data, const float3& normal) { 
    absorb(data); }
template<> BSDF_INLINE void no_contribution(BSDF_auxiliary_data* data, const float3& normal) { 
    absorb(data); }
template<> BSDF_INLINE void no_contribution(EDF_sample_data* data, const float3& normal) { 
    no_emission(data); }
template<> BSDF_INLINE void no_contribution(EDF_evaluate_data* data, const float3& normal) { 
    no_emission(data); }
template<> BSDF_INLINE void no_contribution(EDF_pdf_data* data, const float3& normal) { 
    no_emission(data); }
template<> BSDF_INLINE void no_contribution(EDF_auxiliary_data* data, const float3& normal) {
    no_emission(data); }

// obtain IOR values on both sides of the surface
template<typename Data>
BSDF_INLINE float2 process_ior(Data *data, State *state)
{
    if (data->ior1.x == BSDF_USE_MATERIAL_IOR)
        data->ior1 = get_material_ior(state);
    if (data->ior2.x == BSDF_USE_MATERIAL_IOR)
        data->ior2 = get_material_ior(state);

    // using color IORs would require some sort of spectral rendering, as of now libbsdf
    // reduces that to scalar by averaging
    float2 ior = make_float2(
        (data->ior1.x + data->ior1.y + data->ior1.z) * (float)(1.0 / 3.0),
        (data->ior2.x + data->ior2.y + data->ior2.z) * (float)(1.0 / 3.0));

    // ensure a certain threshold between incoming and outgoing IOR to avoid
    // numerical issues with microfacet BSDFs and half vector computation
    const float IOR_THRESHOLD = 1e-4f;
    const float ior_diff = ior.y - ior.x;
    if (math::abs(ior_diff) < IOR_THRESHOLD) {
        ior.y = ior.x + copysignf(IOR_THRESHOLD, ior_diff);
    }

    return ior;
}

BSDF_INLINE void compute_eta(float &eta, const float3 &ior1, const float3 &ior2)
{
    eta = (ior2.x + ior2.y + ior2.z) / (ior1.x + ior1.y + ior1.z);
}

BSDF_INLINE void compute_eta(float3 &eta, const float3 &ior1, const float3 &ior2)
{
    eta = ior2 / ior1;
}

// variant of the above for Fresnel layering, replaces one of the IORs by
// the parameter of the layerer for weight computation
template<typename Data>
BSDF_INLINE float2 process_ior_fresnel_layer(Data *data, State *state, const float ior_param)
{
    const float3 material_ior = get_material_ior(state);
    
    if (data->ior1.x == BSDF_USE_MATERIAL_IOR)
        data->ior1 = material_ior;
    if (data->ior2.x == BSDF_USE_MATERIAL_IOR)
        data->ior2 = material_ior;

    //!! this property should be communicated by the renderer
    const bool outside =
        (material_ior.x == data->ior2.x) &&
        (material_ior.y == data->ior2.y) &&
        (material_ior.z == data->ior2.z);

    float2 ior = make_float2(
        outside ? (data->ior1.x + data->ior1.y + data->ior1.z) * (float)(1.0 / 3.0) : ior_param,
        outside ? ior_param : (data->ior2.x + data->ior2.y + data->ior2.z) * (float)(1.0 / 3.0));

    const float IOR_THRESHOLD = 1e-4f;
    const float ior_diff = ior.y - ior.x;
    if (math::abs(ior_diff) < IOR_THRESHOLD) {
        ior.y = ior.x + copysignf(IOR_THRESHOLD, ior_diff);
    }

    return ior;
}

// variant of the above for color Fresnel layering, replaces one of the IORs by
// the parameter of the layerer for weight computation
struct Color_fresnel_ior {
    float2 ior;
    float3 eta;
};
template<typename Data>
BSDF_INLINE Color_fresnel_ior process_ior_color_fresnel_layer(Data *data, State *state, const float3 &ior_param)
{
    const float3 material_ior = get_material_ior(state);
    
    if (data->ior1.x == BSDF_USE_MATERIAL_IOR)
        data->ior1 = material_ior;
    if (data->ior2.x == BSDF_USE_MATERIAL_IOR)
        data->ior2 = material_ior;

    //!! this property should be communicated by the renderer
    const bool outside =
        (material_ior.x == data->ior2.x) &&
        (material_ior.y == data->ior2.y) &&
        (material_ior.z == data->ior2.z);

    const float3 ior1 = outside ? data->ior1 : ior_param;
    const float3 ior2 = outside ? ior_param : data->ior2;

    Color_fresnel_ior ret_val;
    ret_val.eta = ior2 / ior1;
        
    ret_val.ior = make_float2(
        (ior1.x + ior1.y + ior1.z) * (float)(1.0 / 3.0),
        (ior2.x + ior2.y + ior2.z) * (float)(1.0 / 3.0));

    const float IOR_THRESHOLD = 1e-4f;
    const float ior_diff = ret_val.ior.y - ret_val.ior.x;
    if (math::abs(ior_diff) < IOR_THRESHOLD) {
        ret_val.ior.y = ret_val.ior.x + copysignf(IOR_THRESHOLD, ior_diff);
    }

    return ret_val;
}

// uniformly sample projected hemisphere
BSDF_INLINE float3 cosine_hemisphere_sample(
    const float2 &v)	// uniform numbers in [0, 1]
{
    if((v.x == 0.0f) && (v.y == 0.0f))
        return make_float3(0.0f, 1.0f, 0.0f); // Origin (prevent 0/0)

    // Map (x, y) to [-1, 1]^2 (while remapping (0, 0))
    float2 u = make_float2(v.x + v.x, v.y + v.y);
    if(u.x >= 1.0f)
        u.x -= 2.0f;
    if(u.y >= 1.0f)
        u.y -= 2.0f;

    // Map [-1, 1]^2 to hemisphere
    float r, phi;
    float y = 1.0f;
    if (u.x * u.x > u.y * u.y) { // use squares instead of absolute values like in Shirley/Chiu
        r = u.x;
        y -= u.x * u.x;
        phi = (float)(-M_PI / 4.0) * (u.y / u.x);
    } else {
        r = u.y;
        y -= u.y * u.y;
        phi = (float)(-M_PI / 2.0) + (float)(M_PI / 4.0) * (u.x / u.y);
    }

    if(y <= 0.0f)
        return make_float3(0.0f, 1.0f, 0.0f); // prevent vectors on plane

    float si, co;
    math::sincos(phi, &si, &co);

    // Compute the y component by "lifting" the point onto the unit hemisphere
    return make_float3(r * si, math::sqrt(y), r * co);
}

// Oren-Nayar diffuse component evaluation
BSDF_INLINE float eval_oren_nayar(
    const float3 	&k1,
    const float3	&k2,
    const float3 	&normal,
    const float         roughness)
{
    const float nk1 = math::dot(k1, normal);
    const float nk2 = math::dot(k2, normal);

    const float3 kp1 = k1 - normal * nk1;
    const float3 kp2 = k2 - normal * nk2;

    const float sigma2 = roughness * roughness;
    const float A = 1.0f - (sigma2 / (sigma2 * 2.0f + 0.66f));
    const float B = 0.45f * sigma2 / (sigma2 + 0.09f);

    // projection might give null-length vectors kp1 or/and kp2, check to avoid division by zero
    const float sl = math::dot(kp1, kp1) * math::dot(kp2, kp2);
    return A + (sl == 0.f ? 0.f : B *
                math::max(
                    0.0f,
                    math::dot(kp1, kp2) * math::sqrt((1.0f - nk1 * nk1) * (1.0f - nk2 * nk2) / sl)
                     / math::max(nk1, nk2)
                    )
        );
}

// Fresnel equation for an equal mix of polarization
BSDF_INLINE float ior_fresnel(
    const float eta,	// refracted / reflected ior
    const float kh)     // cosine between of angle normal/half-vector and direction
{
    float costheta = 1.0f - (1.0f - kh * kh) / (eta * eta);
    if (costheta < 0.0f)
        return 1.0f;
    costheta = math::sqrt(costheta); // refracted angle cosine

    const float n1t1 = kh;
    const float n1t2 = costheta;
    const float n2t1 = kh * eta;
    const float n2t2 = costheta * eta;
    const float r_p = (n1t2 - n2t1) / (n1t2 + n2t1);
    const float r_o = (n1t1 - n2t2) / (n1t1 + n2t2);
    const float fres = 0.5f * (r_p * r_p + r_o * r_o);

    return math::saturate(fres);
}

// Fresnel equation for an equal mix of polarization, with complex ior on transmitted side
// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations
BSDF_INLINE float complex_ior_fresnel(
    const float eta,   // transmitted ior / incoming ior
    const float eta_k, // transmitted extinction / incoming ior
    const float costheta)
{
    const float costheta2 = costheta * costheta;
    const float sintheta2 = 1.0f - costheta2;
    const float eta2 = eta * eta;
    const float eta_k2 = eta_k * eta_k;

    const float t0 = eta2 - eta_k2 - sintheta2;
    const float a2plusb2 = math::sqrt(t0 * t0 + 4.0f * eta2 * eta_k2);
    const float t1 = a2plusb2 + costheta2;
    const float a = math::sqrt(0.5f * (a2plusb2 + t0));
    const float t2 = 2.0f * a * costheta;
    const float rs = (t1 - t2) / (t1 + t2);

    const float t3 = costheta2 * a2plusb2 + sintheta2 * sintheta2;
    const float t4 = t2 * sintheta2;
    const float rp = rs * (t3 - t4) / (t3 + t4);

    return 0.5f * (rp + rs);
}

BSDF_INLINE float3 complex_ior_fresnel(
    const float3 &eta,   // transmitted ior / incoming ior
    const float3 &eta_k, // transmitted extinction / incoming ior
    const float costheta)
{
    return make_float3(
        complex_ior_fresnel(eta.x, eta_k.x, costheta),
        complex_ior_fresnel(eta.y, eta_k.y, costheta),
        complex_ior_fresnel(eta.z, eta_k.z, costheta));
}

// configurable Schlick-style Fresnel curve
BSDF_INLINE float custom_curve_factor(
    const float kh,
    const float exponent,
    const float normal_reflectivity,
    const float grazing_reflectivity)
{
    const float f = 1.0f - kh;
    float f5;
    if (exponent == 5.0f) {
        const float f2 = f * f;
        f5 = f2 * f2 * f;
    } else {
        f5 = math::pow(f, exponent);
    }
    return normal_reflectivity + (grazing_reflectivity - normal_reflectivity) * f5;
}

// color variant of the above
BSDF_INLINE float3 custom_curve_factor(
    const float kh,
    const float exponent,
    const float3 &normal_reflectivity,
    const float3 &grazing_reflectivity)
{
    const float f = 1.0f - kh;
    float f5;
    if (exponent == 5.0f) {
        const float f2 = f * f;
        f5 = f2 * f2 * f;
    } else {
        f5 = math::pow(f, exponent);
    }
    return normal_reflectivity + (grazing_reflectivity - normal_reflectivity) * f5;
}


// compute refraction direction
BSDF_INLINE float3 refract(
    const float3 &k,    // direction (pointing from surface)
    const float3 &n,    // normal
    const float b,	// (reflected side IOR) / (transmitted side IOR)
    const float nk,	// dot(n, k)
    bool &total_reflection)
{
    const float refract = b * b * (1.0f - nk * nk);
    total_reflection = (refract > 1.0f);
    return total_reflection ? (n * (nk + nk) - k)
        : math::normalize((-k * b + n * (b * nk - math::sqrt(1.0f - refract))));
}

// build orthonormal basis around bumped normal, attempting to keep tangent orientation
BSDF_INLINE bool get_bumped_basis(
    float3 &x_axis,
    float3 &z_axis,
    const float3 &aniso_axis,
    const float3 &shading_normal_bumped)
{
    z_axis = math::cross(aniso_axis, shading_normal_bumped);
    const float l = math::dot(z_axis, z_axis);
    if (l < 1e-8f)
        return false;
    z_axis /= math::sqrt(l);
    x_axis = math::cross(shading_normal_bumped, z_axis);
    return true;
}

// get geometry and shading normal, potentially flipped such that they point into the direction of
// a given outgoing direction
BSDF_INLINE void get_oriented_normals(
    float3 &shading_normal,
    float3 &geometry_normal,
    const float3 &shading_normal0,
    const float3 &geometry_normal0,
    const float3 &k1)
{
    geometry_normal = geometry_normal0 * copysignf(1.0f, math::dot(k1, geometry_normal0));
    shading_normal = shading_normal0 * copysignf(1.0f, math::dot(shading_normal0, geometry_normal));
}

struct Normals {
    float3 shading_normal;
    float3 geometry_normal;
};

struct Geometry {
    Normals n;
    float3 x_axis;
    float3 z_axis;
};

BSDF_INLINE bool get_geometry(
    Geometry &g,
    const float3 &shading_normal,
    const float3 &tangent_u,
    const float3 &k1,
    State *state)
{
    get_oriented_normals(
        g.n.shading_normal, g.n.geometry_normal, shading_normal, state->geometry_normal(), k1);
    return get_bumped_basis(g.x_axis, g.z_axis, tangent_u, g.n.shading_normal);
}

// compute half vector (convention: pointing to outgoing direction, like shading normal)
BSDF_INLINE float3 compute_half_vector(
    const float3 &k1,
    const float3 &k2,
    const float3 &shading_normal,
    const float2 &ior,
    const float nk2,
    const bool transmission,
    const bool no_refraction)
{
    float3 h;
    if (transmission) {
        if (no_refraction) {
            h = k1 + (shading_normal * (nk2 + nk2) + k2); // use corresponding reflection direction
        } else {
            h = k2 * ior.y + k1 * ior.x; // points into thicker medium
            if (ior.y > ior.x)
                h *= -1.0f; // make pointing to outgoing direction's medium
        }
    } else {
        h = k1 + k2;
    }
    return math::normalize(h);
}

// compute half vector (convention: pointing to outgoing direction, like shading normal),
// without actual refraction
BSDF_INLINE float3 compute_half_vector(
    const float3 &k1,
    const float3 &k2,
    const float3 &shading_normal,
    const float nk2,
    const bool transmission)
{
    float3 h;
    if (transmission)
        h = k1 + (shading_normal * (nk2 + nk2) + k2);
    else
        h = k1 + k2;
    return math::normalize(h);
}


// evaluate anisotropic Phong half vector distribution on the non-projected hemisphere
BSDF_INLINE float hvd_phong_eval(
    const float2 &exponent,
    const float nh,     // dot(shading_normal, h)
    const float ht,     // dot(x_axis, h)
    const float hb)     // dot(z_axis, h)
{
    const float p = nh > 0.99999f
        ? 1.0f
        : math::pow(nh, (exponent.x * ht * ht + exponent.y * hb * hb) / (1.0f - nh * nh));
    return math::sqrt((exponent.x + 1.0f) * (exponent.y + 1.0f)) * (float)(0.5/M_PI) * p;
}

// sample half vector according to anisotropic Phong distribution
BSDF_INLINE float3 hvd_phong_sample(
    const float2 &samples,
    const float2 &exponent)
{
    const float sy4 = samples.x*4.0f;
    const float cosupper = math::cos((float)M_PI * math::frac(sy4));

    const float2 e = make_float2(exponent.x + 1.0f, exponent.y + 1.0f);

    const float eu1mcu = e.x*(1.0f-cosupper);
    const float ev1pcu = e.y*(1.0f+cosupper);
    const float t      = eu1mcu+ev1pcu;

    const float tt   = (math::pow(1.0f-samples.y, -t/(e.x*e.y)) - 1.0f) / t;
    const float tttv = math::sqrt(ev1pcu*tt);
    const float tttu = math::sqrt(eu1mcu*tt);

    return math::normalize(make_float3(
            ((samples.x < 0.75f) && (samples.x >= 0.25f)) ? -tttv : tttv,
            1.0f,
            ((samples.x >= 0.5f)                          ? -tttu : tttu)));
}

// evaluate anisotropic sheen half vector distribution on the non-projected hemisphere
BSDF_INLINE float hvd_sheen_eval(
    const float inv_roughness,
    const float nh)     // dot(shading_normal, h)
{
    const float sin_theta2 = math::max(0.0f, 1.0f - nh * nh);
    const float sin_theta = math::sqrt(sin_theta2);
    return (inv_roughness + 2.0f) * math::pow(sin_theta, inv_roughness) * (float) (0.5 / M_PI) * nh;
}

// sample half vector according to anisotropic sheen distribution
BSDF_INLINE float3 hvd_sheen_sample(
    const float2 &samples,
    const float inv_roughness)
{
    const float phi = (float) (2.0 * M_PI) * samples.x;
    float sin_phi, cos_phi;
    math::sincos(phi, &sin_phi, &cos_phi);

    const float sin_theta = math::pow(1.0f - samples.y, 1.0f / (inv_roughness + 2.0f));
    const float cos_theta = math::sqrt(1.0f - sin_theta * sin_theta);

    return math::normalize(make_float3(
        cos_phi * sin_theta,
        cos_theta,
        sin_phi * sin_theta));
}

// evaluate anisotropic Beckmann distribution on the non-projected hemisphere
BSDF_INLINE float hvd_beckmann_eval(
    const float2 &alpha_inv,
    const float nh,     // dot(shading_normal, h)
    const float ht,     // dot(x_axis, h)
    const float hb)     // dot(z_axis, h)
{
    const float htalpha2 = ht * alpha_inv.x;
    const float hbalpha2 = hb * alpha_inv.y;
    const float nh2 = nh * nh;

    const float expt = math::exp((htalpha2 * htalpha2 + hbalpha2 * hbalpha2) / nh2);
    return (float)(1.0 / M_PI) / (expt * nh2 * nh) * alpha_inv.x * alpha_inv.y;
}

// sample half vector according to anisotropic Beckmann distribution on the non-projected hemisphere
BSDF_INLINE float3 hvd_beckmann_sample(
    const float2 &samples,
    const float2 &alpha_inv)
{
    const float phi = (float)(2.0 * M_PI) * samples.x;

    float cp, sp;
    math::sincos(phi, &sp, &cp);

    const float iax2 = alpha_inv.x * alpha_inv.x;
    const float iay2 = alpha_inv.y * alpha_inv.y;

    const float is = math::rsqrt(iax2 * cp * cp + iay2 * sp * sp);

    const float sp2 = alpha_inv.x * cp * is;
    const float cp2 = alpha_inv.y * sp * is;

    const float tantheta2 = -math::log(1.0f - samples.y) / (cp2 * cp2 * iax2 + sp2 * sp2 * iay2);
    const float sintheta = math::sqrt(tantheta2 / (1.0f + tantheta2));

    return make_float3(
            cp2 * sintheta,
            math::rsqrt(1.0f + tantheta2),
            sp2 * sintheta);
}


// https://stackoverflow.com/questions/35148198/efficient-faithfully-rounded-implementation-of-error
// -function-erff
BSDF_INLINE float erff(const float a)
{
    float r, s, t, u;
    #define fmaf(a,b,c) (a * b + c)
    t = math::abs(a);
    s = a * a;
    if (t > 0.921875f) {
        // 0.99527 ulp
        r = fmaf(0x1.222900p-16f, t, -0x1.91d2ccp-12f); // 1.72948930e-5, -3.83208680e-4
        u = fmaf(0x1.fd1336p-09f, t, -0x1.8d6300p-06f); // 3.88393435e-3, -2.42545605e-2
        r = fmaf(r, s, u);
        r = fmaf(r, t, 0x1.b55cb0p-4f); // 1.06777847e-1
        r = fmaf(r, t, 0x1.450aa0p-1f); // 6.34846687e-1
        r = fmaf(r, t, 0x1.079d0cp-3f); // 1.28717512e-1
        r = fmaf(r, t, t);
        r = math::exp(-r);
        r = 1.0f - r;
        r = copysignf(r, a);
    } else {
        // 0.99993 ulp
        r = -0x1.3a1a82p-11f;  // -5.99104969e-4
        r = fmaf(r, s, 0x1.473f48p-08f); //  4.99339588e-3
        r = fmaf(r, s, -0x1.b68bd2p-06f); // -2.67667342e-2
        r = fmaf(r, s, 0x1.ce1a46p-04f); //  1.12818025e-1
        r = fmaf(r, s, -0x1.8126e0p-02f); // -3.76124859e-1
        r = fmaf(r, s, 0x1.06eba6p-03f); //  1.28379151e-1
        r = fmaf(r, a, a);
    }
    #undef fmaf
    return r;
}

// from "Mike Giles - approximating the erfinv function"
BSDF_INLINE float erfinvf(const float x)
{
    float w, p;
    w = -math::log((1.0f - x) * (1.0f + x));
    if (w < 5.000000f) {
        w = w - 2.500000f;
        p =   2.81022636e-08f;
        p =   3.43273939e-07f + p*w;
        p =   -3.5233877e-06f + p*w;
        p =  -4.39150654e-06f + p*w;
        p =    0.00021858087f + p*w;
        p =   -0.00125372503f + p*w;
        p =   -0.00417768164f + p*w;
        p =      0.246640727f + p*w;
        p =       1.50140941f + p*w;
    } else {
        w = math::sqrt(w) - 3.000000f;
        p =  -0.000200214257f;
        p =   0.000100950558f + p*w;
        p =    0.00134934322f + p*w;
        p =   -0.00367342844f + p*w;
        p =    0.00573950773f + p*w;
        p =    -0.0076224613f + p*w;
        p =    0.00943887047f + p*w;
        p =       1.00167406f + p*w;
        p =       2.83297682f + p*w;
    }
    return p * x;
}


// sample visible (Smith-masked) half vector according to the anisotropic Beckmann distribution
// (see "Eric Heitz and Eugene d'Eon - Importance Sampling Microfacet-Based BSDFs with the
// Distribution of Visible Normals")
BSDF_INLINE float3 hvd_beckmann_sample_vndf(
    const float3 &k,
    const float2 &roughness,
    const float2 &xi)
{
    // stretch
    const float3 v = math::normalize(make_float3(k.x * roughness.x, k.y, k.z * roughness.y));

    float theta = 0.0f, phi = 0.0f;
    if (v.y < 0.99999f) {
        theta = math::acos(v.y);
        phi = math::atan2(v.z, v.x);
    }

    float slope_x, slope_y;
    {
        // "sample11()" as in "An Improved Visible Normal Sampling Routine for the Beckmann
        // Distribution" by Wenzel Jakob
        const float inv_sqrt_pi = 1.0f / math::sqrt((float)M_PI);

        const float tan_theta = math::tan(theta);
        const float cot_theta = 1.0f / tan_theta;

        float a = -1.0f;
        float c = erff(cot_theta);

        const float fit = 1.0f + theta * (-0.876f + theta * (0.4265f - theta * 0.0594f));
        float b = c - (1.0f + c) * math::pow(1.0f - xi.x, fit);

        const float normalization = 1.0f / (1.0f + c + inv_sqrt_pi * tan_theta *
                                            math::exp(-(cot_theta * cot_theta)));

        int iteration = 0;
        for (;;) {
            if (!(b >= a && b <= c))
                b = 0.5f * (a + c);

            slope_x = erfinvf(b);
            const float value = normalization * (1.0f + b + inv_sqrt_pi * tan_theta *
                                                 math::exp(-(slope_x * slope_x))) - xi.x;

            if (++iteration > 2 || math::abs(value) < 1e-5f)
                break;

            if (value > 0.0f)
                c = b;
            else
                a = b;

            const float derivative = (1.0f - slope_x * tan_theta) * normalization;
            b -= value / derivative;
        }

        slope_y = erfinvf(2.0f * xi.y - 1.0f);
    }

    // rotate
    float cp, sp;
    math::sincos(phi, &sp, &cp);
    const float tmp = cp * slope_x - sp * slope_y;
    slope_y = sp * slope_x + cp * slope_y;
    slope_x = tmp;

    // unstretch
    slope_x *= roughness.x;
    slope_y *= roughness.y;

    const float f = 1.0f / math::sqrt(slope_x * slope_x + slope_y * slope_y + 1.0f);
    return make_float3(-slope_x * f, f, -slope_y * f);
}



// sample anisotropic GGX distribution on the non-projected hemisphere
BSDF_INLINE float3 hvd_ggx_sample(
    const float2 &samples,
    const float2 &inv_alpha)
{
    const float phi = (float)(2.0 * M_PI) * samples.x;
    float cp, sp;
    math::sincos(phi, &sp, &cp);

    const float iax2 = inv_alpha.x * inv_alpha.x;
    const float iay2 = inv_alpha.y * inv_alpha.y;

    const float is = math::rsqrt(iax2 * cp * cp + iay2 * sp * sp);

    const float sp2 = inv_alpha.x * cp * is;
    const float cp2 = inv_alpha.y * sp * is;

    const float tantheta2 =
        samples.y / ((1.0f - samples.y) * (cp2 * cp2 * iax2 + sp2 * sp2 * iay2));
    const float sintheta = math::sqrt(tantheta2 / (1.0f + tantheta2));

    return make_float3(
            cp2 * sintheta,
            math::rsqrt(1.0f + tantheta2),
            sp2 * sintheta);
}

// evaluate anisotropic GGX distribution on the non-projected hemisphere
BSDF_INLINE float hvd_ggx_eval(
    const float2 &inv_alpha,
    const float nh,     // dot(shading_normal, h)
    const float ht,     // dot(x_axis, h)
    const float hb)     // dot(z_axis, h)
{
    const float x = ht * inv_alpha.x;
    const float y = hb * inv_alpha.y;
    const float aniso = x * x + y * y;

    const float f = aniso + nh * nh;
    return (float)(1.0 / M_PI) * inv_alpha.x * inv_alpha.y * nh / (f * f);
}

// sample visible (Smith-masked) half vector according to the anisotropic GGX distribution
// (see Eric Heitz - A Simpler and Exact Sampling Routine for the GGX Distribution of Visible
// normals)
BSDF_INLINE float3 hvd_ggx_sample_vndf(
    const float3 &k,
    const float2 &roughness,
    const float2 &xi)
{
    const float3 v = math::normalize(make_float3(k.x * roughness.x, k.y, k.z * roughness.y));

    const float3 t1 = (v.y < 0.99999f) ?
        math::normalize(math::cross(v, make_float3(0.0f, 1.0f, 0.0f))) :
        make_float3(1.0f, 0.0f, 0.0f);
    const float3 t2 = math::cross(t1, v);

    const float a = 1.0f / (1.0f + v.y);
    const float r = math::sqrt(xi.x);

    const float phi = (xi.y < a) ?
        xi.y / a * (float)M_PI : (float)M_PI + (xi.y - a) / (1.0f - a) * (float)M_PI;
    float sp, cp;
    math::sincos(phi, &sp, &cp);
    const float p1 = r * cp;
    const float p2 = r * sp * ((xi.y < a) ? 1.0f : v.y);

    float3 h = p1 * t1 + p2 * t2 + math::sqrt(math::max(0.0f, 1.0f - p1 * p1 - p2 * p2)) * v;

    h.x *= roughness.x;
    h.y = math::max(0.0f, h.y);
    h.z *= roughness.y;
    return math::normalize(h);
}


// sample a (1 - f)^n distribution on the unit disk
BSDF_INLINE float2 sample_disk_distribution(
    const float2 &samples,
    const float2 &exponent)
{
    const float phi_iso = (float)(2.0 * M_PI) * samples.y;
    float sp0, cp0;
    math::sincos(phi_iso, &sp0, &cp0);

    const float cp0sqr = cp0 * cp0;
    const float tmp0 = exponent.x - cp0sqr * exponent.x;
    const float tmp = tmp0 + cp0sqr * exponent.y;
    const float cp = (tmp != 0.0f) ? cp0 * math::sqrt(exponent.y / tmp) : -1.0f;
    const float tmp2 = sp0 * math::sqrt(exponent.x * tmp);
    const float sp = ((samples.y > 0.743f) && (samples.y < 0.757f)) ?
        -1.0f : ((tmp0 + tmp2) / (tmp + tmp2));


    const float n12 = exponent.x * cp * cp + exponent.y * sp * sp;

    const float r = math::sqrt(1.0f - math::pow(1.0f - samples.x, 1.0f / (n12 + 1.0f)));

    return make_float2(r * cp, r * sp);
}

// evaluate a (1 - f)^n distribution on the unit disk
BSDF_INLINE float eval_disk_distribution(
    const float x,
    const float y,
    const float2 &exponent)
{
    const float dx2 = x * x;
    const float dy2 = y * y;

    const float dxy2 = dx2 + dy2;

    const float f = 1.0f - dxy2;
    if (f < 0.0f)
        return 0.0f;

    const float p = dxy2 > 1e-8f ?
        math::pow(f, (exponent.x * dx2 + exponent.y * dy2) / dxy2) : 1.0f;
    return
        math::sqrt((exponent.x + 1.0f) * (exponent.y + 1.0f)) *
        (float)(1.0 / M_PI) * p;
}

// Cook-Torrance style v-cavities masking term
BSDF_INLINE float microfacet_mask_v_cavities(
    const float nh, // abs(dot(normal, half))
    const float kh, // abs(dot(dir, half))
    const float nk) // abs(dot(normal, dir))
{
    return math::min(1.0f, 2.0f * nh * nk / kh);
}

// Smith-masking for anisotropic GGX
BSDF_INLINE float microfacet_mask_smith_ggx(
    const float roughness_u,
    const float roughness_v,
    const float3 &k)
{
    const float ax = roughness_u * k.x;
    const float ay = roughness_v * k.z;

    const float inv_a_2 = (ax * ax + ay * ay) / (k.y * k.y);

    return 2.0f / (1.0f + math::sqrt(1.0f + inv_a_2));
}

// Smith-masking for anisotropic Beckmann
BSDF_INLINE float microfacet_mask_smith_beckmann(
    const float roughness_u,
    const float roughness_v,
    const float3 &k)
{
    const float ax = roughness_u * k.x;
    const float ay = roughness_v * k.z;

    const float a = k.y / math::sqrt(ax * ax + ay * ay);

#if 0
    // exact formula
    return 2.0f / (1.0f + erff(a) + 1.0f / (a * (float)(math::sqrt(M_PI))) * math::exp(-(a * a)));
#else
    // Walter's rational approximation
    if (a > 1.6f)
        return 1.0f;
    const float a2 = a * a;
    return (3.535f * a + 2.181f * a2) / (1.0f + 2.276f * a + 2.577f * a2);
#endif
}

// clamp roughness values such that the numerics for glossy BSDFs don't fall apart close to
// the perfect specular case
BSDF_INLINE float clamp_roughness(
    const float roughness)
{
    return math::max(roughness, 0.0000001f); // magic.
}

// convert roughness values to a similar Phong-style distribution exponent
BSDF_INLINE float2 roughness_to_exponent(const float roughness_u, const float roughness_v)
{
    return make_float2(2.0f / (roughness_u * roughness_u), 2.0f / (roughness_v * roughness_v));
}


// compute intensity of two superimposed waves at same frequency
// fake spectral color version of the baove
BSDF_INLINE float3 compute_interference_factor(
    const float i0, // intensity of unshifted wave
    const float i1, // intensity of shifted wave
    const float phase_shift_dist) // phase shift (in nm)
{
    const float aabb = i0 * i0 + i1 * i1;
    const float ab2 = 2.0f * i0 * i1;

    float3 xyz = make_float3(0.0f, 0.0f, 0.0f);


    //!! using low res color matching functions here
    const float lambda_min = 400.0f;
    const float lambda_step = (float)((700.0 - 400.0) / 16.0);
    const float3 cie_xyz[16] = {
        {0.02986f, 0.00310f, 0.13609f}, {0.20715f, 0.02304f, 0.99584f},
        {0.36717f, 0.06469f, 1.89550f}, {0.28549f, 0.13661f, 1.67236f},
        {0.08233f, 0.26856f, 0.76653f}, {0.01723f, 0.48621f, 0.21889f},
        {0.14400f, 0.77341f, 0.05886f}, {0.40957f, 0.95850f, 0.01280f},
        {0.74201f, 0.97967f, 0.00060f}, {1.03325f, 0.84591f, 0.00000f},
        {1.08385f, 0.62242f, 0.00000f}, {0.79203f, 0.36749f, 0.00000f},
        {0.38751f, 0.16135f, 0.00000f}, {0.13401f, 0.05298f, 0.00000f},
        {0.03531f, 0.01375f, 0.00000f}, {0.00817f, 0.00317f, 0.00000f}};

    for (unsigned int i = 0; i < 16; ++i) {
        const float lambda = lambda_min + (float)(i) * lambda_step;
        const float angle = float(2.0 * M_PI) * phase_shift_dist / lambda;

        const float intensity = aabb + ab2 * math::cos(angle);

        xyz += cie_xyz[i] * intensity;
    }

    xyz *= float(1.0 / 16.0);

    // ("normalized" such that the loop for no shifted wave gives reflectivity (1,1,1))
    return make_float3(
        xyz.x * (float)( 3.240600 / 0.433509) + xyz.y * (float)(-1.537200 / 0.433509)
                                              + xyz.z * (float)(-0.498600 / 0.433509),
        xyz.x * (float)(-0.968900 / 0.341582) + xyz.y * (float)( 1.875800 / 0.341582)
                                              + xyz.z * (float)( 0.041500 / 0.341582),
        xyz.x * (float)( 0.055700 / 0.326950) + xyz.y * (float)(-0.204000 / 0.326950)
                                              + xyz.z * (float)( 1.057000 / 0.326950));
}

// compute cosine of refracted direction
BSDF_INLINE float refraction_cosine(
    const float nk1,
    const float eta) // reflected ior / transmitted ior (assumed to be <= 1.0)
{
    const float eta_sqr = eta*eta;
    const float sintheta2_sqr = eta_sqr - nk1 * nk1 * eta_sqr;
    return math::sqrt(1.0f - sintheta2_sqr);
}


// compute the color tint caused by a thin-film coating
BSDF_INLINE float3 thin_film_factor(
    const float coating_ior,
    const float coating_thickness,
    const float2 ior,
    const float3 &k1,
    const float3 &k2,
    const float3 &normal,
    const bool transmission,
    const bool thin_walled)
{
    const float nk1 = math::dot(k1, normal);
    const float nk2 = math::abs(math::dot(k2, normal));

    const float3 h = compute_half_vector(
        k1, k2, normal, nk2,
        transmission);

    const float kh = math::abs(math::dot(k1, h));

    //!! clamped to avoid total internal reflection
    const float eta = math::min(ior.x, coating_ior) / coating_ior;

    const float costheta_refr = refraction_cosine(kh, eta);
    const float phase_shift_dist = 2.0f * coating_ior * costheta_refr * coating_thickness;
    const float coating_fresnel = ior_fresnel(1.0f / eta, kh);

    return compute_interference_factor(coating_fresnel, 1.0f - coating_fresnel, phase_shift_dist);
}

// evaluate measured color curve
BSDF_INLINE float3 measured_curve_factor(
    const float cosine,
    const float3 * values,
    const unsigned int num_values)
{
    const float angle01 = math::acos(math::min(cosine, 1.0f)) * (float)(2.0 / M_PI);
    const float f1 = angle01 * (float)(num_values - 1);
    const int idx0 = math::min((int)f1, (int)num_values - 1);
    const int idx1 = math::min(idx0 + 1, (int)num_values - 1);

    const float cw1 = f1 - (float)idx0;
    const float cw0 = 1.0f - cw1;

    return values[idx0] * cw0 + values[idx1] * cw1;
}

BSDF_INLINE float3 measured_curve_factor_eval(
    const float cosine,
    const float3 * values,
    const unsigned int num_values)
{
    if (num_values == 0)
        return make_float3(0, 0, 0);
    else
        return math::saturate(measured_curve_factor(cosine, values, num_values));
}

BSDF_INLINE float3 measured_curve_factor_eval(
    const float cosine,
    const unsigned int measured_curve_idx,
    const unsigned int num_values,
    State *state)
{
    if (num_values == 0)
        return make_float3(0, 0, 0);
    else {
        const float angle01 = math::acos(math::min(cosine, 1.0f)) * (float)(2.0 / M_PI);
        const float f1 = angle01 * (float)(num_values - 1);
        const int idx0 = math::min((int)f1, (int)num_values - 1);
        const int idx1 = math::min(idx0 + 1, (int)num_values - 1);

        const float cw1 = f1 - (float)idx0;
        const float cw0 = 1.0f - cw1;

        float3 res = state->get_measured_curve_value(measured_curve_idx, idx0) * cw0 +
            state->get_measured_curve_value(measured_curve_idx, idx1) * cw1;
        return math::saturate(res);
    }
}

BSDF_INLINE float measured_curve_factor_estimate(
    const float cosine,
    const float3 * values,
    const unsigned int num_values)
{
    if (num_values == 0)
        return 0;
    else
        return math::luminance(math::saturate(measured_curve_factor(
            cosine, values, num_values)));
}

BSDF_INLINE float3 color_measured_curve_factor_eval(
    const float cosine,
    const float3 * values,
    const unsigned int num_values,
    const float3 &weight)
{
    if (num_values == 0)
        return make_float3(0, 0, 0);
    else
        return math::saturate(weight) *
            math::saturate(measured_curve_factor(cosine, values, num_values));
}

BSDF_INLINE float color_measured_curve_factor_estimate(
    const float cosine,
    const float3 * values,
    const unsigned int num_values,
    const float3 &weight)
{
    if (num_values == 0)
        return 0;
    else
        return math::luminance(
            math::saturate(weight) *
            math::saturate(measured_curve_factor(cosine, values, num_values)));
}

BSDF_INLINE float sinh(float x)
{
    float ex = math::exp(x);
    return 0.5f * (ex - 1.f / ex);
}

#endif // MDL_LIBBSDF_UTILITIES_H
