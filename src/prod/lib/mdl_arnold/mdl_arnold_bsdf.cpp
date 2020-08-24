/***************************************************************************************************
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "mdl_arnold_bsdf.h"
#include <mi/mdl_sdk.h>

AI_BSDF_EXPORT_METHODS(MdlBSDFData::methods);

namespace // some helper functions to convert vector types
{
    template<typename TVec> static TVec make(float val);
    template<> AI_DEVICE float3 make(float val) { return {val, val, val}; }

    AI_DEVICE static float3 convert(const AtVector& vec)
    {
        return float3{vec.x, vec.y, vec.z};
    }

    AI_DEVICE static AtVector convert(const float3& vec)
    {
        return AtVector{vec.x, vec.y, vec.z};
    }

    AI_DEVICE static float4x4 convert_ptr(const AtMatrix& mat)
    {
        return reinterpret_cast<float4x4>(&mat);
    }
} // anonymous

template <typename T>
T* align(T* ptr)
{
    return reinterpret_cast<T*>((reinterpret_cast<intptr_t>(ptr) + sizeof(T) - 1) & ~(sizeof(T) - 1));
}

void setup_mdl_state(const AtShaderGlobals* sg, Mdl_extended_state& ext_state)
{
    ext_state.outgoing = convert(-sg->Rd);      // outgoing direction
    ext_state.is_inside = (sg->Ngf != sg->Ng);  // ray is inside or outside the material (medium)

    // forward facing smooth normal without bumps
    // flip the shading towards the forward facing geometry normal
    AtVector Nsf = (sg->Ngf == sg->Ng) ? sg->Ns : -sg->Ns;

    // handle bad tessellated meshes where dot(shading_normal, -ray_dir) < 0
    AtVector k2 = sg->Rd - 2.0f * AiV3Dot(sg->Rd, Nsf) * Nsf;
    if (AiV3Dot(sg->Ngf, k2) < 0.0f)
        Nsf = sg->Ngf;

    // only one set of texture coordinates and tangents
#ifdef ENABLE_DERIVATIVES
    ext_state.uvw.val = { sg->u, sg->v, 0.0f };
    ext_state.uvw.dx.x = sg->dudx;
    ext_state.uvw.dx.y = sg->dudy;
    ext_state.uvw.dx.z = 0;
    ext_state.uvw.dy.x = sg->dvdx;
    ext_state.uvw.dy.y = sg->dvdy;
    ext_state.uvw.dy.z = 0;
#else
    ext_state.uvw = { sg->u, sg->v, 0.0f };
#endif

    // get tangents
    AtVector t = AtVector(0.0f, 0.0f, 0.0f);
    AtVector b = AtVector(0.0f, 0.0f, 0.0f);
    bool estimated = false;
    if (AiV3Dot(sg->dPdu, sg->dPdu) > AI_EPSILON)
    {
        // compute tangent frame by Gram-Schmidt
        AtVector du = AiV3Normalize(sg->dPdu);
        AtVector t1 = AiV3Normalize(du - Nsf * AiV3Dot(du, Nsf));
        AtVector b1 = AiV3Cross(Nsf, t1);
        t += t1;
        b += b1;
        estimated = true;
    }
    if (AiV3Dot(sg->dPdv, sg->dPdv) > AI_EPSILON)
    {
        AtVector dv = AiV3Normalize(sg->dPdv);
        AtVector b2 = AiV3Normalize(dv - Nsf * AiV3Dot(dv, Nsf));
        AtVector t2 = AiV3Cross(b2, Nsf);
        t += t2;
        b += b2;
        estimated = true;
    }
    if (estimated)
    {
        // average both estimates smooth the error, still off
        t = AiV3Normalize(t);
        b = AiV3Normalize(b);
    }
    else
    {
        // guess a system
        // without a proper UV layout the orientation of the tangent frame is meaningless anyway
        AiV3BuildLocalFrame(t, b, Nsf);
    }

    ext_state.tangent_u = convert(t);
    ext_state.tangent_v = convert(b);

    // setup mdl state
    ext_state.state.normal = convert(Nsf);
    ext_state.state.geom_normal = convert(sg->Ngf);
#ifdef ENABLE_DERIVATIVES
    ext_state.state.position.val = convert(sg->P);
    ext_state.state.position.dx = convert(sg->dPdx);
    ext_state.state.position.dy = convert(sg->dPdy);
#else
    ext_state.state.position = convert(sg->P);
#endif
    ext_state.state.animation_time = 0.0f;  // TODO: the time of the current sample in seconds,
                                            //       including the time within a shutter interval.
    ext_state.state.text_coords = &ext_state.uvw;
    ext_state.state.tangent_u = &ext_state.tangent_u;
    ext_state.state.tangent_v = &ext_state.tangent_v;
    ext_state.state.text_results = align(ext_state.texture_results);
    ext_state.state.ro_data_segment = nullptr;  // TODO: add read-only data handling if required
    ext_state.state.world_to_object = convert_ptr(sg->Minv);
    ext_state.state.object_to_world = convert_ptr(sg->M);
    ext_state.state.object_id = 0;  // TODO: objectID provided in a scene, and zero if none given

#ifdef APPLY_BUMP_SHADOW_WEIGHT
    ext_state.forward_facing_smooth_normal = Nsf;
#endif
}


bsdf_init
{
    MdlBSDFData& bsdf_data = *((MdlBSDFData*) AiBSDFGetData(bsdf));
    Mdl_extended_state* ext_state = &bsdf_data;
    setup_mdl_state(sg, *ext_state);

    // for thin-walled materials, there is no 'inside', so the thin_walled property has to be
    // evaluated to set the IORs correctly. As an optimization, we check if the property
    // is always true or false before generating code. If so, we initialize "is_thin_walled"
    // with the corresponding value, when creating the MdlBSDF structure.
    // Otherwise, code is generated for thin_walled and it is evaluated here at runtime.
    if (bsdf_data.shader.thin_walled_function_index != ~0)
    {
        bsdf_data.shader.target_code->execute(
            bsdf_data.shader.thin_walled_function_index,
            reinterpret_cast<mi::neuraylib::Shading_state_material&>(bsdf_data.state),
            /*texture_handler=*/ nullptr,  // allows to provide a custom texturing runtime
            /*arg_block_data=*/ nullptr,   // only relevant when using class-compilation mode
            &bsdf_data.is_thin_walled);
    }
    else
    {
        bsdf_data.is_thin_walled = bsdf_data.shader.thin_walled_constant;
    }

    // init distribution functions
    // here, potentially redundant texture accesses in sample/eval are pre-fetched (text_results).
    uint64_t bsdf_function_index = bsdf_data.is_inside
        ? bsdf_data.shader.backface_bsdf_function_index 
        : bsdf_data.shader.surface_bsdf_function_index;
    if (bsdf_function_index != ~0)
    {
        bsdf_data.shader.target_code->execute_bsdf_init(
            bsdf_function_index + 0,  // bsdf_function_index corresponds to 'init'
            reinterpret_cast<mi::neuraylib::Shading_state_material&>(bsdf_data.state),
            /*texture_handler*/ nullptr,
            /*arg_block_data=*/ nullptr);
    }

    // initialize the BSDF lobes
    // Currently, the entire BSDF is handled by one single lobe.
    // To work with LPEs, this has to change. An issue here will be the number of lobes
    // an arbitrary MDL material can have, which is #event_types x #labels.
    // MDL distinguishes 6 event types: Reflect and Transmit for Diffuse, Glossy and Specular,
    // while Specular covers discrete events only.
    // Since the AtBSDFLobeMask is an uint32_t, only 5 labels per material would be supported.
    AiBSDFInitLobes(bsdf, bsdf_data.lobe_info, 1);
    // TODO: AiBSDFInitLobes(bsdf, bsdf_data.lobe_info, 2);

    // reflection and transmission are both possible.
    AiBSDFInitNormal(bsdf, convert(bsdf_data.state.normal), false);
}


// create a (reduced precision) 2d uniform sample from a 1d uniform pseudo-random sample
// (by uniformly filling the space along a Morton curve)
static float2 to_uniform_2d(const float xi)
{
    const uint32_t n = (uint32_t)(xi * (float)(1ull << 32));
    // de-interleave n into x(16Bits),y(16Bits) (aka Morton or z-curve)
    uint32_t x = (n & 0x11111111) | ((n >> 1) & 0x22222222);
    uint32_t y = ((n >> 1) & 0x11111111) | ((n >> 2) & 0x22222222);
    x = (x & 0x03030303) | ((x >> 2) & 0x0C0C0C0C);
    y = (y & 0x03030303) | ((y >> 2) & 0x0C0C0C0C);
    x = (x & 0x000F000F) | ((x >> 4) & 0x00F000F0);
    y = (y & 0x000F000F) | ((y >> 4) & 0x00F000F0);
    x = (x & 0x000000FF) | ((x >> 8) & 0x0000FF00);
    y = (y & 0x000000FF) | ((y >> 8) & 0x0000FF00);

    const float2 ret = {
        x * (float)(1.0 / (double)(1 << 16)),
        y * (float)(1.0 / (double)(1 << 16))
    };

    return ret;
}

bsdf_sample
{
    MdlBSDFData& bsdf_data = *((MdlBSDFData*)AiBSDFGetData(bsdf));

    uint64_t bsdf_function_index = bsdf_data.is_inside
        ? bsdf_data.shader.backface_bsdf_function_index
        : bsdf_data.shader.surface_bsdf_function_index;
    if (bsdf_function_index != ~0)
    {
        mi::neuraylib::Bsdf_sample_data sample_data;  // input/output data for sample

        // assuming `air` as outside medium
        if (bsdf_data.is_inside && !bsdf_data.is_thin_walled) {
            sample_data.ior1 = make<float3>(MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR);
            sample_data.ior2 = make<float3>(1.0f);
        } else {
            sample_data.ior1 = make<float3>(1.0f);
            sample_data.ior2 = make<float3>(MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR);
        }
        sample_data.k1 = bsdf_data.outgoing;                    // outgoing direction
        const float2 rnd_zw = to_uniform_2d(rnd.z);             // 'create' a 4th random variable
        sample_data.xi = { rnd.x, rnd.y, rnd_zw.x, rnd_zw.y };  // pseudo-random sample number

        bsdf_data.shader.target_code->execute_bsdf_sample(
            bsdf_function_index + 1,                // bsdf_function_index corresponds to 'init'
                                                    // bsdf_function_index+1 to 'sample'
            &sample_data,   // input/output
            reinterpret_cast<mi::neuraylib::Shading_state_material&>(bsdf_data.state),
            /*texture_handler=*/ nullptr,
            /*arg_block_data=*/ nullptr);
        
        if (sample_data.event_type == mi::neuraylib::BSDF_EVENT_ABSORB)
            return AI_BSDF_LOBE_MASK_NONE;  // no valid sample

        // handle discrete events
        if ((sample_data.event_type & mi::neuraylib::BSDF_EVENT_SPECULAR) != 0)
        {
            // since the AI_BSDF_LOBE_SINGULAR is not set for the one single lobe,
            // we set 'infinite' probability instead
            sample_data.pdf = AI_BIG;

            // TODO: implement multiple lobe support
            // select the discrete reflection or the discrete transmission lobe
            if ((sample_data.event_type & mi::neuraylib::BSDF_EVENT_REFLECTION) != 0)
                out_lobe_index = 0; // TODO: use the specular reflection lobe
            else
                out_lobe_index = 0; // TODO: use the specular transmission lobe
        }
        else if ((sample_data.event_type & mi::neuraylib::BSDF_EVENT_GLOSSY) != 0)
            out_lobe_index = 0; // TODO: use the glossy lobe (or lobes for tags, reflect/refract)
        else
            out_lobe_index = 0;  // use the diffuse lobe (or lobes for tags, reflect/refract)

        // return output direction vectors, we don't compute differentials here
        out_wi = AtVectorDv(convert(sample_data.k2));

        // avoid artifacts when the shading normal differs significantly 
        // from the smooth surface normal
#ifdef APPLY_BUMP_SHADOW_WEIGHT
        const float bsw = AiBSDFBumpShadow(
            bsdf_data.forward_facing_smooth_normal,
            convert(bsdf_data.state.normal), 
            convert(sample_data.k2));
#else
        const float bsw = 1.0f;
#endif

        // return weight and PDF
        out_lobes[out_lobe_index] = AtBSDFLobeSample(
            AtRGB(bsw * sample_data.bsdf_over_pdf.x,
                  bsw * sample_data.bsdf_over_pdf.y,
                  bsw * sample_data.bsdf_over_pdf.z),
            sample_data.pdf,  // TODO: reverse PDF, MDL does not support this, yet
            sample_data.pdf);

        // indicate that we have a valid lobe sampled for the lobe at 'out_lobe_index'
        // TODO: returning unwanted lobes resulted in broken renderings in our experiments
        //return (1 << out_lobe_index);
        return lobe_mask & (1 << out_lobe_index);
    }

    // no BSDF, no contribution
    return AI_BSDF_LOBE_MASK_NONE;
}

bsdf_eval
{
    MdlBSDFData& bsdf_data = *((MdlBSDFData*)AiBSDFGetData(bsdf));

    // evaluate only non-singular lobes
    // TODO: for LPE support, the generated code needs to allow the evaluation
    //       of multiple lobes based on tags, returning diffuse and specular
    //       contribution from each of the selected lobes

    uint64_t bsdf_function_index = bsdf_data.is_inside
        ? bsdf_data.shader.backface_bsdf_function_index
        : bsdf_data.shader.surface_bsdf_function_index;
    if (bsdf_function_index != ~0)
    {
        // input/output data for evaluate
        mi::neuraylib::Bsdf_evaluate_data<mi::neuraylib::DF_HSM_NONE> eval_data;

        // assuming `air` as outside medium
        if (bsdf_data.is_inside && !bsdf_data.is_thin_walled) {
            eval_data.ior1 = make<float3>(MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR);
            eval_data.ior2 = make<float3>(1.0f);
        } else {
            eval_data.ior1 = make<float3>(1.0f);
            eval_data.ior2 = make<float3>(MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR);
        }
        eval_data.k1 = bsdf_data.outgoing;  // outgoing direction
        eval_data.k2 = convert(wi);         // incoming direction
        
        eval_data.bsdf_diffuse = make<float3>(0.0f);
        eval_data.bsdf_glossy = make<float3>(0.0f);

        bsdf_data.shader.target_code->execute_bsdf_evaluate(
            bsdf_function_index + 2,        // bsdf_function_index corresponds to 'init'
                                            // bsdf_function_index+2 to 'evaluate'
            &eval_data,
            reinterpret_cast<mi::neuraylib::Shading_state_material&>(bsdf_data.state),
            /*texture_handler=*/ nullptr,
            /*arg_block_data=*/ nullptr);

        // ensure, we got a valid result.
        // it may be invalid when samples below the surface are evaluated ...
        if (eval_data.pdf > 0.0f)
        {
            // return weight and PDF, same as in bsdf_sample

            // avoid artifacts when the shading normal differs significantly 
            // from the smooth surface normal
#ifdef APPLY_BUMP_SHADOW_WEIGHT
            const float bsw = AiBSDFBumpShadow(
                bsdf_data.forward_facing_smooth_normal,
                convert(bsdf_data.state.normal), 
                convert(sample_data.k2));
#else
            const float bsw = 1.0f;
#endif

            // diffuse contribution
            out_lobes[0] = AtBSDFLobeSample(
                AtRGB(bsw * (eval_data.bsdf_diffuse.x + eval_data.bsdf_glossy.x) / eval_data.pdf,
                      bsw * (eval_data.bsdf_diffuse.y + eval_data.bsdf_glossy.y) / eval_data.pdf,
                      bsw * (eval_data.bsdf_diffuse.z + eval_data.bsdf_glossy.z) / eval_data.pdf),
                eval_data.pdf,  // TODO: reverse PDF, MDL does not support this, yet
                eval_data.pdf);

            // TODO: glossy contribution
            // with one of the latest MDL SDK releases we provide glossy and diffuse BSDF values
            // separately. However, passing that out as second lobe did not work straight forward.
            // Probably because we can not sample the lobes individually. More investigation needed.
            // out_lobes[1] = AtBSDFLobeSample(
            //    AtRGB(eval_data.bsdf_glossy.x / eval_data.pdf,
            //        eval_data.bsdf_glossy.y / eval_data.pdf,
            //        eval_data.bsdf_glossy.z / eval_data.pdf),
            //        eval_data.pdf,  // TODO: reverse PDF, MDL does not support this, yet
            //        eval_data.pdf);

            return (1 << 0); // TODO: | (1 << 1);
            //return lobe_mask & ((1 << 0) | (1 << 1));
        }
    }

    // no BSDF, no contribution
    return AI_BSDF_LOBE_MASK_NONE;
}

bsdf_albedo
{
    // albedo is computed for the entire material, no splitting between diffuse and glossy
    // adding it to the diffuse lobe is probably the most sound solution
    if (lobe_mask != 1)
        return AtRGB(0.0f, 0.0f, 0.0f);

     MdlBSDFData& bsdf_data = *((MdlBSDFData*)AiBSDFGetData(bsdf));

    // evaluate only non-singular lobes
    // TODO: for LPE support, the generated code needs to allow the evaluation
    //       of multiple lobes based on tags, returning diffuse and specular
    //       contribution from each of the selected lobes

    uint64_t bsdf_function_index = bsdf_data.is_inside
        ? bsdf_data.shader.backface_bsdf_function_index
        : bsdf_data.shader.surface_bsdf_function_index;
    if (bsdf_function_index != ~0)
    {
        // input/output data for evaluate
        mi::neuraylib::Bsdf_auxiliary_data<mi::neuraylib::DF_HSM_NONE> aux_data;

        // assuming `air` as outside medium
        if (bsdf_data.is_inside && !bsdf_data.is_thin_walled) {
            aux_data.ior1 = make<float3>(MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR);
            aux_data.ior2 = make<float3>(1.0f);
        } else {
            aux_data.ior1 = make<float3>(1.0f);
            aux_data.ior2 = make<float3>(MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR);
        }
        aux_data.k1 = bsdf_data.outgoing;   // outgoing direction
        aux_data.albedo = make<float3>(0.0f);
        aux_data.normal = make<float3>(0.0f);

        bsdf_data.shader.target_code->execute_bsdf_auxiliary(
            bsdf_function_index + 4,        // bsdf_function_index corresponds to 'init'
                                            // bsdf_function_index+4 to 'auxiliary'
            &aux_data,
            reinterpret_cast<mi::neuraylib::Shading_state_material&>(bsdf_data.state),
            /*texture_handler=*/ nullptr,
            /*arg_block_data=*/ nullptr);

        return AtRGB(aux_data.albedo.x, aux_data.albedo.y, aux_data.albedo.z);
    }

    // no BSDF, no albedo
    return AtRGB(0.0f, 0.0f, 0.0f);
}
