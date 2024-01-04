/******************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
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

/// \file mi/mdl/mdl_distiller_options.h
/// \brief Options for distiller rule applications

#ifndef MDL_DISTILLER_OPTIONS_H
#define MDL_DISTILLER_OPTIONS_H

namespace mi {
namespace mdl {

/// Options class to hold all parameters for algorithm and rule customizations.
class Distiller_options {
public:
    int                 verbosity;        ///< log level: 0 = off, 3 = show errs and warns
    int                 trace;            ///< 0=none
    bool                debug_print;      ///< enable MDLTL debug_print statements
    bool                quiet;            ///< quiet, no output
    int                 bake;             ///< texture baking resolution, 0 = no baking
    bool                all_textures;     ///< bake all textures
    const char*         texture_dir;      ///< directory where to store textures
    bool                test_module;      ///< Test the new module by compiling it
    //
    float               top_layer_weight; ///< Weight in specific fresnel layer rules
    bool                layer_normal;     ///< Transfer layer normal maps to global one
    bool                merge_metal_and_base_color;
                                          ///< Merge metal and base color into one
    bool                merge_transmission_and_base_color;
                                          ///< Merge transmission and base color into one
    bool                target_material_model_mode;
                                          ///< Create distilling output material in target material mode mode

    /// Create options with default settings.
    Distiller_options()
        : verbosity(3),
          trace(0),
          debug_print(false),
          quiet(false),
          bake(0),
          all_textures(false),
          texture_dir("."),
          test_module(true),
          top_layer_weight(0.04f),
          layer_normal(true),
          merge_metal_and_base_color(true),
          merge_transmission_and_base_color(false),
          target_material_model_mode(false)
        {}
};

} // namespace mdl
} // namespace mi

#endif // MDL_DISTILLER_OPTIONS_H
