/******************************************************************************
 * Copyright (c) 2017-2023, NVIDIA CORPORATION. All rights reserved.
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
/// \file options.h
/// \brief MDL Distiller option management class.

#pragma once

#include <vector>

const char SLASH =
#ifdef _WIN32
'\\';
#else
'/';
#endif

/// Default logfile filename in test mode
char const * const LOG_FILE = "mdl_distiller.log";

/// Default matching RUID filename in test mode
char const * const RUID_FILE = "rule_matches.txt";

/// Distilling targets used for '-test spec' option.
static char const * const spec_test_targets[]
= { "diffuse", "specular_glossy", "transmissive_pbr", "ue4", "lod"};

/// Returns the dimension of an array.
template<typename T, size_t n>
inline size_t dimension_of(T (&)[n]) { return n; }

enum error_codes {
    ERR_DISTILLING = 1,
    ERR_PRECONDITION = 2,
    ERR_POSTCONDITION = 3,
    ERR_MATERIAL = 4
};

/// Choice of what MDL Specification version is used to export the MDL.
/// mdl_spec_auto is last in the list so that checking against mdl_spec with >= always
/// triggers the highest version export in auto mode.
enum mdl_spec {
    mdl_spec_1_3,  ///< Force 1.3 output, casting all color version BSDFs to float versions
    mdl_spec_1_6,  ///< Force 1.6 version
    mdl_spec_1_7,  ///< Force 1.7 version
    mdl_spec_1_8,  ///< Force 1.7 version
    mdl_spec_auto  ///< Auto detect the needed version between 1.3, 1.6, 1.7 and 1.8
};

/// Options class to hold all parameters for algorithm and rule customizations.
class Options {
public:
    std::vector<const char*> paths;       ///< MDL search paths
    int                 verbosity;        ///< log level: 0 = off, 3 = show errs and warns
    int                 trace;            ///< 0=none
    bool                outline;          ///< DF node outline on stderr
    bool                quiet;            ///< quiet, no output on stderr
    std::string         out_filename;     ///< distilled material output filename, "-" = stdout
    bool                class_compilation;///< use class instead of instance compilation
    mdl_spec            export_spec;      ///< MDL version used for export
    std::string         dist_supp_file;   ///< Distiller support MDL module name
    int                 bake;             ///< texture baking resolution, 0 = no baking
    bool                all_textures;     ///< bake all textures
    std::string         texture_dir;      ///< directory where to store textures
    //
    bool                test_module;      ///< Test the new module by compiling it
    bool                test_suite;       ///< Called from mdl distiller test suite
    bool                spec_test;        ///< MDL specification test mode (default is false)
    std::string         test_dir;         ///< Folder where to store test results
    //
    std::string         top_layer_weight; ///< Weight in specific fresnel layer rules
    bool                layer_normal;     ///< Transfer layer normal maps to global one
    bool                merge_metal_and_base_color;
                                          ///< Merge metal and base color into one
    bool                merge_transmission_and_base_color;
                                          ///< Merge transmission and base color into one
    bool                emission;         ///< Export emission, or no emission if false.
    bool                target_material_model_mode; ///< Compile material in target material model mode.
    std::vector<const char*> additional_modules;
                                          ///< Modules to load (e.g. those defining custom target materials)

    /// Create options with default settings.
    Options()
        : verbosity(3)
        , trace(0)
        , outline(false)
        , quiet(false)
        , out_filename("-")
        , class_compilation(false)
        , export_spec( mdl_spec_auto)
        , dist_supp_file("::nvidia::distilling_support")
        , bake(0)
        , all_textures(false)
        , texture_dir(".")
        , test_module(true)
        , test_suite(false)
        , spec_test(false)
        , test_dir(".")
        , top_layer_weight("0.04")
        , layer_normal(true)
        , merge_metal_and_base_color(true)
        , merge_transmission_and_base_color(false)
        , emission(false)
        , target_material_model_mode(false)
        , additional_modules()
        {}
};
