//*****************************************************************************
// Copyright 2023 NVIDIA Corporation. All rights reserved.
//*****************************************************************************
/// \file mi/mdl/mdl_distiller_options.h
/// \brief Options for distiller rule applications
///
//*****************************************************************************

#ifndef MDL_DISTILLER_OPTIONS_H
#define MDL_DISTILLER_OPTIONS_H

namespace mi {
namespace mdl {

/// Options class to hold all parameters for algorithm and rule customizations.
class Distiller_options {
public:
    int                 verbosity;        ///< log level: 0 = off, 3 = show errs and warns
    int                 trace;            ///< 0=none
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
