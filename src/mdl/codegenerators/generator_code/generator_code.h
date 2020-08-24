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

#ifndef MDL_GENERATOR_CODE_H
#define MDL_GENERATOR_CODE_H 1

#include <mi/base/handle.h>
#include <mi/mdl/mdl_mdl.h>
#include <mi/mdl/mdl_code_generators.h>

#include "mdl/compiler/compilercore/compilercore_cc_conf.h"
#include "mdl/compiler/compilercore/compilercore_allocator.h"
#include "mdl/compiler/compilercore/compilercore_mdl.h"
#include "mdl/compiler/compilercore/compilercore_options.h"

namespace mi {
namespace mdl {

///
/// Base class mixin for code generators.
///
template<typename I>
class Code_generator : public Allocator_interface_implement<I>
{
    typedef Allocator_interface_implement<I> Base;

public:
    /// Access options.
    Options_impl &access_options() MDL_FINAL { return m_options; }

protected:
    /// Constructor.
    ///
    /// \param alloc        The allocator.
    /// \param compiler     The mdl compiler.
    ///
    Code_generator(
        IAllocator *alloc,
        MDL        *compiler)
    : Base(alloc)
    , m_compiler(mi::base::make_handle_dup(compiler))
    , m_options(alloc)
    {
        m_options.add_option(
            MDL_CG_OPTION_INTERNAL_SPACE,
            "coordinate_world",
            "The internal space for which we compile");

        m_options.add_option(
            MDL_CG_OPTION_FOLD_METERS_PER_SCENE_UNIT,
            "true",
            "Whether occurrences of state::meters_per_scene_unit() and "
            "state::scene_units_per_meter() should be folded using the value of the "
            "meters_per_scene_unit option");

        m_options.add_option(
            MDL_CG_OPTION_METERS_PER_SCENE_UNIT,
            "1",
            "The value used for folding state::meters_per_scene_unit() and "
            "state::scene_units_per_meter() if enabled");

        m_options.add_option(
            MDL_CG_OPTION_WAVELENGTH_MIN,
            "380",
            "The smallest supported wavelength");

        m_options.add_option(
            MDL_CG_OPTION_WAVELENGTH_MAX,
            "780",
            "The largest supported wavelength");
    }

protected:
    /// The mdl compiler.
    mi::base::Handle<MDL> m_compiler;

    /// The compilation options.
    Options_impl m_options;
};

/// Check if two coordinate spaces are equal.
///
/// \param a                   One coordinate space.
/// \param b                   The other coordinate space.
/// \param internal_space      The name of the internal space.
/// \returns                   True if the coordinate spaces are equal and false otherwise.
bool equal_coordinate_space(
    IValue const *a,
    IValue const *b,
    char const   *internal_space);

/// Convert a IValue_texture::Bsdf_data_kind to a Resource_tag_tuple kind.
///
/// \param kind   the value's bsdf data kind
/// \param gamma  the value's gamma mode
Resource_tag_tuple::Kind kind_from_bsdf_data_kind(
    IValue_texture::Bsdf_data_kind kind,
    IValue_texture::gamma_mode     gamma);

/// Convert a Resource_tag_tuple to a IValue_texture::Bsdf_data_kind.
///
/// \param kind   the value's resource tag tuple kind
IValue_texture::Bsdf_data_kind bsdf_data_kind_from_kind(
    Resource_tag_tuple::Kind kind);

/// Convert a value to a Resource_tag_tuple kind.
///
/// \param val  the value
Resource_tag_tuple::Kind kind_from_value(IValue const *val);

}  // mdl
}  // mi

#endif // MDL_GENERATOR_CODE_H
