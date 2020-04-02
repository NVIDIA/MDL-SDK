/******************************************************************************
 * Copyright (c) 2011-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \file mi/mdl/mdl_generated_code.h
/// \brief Interfaces for MDL annotations in the AST
#ifndef MDL_GENERATED_CODE_H
#define MDL_GENERATED_CODE_H 1

#include <mi/base/iinterface.h>
#include <mi/base/interface_declare.h>

#include <mi/mdl/mdl_definitions.h>
#include <mi/mdl/mdl_values.h>

namespace mi {
namespace mdl {

class IType;
class Messages;

/// A pair of ((resource_kind, resource_url), (resource_tag, resource_version)).
/// Used to map accessible resources to (tag, version) pair and as a key into resource
/// attribute maps.
struct Resource_tag_tuple {
    enum Kind {
        RK_BAD,                   ///< A bad value.
        RK_INVALID_REF,           ///< A invalid reference value.
        RK_TEXTURE_GAMMA_DEFAULT, ///< A texture value with default gamma.
        RK_TEXTURE_GAMMA_LINEAR,  ///< A texture value with linear gamma.
        RK_TEXTURE_GAMMA_SRGB,    ///< A texture value with SRGB gamma.
        RK_LIGHT_PROFILE,         ///< A light profile value.
        RK_BSDF_MEASUREMENT,      ///< A bsdf_measurement value.
        RK_STRING,                ///< A string value.

        // BSDF_DATA resource kinds
        RK_SIMPLE_GLOSSY_MULTISCATTER,
        RK_BACKSCATTERING_GLOSSY_MULTISCATTER,
        RK_BECKMANN_SMITH_MULTISCATTER,
        RK_GGX_SMITH_MULTISCATTER,
        RK_BECKMANN_VC_MULTISCATTER,
        RK_GGX_VC_MULTISCATTER,
        RK_WARD_GEISLER_MORODER_MULTISCATTER,
        RK_SHEEN_MULTISCATTER,
    };

    Kind        m_kind;    ///< The resource kind.
    int         m_tag;     ///< The assigned tag.
    char const  *m_url;    ///< The resource URL, NULL mapped to "".

    /// Default constructor.
    Resource_tag_tuple()
    : m_kind(RK_BAD)
    , m_tag(0)
    , m_url("")
    {}

    /// Constructor.
    Resource_tag_tuple(
        Kind        kind,
        char const  *url,
        int         tag)
    : m_kind(kind)
    , m_tag(tag)
    , m_url(url == NULL ? "" : url)
    {
    }
};

/// Generic interface for generated code of a MDL Core backend.
class IGenerated_code : public
    mi::base::Interface_declare<0x73d763a2,0x8038,0x45c2,0xbb,0x71,0x3e,0xc4,0x68,0x7e,0x7e,0x0c,
    mi::base::IInterface>
{
public:
    /// The possible kinds of generated code.
    enum Kind {
        CK_DAG,         ///< Generated DAG code.
        CK_PTX,         ///< Generated cuda PTX code.
        CK_GLSL,        ///< Generated GLSL code.
        CK_HLSL,        ///< Generated HLSL code.
        CK_LLVM_IR,     ///< Generated LLVM-IR code.
        CK_EXECUTABLE   ///< Generated (native CPU) executable code.
    };

    /// Get the kind of code generated.
    ///
    /// \returns    The kind of generated code.
    virtual Kind get_kind() const = 0;

    /// Get the target language.
    ///
    /// \returns    The name of the target language for which this code was generated.
    virtual char const *get_target_language() const = 0;

    /// Check if the code contents are valid.
    virtual bool is_valid() const = 0;

    /// Access messages.
    virtual Messages const &access_messages() const = 0;
};

/// Cast to subtype or return NULL if types do not match.
template<typename T>
T *as(IGenerated_code *type) {
    return (type->get_kind() == T::s_kind) ? static_cast<T *>(type) : NULL;
}

/// Cast to subtype or return NULL if types do not match.
template<typename T>
T const *as(IGenerated_code const *type) {
    return (type->get_kind() == T::s_kind) ? static_cast<T const *>(type) : NULL;
}

/// Convert a IValue_texture::Bsdf_data_kind to a Resource_tag_tuple kind.
///
/// \param kind  the value kind
Resource_tag_tuple::Kind kind_from_bsdf_data_kind(IValue_texture::Bsdf_data_kind kind);

/// Convert a value to a Resource_tag_tuple kind.
///
/// \param val  the value
Resource_tag_tuple::Kind kind_from_value(IValue const *val);

}  // mdl
}  // mi

#endif
