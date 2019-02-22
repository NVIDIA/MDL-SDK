/******************************************************************************
 * Copyright (c) 2011-2019, NVIDIA CORPORATION. All rights reserved.
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

namespace mi {
namespace mdl {

class IType;
class Messages;

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

}  // mdl
}  // mi

#endif
