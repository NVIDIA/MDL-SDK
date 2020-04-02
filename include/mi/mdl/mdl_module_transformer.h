/******************************************************************************
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \file mi/mdl/mdl_module_transformer.h
/// \brief Interfaces for transforming MDL modules
#ifndef MDL_MDL_MODULE_TRANSFORMER
#define MDL_MDL_MODULE_TRANSFORMER 1

#include <mi/base/iinterface.h>
#include <mi/base/interface_declare.h>
#include <mi/mdl/mdl_iowned.h>
#include <mi/mdl/mdl_messages.h>

namespace mi {
namespace mdl {

class IModule;

/// This interface gives access to the MDL module transformer.
class IMDL_module_transformer : public
    mi::base::Interface_declare<0x2c8f478e,0xfecd,0x443b,0xa8,0x2c,0xb9,0x9a,0x5f,0xc9,0x05,0xf4,
    mi::base::IInterface>
{
public:
    /// Inline all imports of a module, creating a new one.
    ///
    /// \param module       the module
    ///
    /// This function inlines ALL except standard library imports and produces a new module.
    /// The imported functions, materials, and types are renamed and only exported if visible
    /// in the interface.
    ///
    /// The annotation "origin(string)" holds the original full qualified name.
    virtual IModule const *inline_imports(IModule const *module) = 0;

    /// Inline all MDLE imports of a module, creating a new one.
    ///
    /// \param module       the module
    ///
    /// This function inlines ALL MDLE imports and produces a new module.
    /// The imported functions, materials, and types are renamed and only exported if visible
    /// in the interface.
    ///
    /// The annotation "origin(string)" holds the original full qualified name.
    virtual IModule const *inline_mdle(IModule const *module) = 0;

    /// Access messages of the last operation.
    virtual Messages const &access_messages() const = 0;
};

}  // mdl
}  // mi

#endif // MDL_MDL_MODULE_TRANSFORMER
