/******************************************************************************
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
/// \file mi/mdl/mdl_translator_plugin.h
/// \brief Interfaces for the MDL core compiler translator plugin.
#ifndef MDL_TRANSLATOR_PLUGIN_H
#define MDL_TRANSLATOR_PLUGIN_H 1

#include <mi/base/interface_declare.h>

namespace mi {
namespace mdl {

class IModule;
class IModule_cache;
class IThread_context;

/// An interface handling transparent conversions of foreign modules to MDL.
class IMDL_foreign_module_translator : public
    mi::base::Interface_declare<0x02713b54,0x6aac,0x4721,0x8f,0x52,0xbc,0xfe,0xa3,0x25,0xaf,0x1e,
    mi::base::IInterface>
{
public:
    /// Returns true if the given fully absolute MDL module name is a foreign module.
    ///
    /// \param module_name  an absolute MDL module name
    virtual bool is_foreign_module(char const *module_name) = 0;

    /// Translate a foreign module into a MDL module.
    ///
    /// \param ctx          the current thread context
    /// \param module_name  an absolute MDL module name
    /// \param cache        the current module cache
    ///
    /// \return the translated MDL module
    virtual IModule const *compile_foreign_module(
        IThread_context *ctx,
        char const      *module_name,
        IModule_cache   *cache) = 0;
};

} // mdl
} // mi

#endif // MDL_TRANSLATOR_PLUGIN_H
