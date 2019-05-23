/******************************************************************************
 * Copyright (c) 2012-2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_INTERGRATION_MDLNR_H
#define MDL_INTERGRATION_MDLNR_H 1

#include <mdl/integration/mdlnr/i_mdlnr.h>

#include <mi/base/handle.h>

namespace mi { namespace base { class IAllocator; } }

namespace MI {

namespace MDLC {

/// The implementation of the MDL compiler module.
class Mdlc_module_impl : public Mdlc_module
{

public:

    /// Constructor.
    Mdlc_module_impl();

    bool init();

    void exit();

    mi::mdl::IMDL *get_mdl() const;

    void serialize_module(SERIAL::Serializer *serializer, const mi::mdl::IModule *module);

    const mi::mdl::IModule *deserialize_module(SERIAL::Deserializer *deserializer);

    void serialize_code_dag(
        SERIAL::Serializer *serializer, const mi::mdl::IGenerated_code_dag *code);

    const mi::mdl::IGenerated_code_dag *deserialize_code_dag(
        SERIAL::Deserializer *deserializer);

    void serialize_lambda_function(
        SERIAL::Serializer *serializer, const mi::mdl::ILambda_function *lambda);

    mi::mdl::ILambda_function *deserialize_lambda_function(
        SERIAL::Deserializer *deserializer);

    void set_used_with_mdl_sdk(bool flag);

    bool get_used_with_mdl_sdk() const;

    void client_build_version(const char* build, const char* bridge_protocol) const;

    mi::mdl::ICode_cache *get_code_cache() const;

    bool utf8_match(char const *file_mask, char const *file_name) const;

    bool get_implicit_cast_enabled() const;

    void set_implicit_cast_enabled(bool v);

private:

    /// Pointer to the MDL interface.
    mi::mdl::IMDL *m_mdl;

    /// The allocator that wraps new/delete (overloaded in MI::MEM) as IAllocator.
    mi::base::Handle<mi::base::IAllocator> m_allocator;

    /// Flag returned by #get_used_with_mdl_sdk().
    bool m_used_with_mdl_sdk;

    /// Flag that indicates whether #m_used_with_mdl_sdk was already explicitly set.
    bool m_used_with_mdl_sdk_set;

    /// The code cache used for JIT-generated source code.
    mi::mdl::ICode_cache *m_code_cache;

    /// Flag that indicates weather the integration should insert casts when needed (and possible).
    bool m_implicit_cast_enabled;
};

} // namespace MDLC

} // namespace MI

#endif // MDL_INTERGRATION_MDLNR_H
