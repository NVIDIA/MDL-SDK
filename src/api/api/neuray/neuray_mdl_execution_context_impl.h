/***************************************************************************************************
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
 **************************************************************************************************/

#ifndef API_API_NEURAY_NEURAY_MDL_EXECUTION_CONTEXT_IMPL_H
#define API_API_NEURAY_NEURAY_MDL_EXECUTION_CONTEXT_IMPL_H

#include <mi/neuraylib/imdl_execution_context.h>
#include <mi/base/interface_implement.h>
#include <base/system/main/neuray_cc_conf.h>


namespace MI {

namespace MDL { class Execution_context; }

namespace NEURAY {

/// The execution context
class Mdl_execution_context_impl
  : public mi::base::Interface_implement<mi::neuraylib::IMdl_execution_context>
{
public:

    // constructor 
    Mdl_execution_context_impl();
    
    // destructor 
    virtual ~Mdl_execution_context_impl();
    
    // public interface functions
    mi::Size get_messages_count() const NEURAY_FINAL;

    mi::Size get_error_messages_count() const NEURAY_FINAL;

    const mi::neuraylib::IMessage* get_message(mi::Size index) const NEURAY_FINAL;

    const mi::neuraylib::IMessage* get_error_message(mi::Size index) const NEURAY_FINAL;


    mi::Size get_option_count() const NEURAY_FINAL;

    const char* get_option_name(mi::Size index) const NEURAY_FINAL;

    const char* get_option_type(const char* name) const NEURAY_FINAL;

    mi::Sint32 get_option(const char* name, const char*& value) const NEURAY_FINAL;

    mi::Sint32 get_option(const char* name, mi::Float32& value) const NEURAY_FINAL;

    mi::Sint32 get_option(const char* name, bool& value) const NEURAY_FINAL;

    mi::Sint32 get_option(const char* name, mi::base::IInterface** value) const NEURAY_FINAL;

    mi::Sint32 set_option(const char* name, const char* value) NEURAY_FINAL;

    mi::Sint32 set_option(const char* name, mi::Float32 value) NEURAY_FINAL;

    mi::Sint32 set_option(const char* name, bool value) NEURAY_FINAL;

    mi::Sint32 set_option(const char* name, mi::base::IInterface* value) NEURAY_FINAL;

    
    // own stuff

    MI::MDL::Execution_context& get_context() const;
    
private:

    MI::MDL::Execution_context* m_context;
};

/// Unwrap execution context.
MDL::Execution_context* unwrap_and_clear_context(
    mi::neuraylib::IMdl_execution_context* context,
    MDL::Execution_context& default_context);

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_NEURAY_MDL_EXECUTION_CONTEXT_IMPL_H
