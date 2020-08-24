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

#include <string>
#include <mi/base/interface_implement.h>

namespace MI {

namespace MDL { class Execution_context; }

namespace NEURAY {

/// The execution context
class Mdl_execution_context_impl
  : public mi::base::Interface_implement<mi::neuraylib::IMdl_execution_context>
{
public:

    Mdl_execution_context_impl();

    virtual ~Mdl_execution_context_impl();

    // public API methods

    mi::Size get_messages_count() const final;

    mi::Size get_error_messages_count() const final;

    const mi::neuraylib::IMessage* get_message(mi::Size index) const final;

    const mi::neuraylib::IMessage* get_error_message(mi::Size index) const final;

    void clear_messages() final;

    void add_message(
        mi::neuraylib::IMessage::Kind kind,
        mi::base::Message_severity severity,
        mi::Sint32 code,
        const char* message) final;

    mi::Size get_option_count() const final;

    const char* get_option_name(mi::Size index) const final;

    const char* get_option_type(const char* name) const final;

    mi::Sint32 get_option(const char* name, const char*& value) const final;

    mi::Sint32 get_option(const char* name, mi::Sint32& value) const final;

    mi::Sint32 get_option(const char* name, mi::Float32& value) const final;

    mi::Sint32 get_option(const char* name, bool& value) const final;

    mi::Sint32 get_option(const char* name, const mi::base::IInterface** value) const final;

    mi::Sint32 set_option(const char* name, const char* value) final;

    mi::Sint32 set_option(const char* name, mi::Sint32 value) final;

    mi::Sint32 set_option(const char* name, mi::Float32 value) final;

    mi::Sint32 set_option(const char* name, bool value) final;

    mi::Sint32 set_option(const char* name, const mi::base::IInterface* value) final;

    // internal methods

    MDL::Execution_context& get_context() const;

private:

    MDL::Execution_context* m_context;
};

/// Unwraps the passed execution context, i.e., casts to the implementation class.
///
/// Returns address of \p default_context if \p context is \c NULL.
MDL::Execution_context* unwrap_context(
    mi::neuraylib::IMdl_execution_context* context,
    MDL::Execution_context& default_context);

/// Unwraps the passed execution context, i.e., casts to the implementation class, clears
/// all messages and sets the result to 0.
///
/// Uses \p default_context if \p context is \c NULL.
MDL::Execution_context* unwrap_and_clear_context(
    mi::neuraylib::IMdl_execution_context* context,
    MDL::Execution_context& default_context);

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_NEURAY_MDL_EXECUTION_CONTEXT_IMPL_H
