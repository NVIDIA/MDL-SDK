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

/** \file
 ** \brief Implementation of the IMdl_execution_context_impl interface
 **/

#include "pch.h"

#include "neuray_mdl_execution_context_impl.h"

#include <mi/neuraylib/idata.h>
#include <io/scene/mdl_elements/i_mdl_elements_utilities.h>


namespace MI {

namespace NEURAY {

class Message_impl
    : public mi::base::Interface_implement<mi::neuraylib::IMessage>
{
public:

    // constructor
    Message_impl(const MI::MDL::Message& message) : m_message(message) { }

    // public interface functions
    mi::base::Message_severity get_severity() const NEURAY_FINAL
    {
        return m_message.m_severity;
    }
    
    Kind get_kind() const NEURAY_FINAL {

        switch (m_message.m_kind)
        {
        case MDL::Message::MSG_COMILER_CORE:
            return MSG_COMILER_CORE;
        case MDL::Message::MSG_COMILER_BACKEND:
            return MSG_COMILER_BACKEND;
        case MDL::Message::MSG_COMPILER_ARCHIVE_TOOL:
            return MSG_COMPILER_ARCHIVE_TOOL;
        case MDL::Message::MSG_INTEGRATION:
            return MSG_INTEGRATION;
        case MDL::Message::MSG_IMP_EXP:
            return MSG_IMP_EXP;
        default:
            break;
        }
        return MSG_UNCATEGORIZED;
    }

    const char* get_string() const NEURAY_FINAL
    {
        return m_message.m_message.c_str();
    }
    
    mi::Sint32 get_code() const NEURAY_FINAL
    {
        return m_message.m_code;
    }

    mi::Size get_notes_count() const NEURAY_FINAL
    {
        return m_message.m_notes.size();
    }

    const IMessage* get_note(mi::Size index) const NEURAY_FINAL
    {
        if (index >= m_message.m_notes.size())
            return nullptr;
        
        Message_impl* note = new Message_impl(m_message.m_notes[index]);
        return note;
    }
    
private:

    MI::MDL::Message m_message;
};


Mdl_execution_context_impl::Mdl_execution_context_impl()
{
    m_context = new MI::MDL::Execution_context();
}

Mdl_execution_context_impl::~Mdl_execution_context_impl()
{
    delete m_context;
}

mi::Size Mdl_execution_context_impl::get_messages_count() const
{
    return m_context->get_messages_count();
}

mi::Size Mdl_execution_context_impl::get_error_messages_count() const
{
    return m_context->get_error_messages_count();
}

const mi::neuraylib::IMessage* Mdl_execution_context_impl::get_message(mi::Size index) const
{
     if (index >= m_context->get_messages_count())
            return nullptr;

    Message_impl* msg = new Message_impl(m_context->get_message(index));
    return msg;
}

const mi::neuraylib::IMessage* Mdl_execution_context_impl::get_error_message(mi::Size index) const
{
     if (index >= m_context->get_error_messages_count())
            return nullptr;

    Message_impl* msg = new Message_impl(m_context->get_error_message(index));
    return msg;
}

MI::MDL::Execution_context& Mdl_execution_context_impl::get_context() const
{
    return *m_context;
}

mi::Size Mdl_execution_context_impl::get_option_count() const
{
    return m_context->get_option_count();
}

const char* Mdl_execution_context_impl::get_option_name(mi::Size index) const 
{
    // index valid?
    if (index >= m_context->get_option_count())
        return 0;

    return m_context->get_option_name(index);
}

const char* Mdl_execution_context_impl::get_option_type(const char* name) const
{
    STLEXT::Any option;
    if (m_context->get_option(name, option) == -1)
        return 0;

    if (option.type() == typeid(bool))
        return "Boolean";
    if (option.type() == typeid(std::string))
        return "String";
    if (option.type() == typeid(mi::Float32))
        return "Float32";
    return 0;
}

namespace {

template<typename T>
mi::Sint32 get_option_value(MDL::Execution_context* context, const char* name, T& value)
{
    STLEXT::Any option;
    if (context->get_option(name, option) == -1)
        return -1;

    if (option.type() != typeid(T))
        return -2;

    value = STLEXT::any_cast<T>(option);
    return 0;
}

}

mi::Sint32 Mdl_execution_context_impl::get_option(const char* name, mi::Float32& value) const
{
    return get_option_value(m_context, name, value);
}

mi::Sint32 Mdl_execution_context_impl::get_option(const char* name, bool& value) const
{
    return get_option_value(m_context, name, value);
}

mi::Sint32 Mdl_execution_context_impl::get_option(const char* name, const char*& value) const
{
    STLEXT::Any option;
    if (m_context->get_option(name, option) == -1)
        return -1;
    if (option.type() != typeid(std::string))
        return -2;

    const std::string& str_ref = STLEXT::any_cast<const std::string&>(option);
    value = str_ref.c_str();
    return 0;
}

mi::Sint32 Mdl_execution_context_impl::get_option(
    const char* name, 
    mi::base::IInterface** value) const
{
    STLEXT::Any option;
    if (m_context->get_option(name, option) == -1)
        return -1;
    if (option.type() != typeid(mi::base::Handle<mi::base::IInterface>))
        return -2;

    mi::base::Handle<mi::base::IInterface> handle = 
        STLEXT::any_cast<mi::base::Handle<mi::base::IInterface>>(option);

    if (handle)
    {
        *value = handle.get();
        handle->retain();
    }
    else
    {
        *value = nullptr;
    }

    return 0;
}

mi::Sint32 Mdl_execution_context_impl::set_option(const char* name, const char* value)
{
    return m_context->set_option(name, std::string(value));
}

mi::Sint32 Mdl_execution_context_impl::set_option(const char* name, mi::Float32 value)
{
      return m_context->set_option(name, value);
}

mi::Sint32 Mdl_execution_context_impl::set_option(const char* name, bool value)
{
    return m_context->set_option(name, value);
}

mi::Sint32 Mdl_execution_context_impl::set_option(const char* name, mi::base::IInterface* value)
{
    mi::base::Handle<mi::base::IInterface> handle(mi::base::make_handle_dup(value));
    return m_context->set_option(name, handle);
}

MDL::Execution_context* unwrap_and_clear_context(
    mi::neuraylib::IMdl_execution_context* context,
    MDL::Execution_context& default_context)
{
    if (context)
    {
        NEURAY::Mdl_execution_context_impl* context_impl =
            static_cast<NEURAY::Mdl_execution_context_impl*>(context);
        if (context_impl) {
            MDL::Execution_context& wrapped_context = context_impl->get_context();
            wrapped_context.clear_messages();
            wrapped_context.set_result(0);
            return &wrapped_context;
        }
    }
    return &default_context;
}

} // namespace NEURAY

} // namespace MI
