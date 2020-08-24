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

#include <io/scene/mdl_elements/i_mdl_elements_utilities.h>


namespace MI {

namespace NEURAY {

namespace {

mi::neuraylib::IMessage::Kind convert_kind( MDL::Message::Kind kind)
{
    switch( kind) {
        case MDL::Message::MSG_COMILER_CORE:
            return mi::neuraylib::IMessage::MSG_COMILER_CORE;
        case MDL::Message::MSG_COMILER_BACKEND:
            return mi::neuraylib::IMessage::MSG_COMILER_BACKEND;
        case MDL::Message::MSG_COMPILER_ARCHIVE_TOOL:
            return mi::neuraylib::IMessage::MSG_COMPILER_ARCHIVE_TOOL;
        case MDL::Message::MSG_INTEGRATION:
            return mi::neuraylib::IMessage::MSG_INTEGRATION;
        case MDL::Message::MSG_IMP_EXP:
            return mi::neuraylib::IMessage::MSG_IMP_EXP;
        default:
            break;
    }
    return mi::neuraylib::IMessage::MSG_UNCATEGORIZED;
}

MDL::Message::Kind convert_kind( mi::neuraylib::IMessage::Kind kind)
{
    switch( kind) {
        case mi::neuraylib::IMessage::MSG_COMILER_CORE:
            return MDL::Message::MSG_COMILER_CORE;
        case mi::neuraylib::IMessage::MSG_COMILER_BACKEND:
            return MDL::Message::MSG_COMILER_BACKEND;
        case mi::neuraylib::IMessage::MSG_COMPILER_ARCHIVE_TOOL:
            return MDL::Message::MSG_COMPILER_ARCHIVE_TOOL;
        case mi::neuraylib::IMessage::MSG_INTEGRATION:
            return MDL::Message::MSG_INTEGRATION;
        case mi::neuraylib::IMessage::MSG_IMP_EXP:
            return MDL::Message::MSG_IMP_EXP;
        default:
            break;
    }
    return MDL::Message::MSG_UNCATEGORIZED;
}

} // namespace

class Message_impl
    : public mi::base::Interface_implement<mi::neuraylib::IMessage>
{
public:

    Message_impl(const MDL::Message& message) : m_message(message) { }

    // public API methods

    mi::base::Message_severity get_severity() const final
    {
        return m_message.m_severity;
    }

    Kind get_kind() const final
    {
        return convert_kind(m_message.m_kind);
    }

    const char* get_string() const final
    {
        return m_message.m_message.c_str();
    }

    mi::Sint32 get_code() const final
    {
        return m_message.m_code;
    }

    mi::Size get_notes_count() const final
    {
        return m_message.m_notes.size();
    }

    const IMessage* get_note(mi::Size index) const final
    {
        if (index >= m_message.m_notes.size())
            return nullptr;

        Message_impl* note = new Message_impl(m_message.m_notes[index]);
        return note;
    }

private:

    MDL::Message m_message;
};


Mdl_execution_context_impl::Mdl_execution_context_impl()
{
    m_context = new MDL::Execution_context();
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

void Mdl_execution_context_impl::clear_messages()
{
    m_context->clear_messages();
}

void Mdl_execution_context_impl::add_message(
    mi::neuraylib::IMessage::Kind kind,
    mi::base::Message_severity severity,
    mi::Sint32 code,
    const char* message)
{
    MDL::Message::Kind mdl_kind = convert_kind( kind);
    MDL::Message m( severity, message ? message : "", code, mdl_kind);
    m_context->add_message( m);
    if(    severity == mi::base::MESSAGE_SEVERITY_ERROR
        || severity == mi::base::MESSAGE_SEVERITY_FATAL)
        m_context->add_error_message( m);
}

MDL::Execution_context& Mdl_execution_context_impl::get_context() const
{
    return *m_context;
}

mi::Size Mdl_execution_context_impl::get_option_count() const
{
    return m_context->get_option_count();
}

const char* Mdl_execution_context_impl::get_option_name(mi::Size index) const
{
    if (index >= m_context->get_option_count())
        return nullptr;

    return m_context->get_option_name(index);
}

const char* Mdl_execution_context_impl::get_option_type(const char* name) const
{
    boost::any option;
    if (m_context->get_option(name, option) == -1)
        return nullptr;

    if (option.type() == typeid(bool))
        return "Boolean";
    if (option.type() == typeid(std::string))
        return "String";
    if (option.type() == typeid(mi::Float32))
        return "Float32";
    if (option.type() == typeid(mi::base::Handle<const mi::base::IInterface>))
        return "IInterface";
    return nullptr;
}

namespace {

template<typename T>
mi::Sint32 get_option_value(MDL::Execution_context* context, const char* name, T& value)
{
    boost::any option;
    if (context->get_option(name, option) == -1)
        return -1;

    if (option.type() != typeid(T))
        return -2;

    value = boost::any_cast<T>(option);
    return 0;
}

}

mi::Sint32 Mdl_execution_context_impl::get_option(const char* name, mi::Sint32& value) const
{
    return get_option_value(m_context, name, value);
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
    boost::any option;
    if (m_context->get_option(name, option) == -1)
        return -1;
    if (option.type() != typeid(std::string))
        return -2;

    const std::string& str_ref = boost::any_cast<const std::string&>(option);
    value = str_ref.c_str();
    return 0;
}

mi::Sint32 Mdl_execution_context_impl::get_option(
    const char* name, const mi::base::IInterface** value) const
{
    boost::any option;
    if (m_context->get_option(name, option) == -1)
        return -1;
    if (option.type() != typeid(mi::base::Handle<const mi::base::IInterface>))
        return -2;

    mi::base::Handle<const mi::base::IInterface> handle
        = boost::any_cast<const mi::base::Handle<const mi::base::IInterface>>(option);

    if (handle)
    {
        handle->retain();
        *value = handle.get();
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

mi::Sint32 Mdl_execution_context_impl::set_option(const char* name, mi::Sint32 value)
{
      return m_context->set_option(name, value);
}

mi::Sint32 Mdl_execution_context_impl::set_option(const char* name, mi::Float32 value)
{
      return m_context->set_option(name, value);
}

mi::Sint32 Mdl_execution_context_impl::set_option(const char* name, bool value)
{
    return m_context->set_option(name, value);
}

mi::Sint32 Mdl_execution_context_impl::set_option(
    const char* name, const mi::base::IInterface* value)
{
    mi::base::Handle<const mi::base::IInterface> handle(value, mi::base::DUP_INTERFACE);
    return m_context->set_option(name, handle);
}

MDL::Execution_context* unwrap_context(
    mi::neuraylib::IMdl_execution_context* context,
    MDL::Execution_context& default_context)
{
    if( !context)
        return &default_context;

    Mdl_execution_context_impl* context_impl
        = static_cast<Mdl_execution_context_impl*>(context);
    MDL::Execution_context& wrapped_context = context_impl->get_context();
    return &wrapped_context;
}

MDL::Execution_context* unwrap_and_clear_context(
    mi::neuraylib::IMdl_execution_context* context,
    MDL::Execution_context& default_context)
{
    MDL::Execution_context* result = unwrap_context( context, default_context);
    result->clear_messages();
    result->set_result( 0);
    return result;
}

} // namespace NEURAY

} // namespace MI
