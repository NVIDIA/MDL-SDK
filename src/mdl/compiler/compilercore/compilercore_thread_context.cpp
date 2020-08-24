/******************************************************************************
 * Copyright (c) 2014-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "pch.h"

#include "compilercore_thread_context.h"

namespace mi {
namespace mdl {

// Access compiler messages of last  operation.
Messages const &Thread_context::access_messages() const
{
    return m_msg_list;
}

// Set the module replacement path.
void Thread_context::set_module_replacement_path(
    char const *module_name,
    char const *file_name)
{
    if (module_name == NULL || file_name == NULL ||
        module_name[0] == '\0' || file_name[0] == '\0')
    {
        m_repl_module_name.clear();
        m_repl_file_name.clear();
    } else {
        m_repl_module_name = module_name;
        m_repl_file_name = file_name;
    }
}

// Access compiler options for the invocation.
Options_impl const &Thread_context::access_options() const
{
    return m_options;
}

// Access compiler options for the invocation.
Options_impl &Thread_context::access_options()
{
    return m_options;
}

// Access compiler messages of last  operation.
Messages_impl &Thread_context::access_messages_impl()
{
    return m_msg_list;
}

// Get a string buffer.
char const *Thread_context::get_string_buffer(String_buffer_id id) const
{
    switch (id) {
    case SBI_MANGLE_BUFFER:     return m_mangle_buffer.c_str();
    case SBI_DAG_MANGLE_BUFFER: return m_dag_buffer.c_str();
    }
    return NULL;
}

// Set a string buffer.
char const *Thread_context::set_string_buffer(String_buffer_id id, char const *s)
{
    switch (id) {
    case SBI_MANGLE_BUFFER:     m_mangle_buffer = s; return m_mangle_buffer.c_str();
    case SBI_DAG_MANGLE_BUFFER: m_dag_buffer    = s; return m_dag_buffer.c_str();
    }
    return NULL;
}

// Constructor.
Thread_context::Thread_context(
    IAllocator         *alloc,
    Options_impl const *options)
: Base(alloc)
, m_mangle_buffer(alloc)
, m_dag_buffer(alloc)
, m_msg_list(alloc, /*owner_fname=*/"")
, m_options(alloc)
, m_rrh(NULL)
, m_front_path(alloc)
, m_repl_module_name(alloc)
, m_repl_file_name(alloc)
, m_all_warnings_are_off(false)
{
    // copy options
    for (int i = 0, n = options->get_option_count(); i < n; ++i) {
        Options_impl::Option const &opt = options->get_option(i);

        if (opt.is_binary()) {
            m_options.add_binary_option(opt.get_name(), opt.get_description());
        } else {
            m_options.add_option(opt.get_name(), opt.get_default_value(), opt.get_description());
        }
    }
}

}  // mdl
}  // mi

