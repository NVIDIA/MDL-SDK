/***************************************************************************************************
 * Copyright (c) 2007-2020, NVIDIA CORPORATION. All rights reserved.
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

/// \file
/// \brief The registration entry.

#include "pch.h"

#include "module_registration_entry.h"
#include "module.h"

#include <base/lib/log/i_log_assert.h>

#include <cstdio>
#include <cstring>
#include <cstdlib>

namespace MI
{
namespace SYSTEM
{

//==================================================================================================

// Initialization of static member.
detail::Linked_list* detail::Linked_list::g_first = 0;


//==================================================================================================

//--------------------------------------------------------------------------------------------------

// Constructor.
Module_registration_entry::Module_registration_entry(
    const char* name)					/// name of the module
  : m_name(name), m_reference_count(0), m_status(MODULE_STATUS_UNINITIALIZED), m_enabled(true)
{
//    fprintf(stderr, "Module \"%s\" is registered\n", name);
}


//--------------------------------------------------------------------------------------------------

// Destructor.
Module_registration_entry::~Module_registration_entry()
{
//    fprintf(stderr, "Unregister module \"%s\"\n", m_name);
}


//--------------------------------------------------------------------------------------------------

// Retrieve the name.
const char* Module_registration_entry::get_name() const
{
    return m_name;
}


//--------------------------------------------------------------------------------------------------

// Retrieve the status.
Module_state Module_registration_entry::get_status() const
{
    return m_status;
}


//--------------------------------------------------------------------------------------------------

// Initialize the given module. Make this module ready for a(nother) client.
Module_registration_entry* Module_registration_entry::init_module(
    const char* name)					/// name of the module
{
    Module_registration_entry* module = find(name);
    if (module)
	module->call_init();
    return module;
}


//--------------------------------------------------------------------------------------------------

// Finalize the given module.
void Module_registration_entry::exit_module(
    const char* name)					/// name of the module
{
    Module_registration_entry* module = find(name);
    if (module)
	module->call_exit();
}


//--------------------------------------------------------------------------------------------------

// Find the \c Module_registration_entry with the given \p name.
Module_registration_entry* Module_registration_entry::find(
    const char* name)					/// name of the module
{
    Module_registration_entry* module = static_cast<Module_registration_entry*>(g_first);
    while (module != 0) {
	if (strcmp(module->m_name, name) == 0) {
	    break;
	}
	module = static_cast<Module_registration_entry*>(module->m_next);
    }

    return module;
}


//--------------------------------------------------------------------------------------------------

// Initialize yourself for another user. For the first call create the interface impl.
void Module_registration_entry::call_init()
{
    mi::base::Lock::Block lock(&m_lock);

    if (m_reference_count++ == 0) {
	ASSERT(M_MAIN,
	    m_status == MODULE_STATUS_UNINITIALIZED ||
	    m_status == MODULE_STATUS_FAILED ||
	    m_status == MODULE_STATUS_EXITED);
	// switch state to STARTING
	m_status = MODULE_STATUS_STARTING;
	// virtual call to template method
	if (m_enabled && this->do_init())
	    m_status = MODULE_STATUS_INITIALIZED;
	else
	    m_status = MODULE_STATUS_FAILED;
    }
    else if (m_status == MODULE_STATUS_STARTING) {
	const char* name = this->get_name();
	//LOG_ERROR_ABOUT_CIRCULAR_DEPENDENCIES but log module is probably not initialized yet
	fprintf(stderr, "Module %s has circular dependencies. Please fix. Will abort now.", name);
	abort();
    }
}


//--------------------------------------------------------------------------------------------------

// Finalize yourself. If ref count goes down to 0, then tidy-up.
void Module_registration_entry::call_exit()
{
    mi::base::Lock::Block lock(&m_lock);
    if (m_reference_count == 0)
	return;
    if (--m_reference_count != 0)
	return;

    // virtual call to template method
    this->do_exit();
    m_status = MODULE_STATUS_EXITED;
}


//--------------------------------------------------------------------------------------------------

const char* enum_to_string(Module_state state)
{
    switch (state) {
	case MODULE_STATUS_UNINITIALIZED:
	    return "UNINITIALIZED";
	    break;
	case MODULE_STATUS_STARTING:
	    return "STARTING";
	    break;
	case MODULE_STATUS_INITIALIZED:
	    return "INITIALIZED";
	    break;
	case MODULE_STATUS_EXITED:
	    return "EXITED";
	    break;
	case MODULE_STATUS_FAILED:
	    return "FAILED";
	    break;
	default:
	    return "unknown";
	    break;
    }
}

// Dump all the currently registered modules.
void Module_registration_entry::dump_registered_modules()
{
    Module_registration_entry* module = static_cast<Module_registration_entry*>(g_first);
    while (module != 0) {
	fprintf(stderr, "Module %s\t(%s)\n",
	    module->get_name(), enum_to_string(module->get_status()));
	module = static_cast<Module_registration_entry*>(module->m_next);
    }
}


size_t Module_registration_entry::count_alive_modules()
{
    size_t n = 0;
    for (Module_registration_entry* module = static_cast<Module_registration_entry*>(g_first);
         module != 0; module = static_cast<Module_registration_entry*>(module->m_next))
    {
        if (module->m_reference_count > 0) {
            ++n;
        }
    }
    return n;
}


void Module_registration_entry::dump_alive_modules()
{
    // Emulate log message prefix to simplify message parsing.
    fprintf(stderr, "  1.0   MAIN main error: ------ Modules still alive:\n");
    for (Module_registration_entry* module = static_cast<Module_registration_entry*>(g_first);
         module != 0; module = static_cast<Module_registration_entry*>(module->m_next))
    {
        if (module->m_reference_count > 0) {
            fprintf(stderr, "    Module %s\t(%s) refcount:%u\n",
                module->get_name(), enum_to_string(module->get_status()), 
                module->m_reference_count);
        }
    }
    fprintf(stderr, "-----------\n");
}


// Disallow initialization.
void Module_registration_entry::disable()
{
    m_enabled = false;
}


}
}
