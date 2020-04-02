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
/// \brief Registry, where every module must register itself before it can be used.

#include "module_registration_entry.h"
#include "module.h"

#include <base/system/stlext/i_stlext_concepts.h>
#include "i_assert.h"

// Try to ensure that this file is included in .cpp files only.
#ifdef MODULE_REGISTRATION_H
#error You are not allowed to include this file in any header.
#endif

#define MODULE_REGISTRATION_H


namespace MI {
namespace SYSTEM {

//==================================================================================================

/// The module, ie the IModule manager. This class ties together the registration and the
/// interface implementation. The member pointer \c m_module is handled through the internal
/// \c do_init() and \c do_exit() functions only, i.e. not through constructor or destructor.
/// \note Module_registration can only be instantiated for T's deriving from IModule. Why? Because
/// this allows for run-time checks whether or not the client tries to access the correct type.
/// \sa Access_module
template <typename T>
class Module_registration : public Module_registration_entry
{
  public:
    /// Constructor.
    /// \param mod_id the unique module's id for registration
    /// \param mod_name the four chars module's name for registration
    /// \param log_reg whether or not being registered for logging; certain mods don't need this
    Module_registration(
	Module_id mod_id,				/// the unique mod_id
	const char* mod_name,				/// the unique mod_name
	bool log_reg=true)				/// whether or not reg it for logging
      : Module_registration_entry(T::get_name()), m_module(0), m_mod_id(mod_id), m_log_reg(log_reg)
    {
	// check whether T is convertible to an IModule, ie if it is derived from the latter
	STLEXT::Derived_from<T, IModule>();

	if (m_log_reg)
	    // create appropriate entries for logging. Multiple registrations of the same module
	    // does not harm and is overwriting the values only.
	    Module::register_module(mod_id, mod_name);
    }

    /// Destructor.
    ~Module_registration()
    {
	if (m_log_reg)
	    // unregister the registered name from the Module
	    Module::unregister_module(m_mod_id);

        // If the m_module pointer is still valid, we cannot do much here anymore.
        // This destructor is called after main(), or in exit(), so all other
        // cleanup has already happened (in particular: logging and memory management).
        // So we do nothing here, no delete and no assert.
//	MI_ASSERT(m_module == 0);
//      delete m_module; // dangerous here!
	m_module = 0;
    }

    /// Retrieve the held module.
    /// \return the stored module reference
    IModule* get_module()
    {
	return m_module;
    }

  private:
    T* m_module;					///< the held module
    Module_id m_mod_id;					///< the corresponding Module_id
    bool m_log_reg;					///< whether or not this is log-registered

    /// Initialize the held module.
    /// \return the success of the initialization
    bool do_init()
    {
	MI_ASSERT(m_module == 0);
	m_module = new T;
	return m_module->init();
    }

    /// Finalize the held module.
    void do_exit()
    {
	if (m_module != 0)
	    m_module->exit();
	delete m_module;
	m_module = 0;
    }

    // disallow copying
    Module_registration(const Module_registration&);
    // disallow copying
    Module_registration& operator=(const Module_registration&);
};

}
}
