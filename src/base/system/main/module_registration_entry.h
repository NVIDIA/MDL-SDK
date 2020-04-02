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
///
/// The registration functionality plus some base functions for all Module_registrations.

#ifndef BASE_SYSTEM_MAIN_MODULE_REGISTRATION_ENTRY_H
#define BASE_SYSTEM_MAIN_MODULE_REGISTRATION_ENTRY_H

#include "i_module.h"

#include <mi/base/lock.h>
#include <base/system/main/types.h>

namespace MI {
namespace SYSTEM {

//==================================================================================================

/// The possible states a module might be in.
enum Module_state
{
    MODULE_STATUS_UNINITIALIZED,
    MODULE_STATUS_STARTING,
    MODULE_STATUS_INITIALIZED,
    MODULE_STATUS_EXITED,
    MODULE_STATUS_FAILED
};

//==================================================================================================

/// Switch between two different Access_module implementations.
enum Module_variant { MANDATORY_MODULE, OPTIONAL_MODULE };

namespace detail
{

//==================================================================================================

/// This class acts as a linked list built up at static initialization time. Due to this fact
/// the order of initialization is exactly the order of the finalization. Using this fact a simple
/// single-linked list is sufficient when every new element is prepended and finally elements are
/// erased from the top of the list.
/// \note Replacing this list with a set-like structure would be preferable but will require
/// quite some brain to get the initialization order of all those static data right. If not, some
/// modules might register before the set-like structure gets constructed, whiping out the initially
/// registered data. One easier solution would be a Module_manager, which checks after main() if
/// it has translated this static structure into its own dynamic member, eg
/// std::map<string, Module_registration_entry>. This would open the way for (un)registration
/// during run-time. But all in all, for now the name-lookup requires linear time.
class Linked_list
{
  public:
    /// Prepend this object at the top of the list.
    Linked_list() : m_next(g_first) {
	g_first = this;
    }
    /// Erase this element from the list.
    ~Linked_list() {
	g_first = m_next;
    }

  protected:
    Linked_list* m_next;
    static Linked_list* g_first;
};

}

//==================================================================================================

/// The entry for a module. This class adds the book-keeping information to the \c Linked_list
/// object, eg ref counting, the status, and a name. Now, a module can get registered such that
/// it ends up as a registration entry in the linked list. Initially uninitialized, it can get
/// change this state through its members call_init(), call_exit().
class Module_registration_entry : private detail::Linked_list
{
  public:
    /// Constructor.
    /// \param the module's name
    Module_registration_entry(
	const char* name);
    /// Destructor.
    virtual ~Module_registration_entry();

    /// Retrieve the name.
    const char* get_name() const;
    /// Retrieve the status.
    Module_state get_status() const;
    /// Disable this module. This will disallow initialization.
    void disable();

    /// Virtual retrieval of the actually stored \c IModule reference.
    virtual IModule* get_module() = 0;

  protected:
    /// Initialize yourself for another user.
    void call_init();
    /// Finalize yourself. If ref count goes down to 0 tidy-up.
    void call_exit();

    /// Do the initialization.
    /// \return success
    virtual bool do_init() = 0;
    /// Do the finalization.
    virtual void do_exit() = 0;

  private:
    const char* m_name;					///< its name
    Uint m_reference_count;				///< its ref count
    Module_state m_status;				///< its status
    mi::base::Lock m_lock;				///< the lock
    bool m_enabled;					///< is the module enabled to work?

  public:
    /// Initialize. Make this module ready for a(nother) client.
    /// \param the module's name
    static Module_registration_entry* init_module(
	const char* name);
    /// Finalize.
    /// \param the module's name
    static void exit_module(
	const char* name);

    /// \name Debug_functionality
    /// The following functions offer some debugging functionality.
    //@{
    /// Find the Module_registration_entry for the given \p name.
    /// \param name the module's name
    /// \return the found entry or 0 else
    static Module_registration_entry* find(
	const char* name);
    /// Dump all the currently registered modules.
    static void dump_registered_modules();

    /// Count modules, which are still alive.
    static size_t count_alive_modules();

    /// Dump all currently alive modules.
    static void dump_alive_modules();
    //@}

    /// grant required access to init_module() and exit_module()
    template <typename T, Module_variant is_essential> friend class Access_module;
};

}
}

#endif
