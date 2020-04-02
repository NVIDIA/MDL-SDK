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
/// \brief  accessor to a module

#ifndef BASE_SYSTEM_MAIN_ACCESS_MODULE_H
#define BASE_SYSTEM_MAIN_ACCESS_MODULE_H

#include "module_registration_entry.h"
#include "module.h"

#include <string>
#include <base/system/stlext/i_stlext_concepts.h>

namespace MI
{
namespace SYSTEM
{

//==================================================================================================

/// Accessing a module. This is the only way to gain access to a module. This class issues all
/// the required calls to get the module properly initialized and finally released. Initialization
/// and finalization can be achieved both automatically through constructor and destructor execution
/// or manually through explicit calls to set() and reset().
/// \code
/// Access_mode<MyInterface> myInterface(false);
/// if (myInterface.get_status() == MODULE_STATUS_INITIALIZED) {
///   // now we can use it
///   myInterface->foo();
/// }
/// \endcode
template <typename T, Module_variant is_essential=MANDATORY_MODULE>
class Access_module : private STLEXT::Non_copyable
{
  public:
    /// Default constructor.
    /// \param deferred shall the initialization be deferred?
    explicit Access_module(
	bool deferred=true);
    /// Destruction.
    ~Access_module();

    /// Setup the module.
    void set();
    /// Tear down the module.
    void reset();

    /// Retrieve the module's status.
    /// \return the current status of the module
    Module_state get_status() const;

    /// Access the underlying module.
    /// \return reference to this, could be 0
    T* operator->();
    /// Access the underlying module.
    /// \return const reference to this, could be 0
    const T* operator->() const;

    /// Retrieve whether the module was already initialized or not.
    /// \param name name of the module
    static bool is_module_initialized();

  private:
    /// the module pointer
    Module_registration_entry* m_module;
};

//--------------------------------------------------------------------------------------------------

// Default constructor. This leaves the module untouched.
template <typename T, Module_variant is_essential>
Access_module<T, is_essential>::Access_module(
    bool deferred)
  : m_module(0)
{
    if (!deferred)
	set();
}


//--------------------------------------------------------------------------------------------------

// Destruction. Finalize this usage of the module - in other words, decrement use count.
template <typename T, Module_variant is_essential>
Access_module<T, is_essential>::~Access_module()
{
    reset();
}


//--------------------------------------------------------------------------------------------------

// Setup the module with the given name. This might be called on an already setup Access_module.
template <typename T, Module_variant is_essential>
void Access_module<T, is_essential>::set()
{
    if (!m_module)
	m_module = T::get_instance();
}


//--------------------------------------------------------------------------------------------------

// Tear down the module with the given name.
template <typename T, Module_variant is_essential>
void Access_module<T, is_essential>::reset()
{
    if (m_module) {
	Module_registration_entry::exit_module(m_module->get_name());
	m_module = 0;
    }
}


//--------------------------------------------------------------------------------------------------

// Retrieve the module's status.
template <typename T, Module_variant is_essential>
Module_state Access_module<T, is_essential>::get_status() const
{
    if (m_module == 0)
	return MODULE_STATUS_UNINITIALIZED;
    // if finally everything got cleaned up
    if (m_module->get_status() == MODULE_STATUS_EXITED)
	return MODULE_STATUS_EXITED;
    // this usage of RTTI is in non-performance critical code and ensures safety of usage
    if (dynamic_cast<T*>(m_module->get_module()) == 0)
	return MODULE_STATUS_FAILED;
    return m_module->get_status();
}


//--------------------------------------------------------------------------------------------------

// Access the underlying module. Here we do a run-time check whether the stored module
// pointer is actually convertible to T.
template <typename T, Module_variant is_essential>
const T* Access_module<T, is_essential>::operator->() const
{
    // this usage of RTTI is in non-performance critical code and ensures safety of usage
    return dynamic_cast<const T*>(m_module->get_module());
}


//--------------------------------------------------------------------------------------------------

// Access the underlying module. Here we do a run-time check whether the stored module
// pointer is actually convertible to T.
template <typename T, Module_variant is_essential>
T* Access_module<T, is_essential>::operator->()
{
    // fall-back on the const version
    return const_cast<T*>(static_cast<const Access_module<T, is_essential>*>(this)->operator->());
}


//--------------------------------------------------------------------------------------------------

template <typename T, Module_variant is_essential>
bool Access_module<T, is_essential>::is_module_initialized()
{
    Module_registration_entry* entry = Module_registration_entry::find(T::get_name());
    return entry? entry->m_reference_count > 0 : false;
}


//==================================================================================================

/// Specialization for accessing the optional modules.
template <typename T>
class Access_module<T, OPTIONAL_MODULE> : private STLEXT::Non_copyable
{
  public:
    /// Default constructor.
    Access_module();
    /// Initializing constructor.
    /// \param name the name of the module
    Access_module(
	const std::string& name);
    /// Destruction.
    ~Access_module();

    /// Setup the module with the given name.
    /// \param name the name of the module
    void set(
	const std::string& name);
    /// Tear down the module with the given name.
    void reset();

    /// Retrieve the module's status.
    /// \return the status of the module
    Module_state get_status() const;

    /// Access to the underlying module. Note that this could be 0!
    /// \return reference to this, could be 0
    T* operator->();
    /// Const access to the underlying module. Note that this could be 0!
    /// \return const reference to this, could be 0
    const T* operator->() const;

    /// Retrieve whether the module was already initialized or not.
    /// \param name name of the module
    static bool is_module_initialized();

  private:
    /// pointer to the module
    Module_registration_entry* m_module;
};


//--------------------------------------------------------------------------------------------------

// Default constructor. This leaves the module untouched.
template <typename T>
Access_module<T, OPTIONAL_MODULE>::Access_module()
  : m_module(0)
{}


//--------------------------------------------------------------------------------------------------

// Accessing constructor. This (tries to) initializes the module.
template <typename T>
Access_module<T, OPTIONAL_MODULE>::Access_module(
    const std::string& name)
  : m_module(0)
{
    set(name);
}


//--------------------------------------------------------------------------------------------------

// Destruction. Finalize this usage of the module - in other words, decrement use count.
template <typename T>
Access_module<T, OPTIONAL_MODULE>::~Access_module()
{
    reset();
}


//--------------------------------------------------------------------------------------------------

// Setup the module with the given name. This might be called on an already setup Access_module.
// Do we want to allow this? Currently no. Why? Because otherwise we would need to reset() it
// first which might result in a finalization of the module + a following re-initialization of
// it. Since we do have dependencies among modules this might result in undefined behaviour.
template <typename T>
void Access_module<T, OPTIONAL_MODULE>::set(
    const std::string& name)
{
    if (!m_module)
	m_module = Module_registration_entry::init_module(name.c_str());
}


//--------------------------------------------------------------------------------------------------

// Tear down the module with the given name.
template <typename T>
void Access_module<T, OPTIONAL_MODULE>::reset()
{
    if (m_module) {
	Module_registration_entry::exit_module(m_module->get_name());
	m_module = 0;
    }
}


//--------------------------------------------------------------------------------------------------

// Retrieve the module's status.
template <typename T>
Module_state Access_module<T, OPTIONAL_MODULE>::get_status() const
{
    if (m_module == 0)
	return MODULE_STATUS_UNINITIALIZED;
    // if finally everything got cleaned up
    if (m_module->get_status() == MODULE_STATUS_EXITED)
	return MODULE_STATUS_EXITED;
    // this usage of RTTI is in non-performance critical code and ensures safety of usage
    if (dynamic_cast<T*>(m_module->get_module()) == 0)
	return MODULE_STATUS_FAILED;
    return m_module->get_status();
}


//--------------------------------------------------------------------------------------------------

// Access the underlying module. Here we do a run-time check whether the stored module
// pointer is actually convertible to T.
template <typename T>
T* Access_module<T, OPTIONAL_MODULE>::operator->()
{
    // this usage of RTTI is in non-performance critical code and ensures safety of usage
    return dynamic_cast<T*>(m_module->get_module());
}


//--------------------------------------------------------------------------------------------------

// Const access to the underlying module. Here we do a run-time check whether the stored module
// pointer is actually convertible to T.
template <typename T>
const T* Access_module<T, OPTIONAL_MODULE>::operator->() const
{
    // this usage of RTTI is in non-performance critical code and ensures safety of usage
    return const_cast<const T*>(dynamic_cast<T*>(m_module->get_module()));
}

//--------------------------------------------------------------------------------------------------

template <typename T>
bool Access_module<T, OPTIONAL_MODULE>::is_module_initialized()
{
    Module_registration_entry* entry = Module_registration_entry::find(T::get_name());
    return entry? entry->m_reference_count > 0 : false;
}

}
}

#endif
