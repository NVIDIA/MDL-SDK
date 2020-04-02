/******************************************************************************
 * Copyright (c) 2003-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \file
/// \brief list of all modules, Module base class for all modules.
///
/// We occasionally need module numbers, for example in base/data/log. List
/// them all here. This file sits above all other modules and can't be part
/// of one, so it lives in system/main. All module classes Mod_X must be
/// derived from Module.

#ifndef BASE_SYSTEM_MODULE_H
#define BASE_SYSTEM_MODULE_H

#include "i_module_id.h"

namespace MI {
namespace SYSTEM {

//
// the base class from which all modules must be derived. Here is a skeleton
// for the X.h header of a module X (where X could be net, serial, gap, etc):
//
//	namespace MI {
//	namespace X {
//	class Mod_X: public Module {
//	    static Module_status init()
//		ASSERT(M_X, !mod_X);
//		mod_X = new Mod_X;
//		mod_X->m_status = startup_code_that_may_fail();
//		return mod_X->m_status = ST_...;
//
//	    static void exit()
//		delete mod_X;
//		mod_X = 0;
//
//	    private: Mod_X() : Module(M_X, "X") 	// "X" is 4 chars max
//		m_xxx = default value;
//		mod_config->update(M_X, "xxx_name", "xxx_help", &m_xxx,
//								xxx_nmembers);
//	    public: somefunc()
//		m_xxx = newvalue;
//		mod_config->modified(&m_xxx);
//	};
//	}}
//	extern MI::X::Mod_X *mod_X;			// is defined in X.C
//
// For brevity, coding guidelines have been ignored in this example. The init
// and exit functions, and the constructor and destructor for X, uas well as
// the MI::X::Mod_X definition, should be in a separate file X.C by convention.
// Note the three-stage initialization:
//
// 1. the system init calls each module's init() function to create the module.
//    The advantage of this is that an error code can be returned. Subsystem
//    leader modules may init other subsystem member modules here; for example
//    base/data/data inits base/data/db, base/data/cluster, base/data/net, etc.
// 2. init() causes the module constructor to run, which initializes member
//    fields only, using Mod_config unless mod_config is still a null pointer.
// 3. init() calls other functions that may fail, like creating net connections
//    or starting threads, or calling other modules (other than Mod_config).
//    This can't be in the constructor because constructors can't return errors
//
// Note that step 2 and 3 are run by init() and so are purely internal things
// hidden from outside modules. init() is basically a module class factory that
// is called at most once per module. Similarly, there is an exit function that
// takes down the module (and any submodules that init() had created). It is
// necessary because the original init() caller doesn't know the submodules.
//

enum Module_status {
    ST_NOTINIT, 	// not yet initialized, or shut down, or not present
    ST_OK,		// initialized and fully functional
    ST_LIMITED, 	// initialized, recoverable errors, lacks functionality
    ST_FAIL		// failed to init, not functional, cannot use neuray
};

class Module
{
  public:
    // health of the module
    Module_status get_status() const;
    static Module_status get_status(
	Module_id	 mod);		// return status of this module

    // name of this module
    const char *get_name() const;

    // map module IDs to/from short module names for error msg etc, max 4 chars
    static const char *id_to_name(
	Module_id	 mod);		// convert M_X to "X"
    static Module_id name_to_id(
	const char	*name); 	// convert "X" to M_X

  protected:
    Module(				// constructor
	Module_id	 mod,		// module ID, one of M_*
	const char	*name); 	// module name, up to four characters

    virtual ~Module();			// destructor

    Module_status	 m_status;	// updated by init and constructor etc
    Module_id		 m_mod; 	// module ID from constructor
    const char		*m_name;	// module name from constructor, max 4
					// chars, must be all uppercase
  private:
    // register the given name for the given id.
    static void register_module(
	Module_id	 mod,		// module ID, one of M_*
	const char	*name); 	// module name, up to four characters
    // unregister the name for the given id.
    static void unregister_module(
	Module_id	 mod);		// module ID, one of M_*
    // all Module_registration classes are allowed to access register_module().
    template <typename T>
    friend class Module_registration;
};
}	// namespace SYSTEM

using namespace SYSTEM;

}	// namespace MI

#endif	// BASE_SYSTEM_MODULE_H
