/***************************************************************************************************
 * Copyright (c) 2004-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief list of all modules, Module base class for all modules
///
/// Implements the Module base class that all modules are derived from. That's why this code
/// is in base/system/main, which is outside the normal module hierarchy. See module.h for info
/// on how to derive a module class. This file keeps a list of modules that have been initialized,
/// so that we can find out information on a module by module ID (which is an index into the
/// module list).

#include "pch.h"
#include <base/system/main/module.h>
#include <base/util/string_utils/i_string_utils.h>
#include <base/lib/log/log.h>

#include <cstring>

namespace MI {
namespace SYSTEM {

static Module *modules[NUM_OF_MODULES];


//
// List of names of modules which have no module class. This must be extended
// when adding such a module! The name of the module must be put at the array
// index corresponding to its module id.
//

static const char *g_module_names[NUM_OF_MODULES];


// Create one such class for each module which does not have a module class.
struct Static_module
{
    // The constructor puts the name of the module in the static array.
    Static_module(
	Module_id id,		// the id of the module
	const char *name)	// the name of the module
    : m_id(id)
    {
	ASSERT(M_MAIN, id >= 0 && id < NUM_OF_MODULES);
	g_module_names[id] = name;
    }

    // The destructor removes the name of the module from the static array.
    ~Static_module()
    {
	g_module_names[m_id] = NULL;
    }

    const Module_id m_id;		// for the destructor
};


//
// Convenience macro to put the name of a module in the table above
//

#define STATIC_MODULE(id, name) static const Static_module g_static_module_ ## id(id, name)


//
// create one such line for every module which does not have a module class
//

// start of modules without own module class
STATIC_MODULE(M_BACKENDS,	"BAKEND");
STATIC_MODULE(M_BRDF, 		"BRDF"  );
STATIC_MODULE(M_BSDF_MEASUREMENT, "BSDFM" );
STATIC_MODULE(M_BVH,            "BVH"   );
STATIC_MODULE(M_CACHEOPT,       "COPT"  );
STATIC_MODULE(M_CACHE_MANAGER,  "CACMGR");
STATIC_MODULE(M_CAMERA,		"CAMERA");
STATIC_MODULE(M_COLOR, 		"COLOR" );
STATIC_MODULE(M_CONT,		"CONT"  );
STATIC_MODULE(M_DB,		"DB"    );
STATIC_MODULE(M_DBIMAGE,	"DBIMAG");
STATIC_MODULE(M_DIFFEQ,		"DIFFEQ");
STATIC_MODULE(M_DISK,		"DISK"  );
STATIC_MODULE(M_DISK_CACHE,     "DISKCA");
STATIC_MODULE(M_DISPLACE, 	"DISPLA");
STATIC_MODULE(M_ENTY, 		"ENTY"  );
STATIC_MODULE(M_FFS, 		"FFS"   );
STATIC_MODULE(M_FRAMEBUFFER, 	"FBFR"  );
STATIC_MODULE(M_GCMP, 		"GCMP"  );
STATIC_MODULE(M_GDIFF,          "GDIFF" );
STATIC_MODULE(M_GPRIM, 		"GPRIM" );
STATIC_MODULE(M_GROUP,		"GROUP" );
STATIC_MODULE(M_PROXY,		"PROXY" );
STATIC_MODULE(M_HAL,		"HAL"   );
STATIC_MODULE(M_INSTANCE,	"INSTAN");
STATIC_MODULE(M_KDTREE,		"KDTREE");
STATIC_MODULE(M_LALG,		"LALG"  );
STATIC_MODULE(M_LIGHT,		"LIGHT" );
STATIC_MODULE(M_LIGHTPROFILE,	"LIGPRF");
STATIC_MODULE(M_LPEXPR,         "LPEXPR");
STATIC_MODULE(M_MAIN, 		"MAIN"  );
STATIC_MODULE(M_MESH,		"MESH"  );
STATIC_MODULE(M_NEURAY_API,	"API"   );
STATIC_MODULE(M_OPTIONS,	"OPTION");
STATIC_MODULE(M_PAGER, 		"PAGE"  );
STATIC_MODULE(M_PHT, 		"PHT"   );
STATIC_MODULE(M_RAL,		"RAL"   );
STATIC_MODULE(M_RDMA,		"RDMA"  );
STATIC_MODULE(M_SCH,		"SCH"   );
STATIC_MODULE(M_SIMP,		"SIMP"  );
STATIC_MODULE(M_STREAM,		"STREAM");
STATIC_MODULE(M_STREAMS,	"STREAMS");
STATIC_MODULE(M_STRING,		"STRING");
STATIC_MODULE(M_SWRCLIB,        "BAKE"  );
STATIC_MODULE(M_TEXTURE,	"TEXTUR");
STATIC_MODULE(M_TCPNET,		"TCPNET");
STATIC_MODULE(M_THREAD,		"THREAD");
STATIC_MODULE(M_THREAD_POOL,	"THRDPL");
STATIC_MODULE(M_TIME,		"TIME"  );
STATIC_MODULE(M_TRAVERSE_API,	"RNDAPI");
STATIC_MODULE(M_WELD,		"WELD"  );
STATIC_MODULE(M_XML, 		"XML"   );
STATIC_MODULE(M_ZLIB,		"ZLIB"  );
STATIC_MODULE(M_DIST,		"DIST"  );
STATIC_MODULE(M_BAKER,		"BAKER" );
STATIC_MODULE(M_POSTPROCESSING,	"POST"  );

// end of modules without own module class


//
// register the given name for the given id.
//

void Module::register_module(
    Module_id	mod,			// module ID, one of M_*
    const char	*name) 			// module name, up to four characters
{
    ASSERT(M_MAIN, name);
    ASSERT(M_MAIN, name && strlen(name) <= 6);
    g_module_names[mod] = name;
}


//
// unregister the name for the given id.
//

void Module::unregister_module(
    Module_id	mod)			// module ID, one of M_*
{
    g_module_names[mod] = NULL;
}


//
// health of this, or any, module. Tells callers whether the module exists
// and has succeeded to initialize. The module status is useful to find out
// if a module init() has failed, for example base/data/data failing to
// connect to the network.
//

Module_status Module::get_status() const
{
    return m_status;
}


Module_status Module::get_status(
    Module_id		 mod)		// return status of this module
{
    ASSERT(M_MAIN, mod >= 0 && mod < NUM_OF_MODULES);
    if (modules[mod] != NULL)
	return modules[mod]->get_status();

    // check for modules which have no module structure but a name
    if (g_module_names[mod] != NULL)
	// this is one such module. This is okay by definition.
	return ST_OK;

    return ST_NOTINIT;
}


//
// map module IDs to/from short module names for error msg etc, max 4 chars.
// This only works for modules that have been initialized because modules
// define their own names. Unknown modules return a null string. Note that
// the main module is typically a main() function, not a Module subclass, so
// catch that case explicitly.
//

const char *Module::get_name() const
{
    return m_name;
}


const char *Module::id_to_name(
    Module_id		 mod)		// convert M_X to "X"
{
    ASSERT(M_MAIN, mod >= 0 && mod < NUM_OF_MODULES);
    const char *res = modules[mod] ? modules[mod]->get_name() : g_module_names[mod];
    return res ? res : "UNKNOWN_MODULE";
}


Module_id Module::name_to_id(
    const char		*name)		// convert "X" to M_X
{
    int			mod;		// for iterating over modules

    for (mod=1; mod < NUM_OF_MODULES; mod++)
	if (!STRING::compare_case_insensitive(name, id_to_name((Module_id)mod), 4))
	    break;
    ASSERT(M_MAIN, mod != NUM_OF_MODULES);
    return (Module_id)mod;
}

//
// protected constructor, may only be used by the static init() factory of
// the derived module class. Performs some checks on the name (max 4 chars,
// all uppercase; will be used in error messages and config var name etc),
// and records the module in the module table. After construction, the module
// is still ST_NOTINIT, until the calling init() is done and sets it to ST_OK.
//

Module::Module(
    Module_id		mod,		// module ID, one of M_*
    const char		*name)		// module name, up to four characters
{
#ifdef DEBUG
    ASSERT(M_MAIN, mod >= 0 && mod < NUM_OF_MODULES);
    ASSERT(M_MAIN, !modules[mod]);		// no double inits
    int namelen;
    for (namelen=0; namelen < 5 && name[namelen]; namelen++)
	ASSERT(M_MAIN, (name[namelen] >= 'A' && name[namelen] <= 'Z') ||
		       (name[namelen] >= '0' && name[namelen] <= '9'));
    ASSERT(M_MAIN, namelen > 0 && namelen < 5);
#endif
    m_status = ST_NOTINIT;
    m_mod    = mod;
    m_name   = name;
    modules[mod] = this;
}


//
// destructor. No need to delete the name because it wasn't allocated, but
// need to clear the module slot in the module table to permit later re-init.
// Called from the static exit function of the derived module class only.
//

Module::~Module()
{
    ASSERT(M_MAIN, m_mod >= 0 && m_mod < NUM_OF_MODULES);
    ASSERT(M_MAIN, modules[m_mod]);
    modules[m_mod] = 0;
}

}
}
