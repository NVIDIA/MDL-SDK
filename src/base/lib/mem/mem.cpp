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
/// \brief The memory module implementation.

#include "pch.h"

#include <base/lib/mem/mem.h>
#include <base/lib/log/i_log_assert.h>
#include <base/system/main/module_registration.h>

#include "mem_debug_alloc.h"

#if defined(DEBUG)
// define this for the new tracking memory allocator
//#define USE_DBG_ALLOC
#endif

#if defined(MI_PLATFORM_WINDOWS) && defined(_DEBUG)
// define this for extensive memory tests with the VC runtime
//#define WIN32_MEM_DEBUGGING_NOW
#endif

#ifdef WIN32_MEM_DEBUGGING_NOW
#include <crtdbg.h>
#endif

#if defined(BIT64)
mi_static_assert(sizeof(void *) == 8u);
#else
mi_static_assert(sizeof(void *) == 4u);
#endif

namespace MI {

namespace MEM {

#ifdef USE_DBG_ALLOC

// static memory for the debug allocator
union U {
    double unused;
    char   dbgMallocAllocator[sizeof(DBG::DebugMallocAllocator)];
} u;

/// The global debug allocator.
static DBG::DebugMallocAllocator *g_dbg = NULL;

/// Terminates the debug allocator, but check for memory leaks first.
static void dbg_exit() {
    g_dbg->~DebugMallocAllocator();
    g_dbg = NULL;
}

/// Get the debug memory allocator.
static DBG::DebugMallocAllocator *dbg_alloc() {
    if (g_dbg == NULL) {
        // Assume the very first allocation happens in single threaded code.
        // This typically happens on in the CRT startup already, so it is safe.
        new (u.dbgMallocAllocator) DBG::DebugMallocAllocator;

        g_dbg = (DBG::DebugMallocAllocator *)u.dbgMallocAllocator;

        // The following IS windows specific. We try to call the destructor "at the same scope"
        // as the constructor of the debug allocator to catch all delete calls inside static
        // objects. The Windows runtime does this by registering the destructor using
        //  atexit(). Hence we do this just when the allocator is created. Because the registration
        // is done in Windows at the end of a compiler generated wrapper AFTER the constructor call
        // it is ensured that the allocator destructor runs AFTER the last static object
        // constructor that had used allocation for the first time.
        atexit(dbg_exit);
    }
    return g_dbg;
}

#endif // USE_DBG_ALLOC

static SYSTEM::Module_registration<Mem_module> s_module(SYSTEM::M_MEM, "MEM");

SYSTEM::Module_registration_entry* Mem_module::get_instance()
{
    return s_module.init_module(s_module.get_name());
}

void Mem_module::set_exit_cb(void (*cb)())
{
    m_exit_memory_callback = cb;
}

bool Mem_module::init()
{
#ifdef WIN32_MEM_DEBUGGING_NOW
    //
    // Configure some memory checking in the VC runtime
    //
    // Get current flag
    int tmpFlag = _CrtSetDbgFlag( _CRTDBG_REPORT_FLAG );

    // VERY expensive:
    //tmpFlag |= _CRTDBG_CHECK_ALWAYS_DF;

    // Turn on leak-checking bit.
    tmpFlag |= _CRTDBG_LEAK_CHECK_DF;
    tmpFlag |= _CRTDBG_CHECK_CRT_DF;
    tmpFlag |= _CRTDBG_ALLOC_MEM_DF;
    tmpFlag |= _CRTDBG_DELAY_FREE_MEM_DF;

    // Clear the upper 16 bits and OR in the desired frequency
    tmpFlag = (tmpFlag & 0x0000FFFF) | _CRTDBG_CHECK_EVERY_1024_DF;

    // Set flag to the new value.
    _CrtSetDbgFlag( tmpFlag );
#endif //WIN32_MEM_DEBUGGING_NOW
    return true;
}

void Mem_module::exit()
{
    if (m_exit_memory_callback)
        m_exit_memory_callback();
}

void* alloc_aligned(const size_t size, const size_t alignment)
{
    // The memory block layout looks as follows:
    // [padding | size_t | memory].
    // The size_t holds the size in bytes of the padding area
    // at the beginning, in order for the start of 'memory' to be
    // aligned as requested.
    ASSERT(M_MEM, alignment % sizeof(size_t) == 0);
#ifdef USE_DBG_ALLOC
    char* const block =  (char *)dbg_alloc()->malloc(alignment - 1 + sizeof(size_t) + size);
#else
    char* const block = new char[alignment - 1 + sizeof(size_t) + size];
#endif
    mi_static_assert(sizeof(char*) == sizeof(size_t));
    const size_t address = reinterpret_cast<size_t>(block);
    const size_t alignment_delta = (address + sizeof(size_t)) % alignment;
    const size_t padding = alignment_delta ? alignment - alignment_delta : 0;
    ASSERT(M_MEM, (address + padding + sizeof(size_t)) % alignment == 0);
    size_t* const begin = reinterpret_cast<size_t*>(block + padding);
    *begin = padding;
    return static_cast<void*>(begin + 1);
}


void free_aligned(void* ptr)
{
    // Retrieve the alignment padding.
    size_t* const begin = reinterpret_cast<size_t*>(ptr) - 1;
    const size_t padding = *begin;
#ifdef USE_DBG_ALLOC
    dbg_alloc()->free(reinterpret_cast<char*>(begin) - padding);
#else
    delete [] (reinterpret_cast<char*>(begin) - padding);
#endif
}

} // namespace MEM

} // namespace MI

#ifdef USE_DBG_ALLOC

#ifdef MI_PLATFORM_WINDOWS
extern "C" __declspec(dllexport) void *neuray_alloc_block(const size_t size)
{
    return MI::MEM::dbg_alloc()->malloc(size);
}

extern "C" __declspec(dllexport) void neuray_free_block(void *memory)
{
    return MI::MEM::dbg_alloc()->free(memory);
}
#endif // MI_PLATFORM_WINDOWS

//--------------------------------------------------------------------------------------------------

// Overwritten global new operator
void* operator new(
    size_t size)			// requested size
    throw(std::bad_alloc)		// we don't throw bad_alloc
{
    return MI::MEM::dbg_alloc()->malloc(size);
}

//--------------------------------------------------------------------------------------------------

// Overwritten global delete operator
void operator delete(
    void* memory)		 	// memory to be released
    throw()				// avoid a warning on IRIX
{
    return MI::MEM::dbg_alloc()->free(memory);
}

//--------------------------------------------------------------------------------------------------

// Forward 'operator new []' to 'operator new'.
void * operator new [] (
    size_t	n)			// number of _bytes_ to allocate
    throw(std::bad_alloc)		// we don't throw bad_alloc
{
    return operator new(n);
}

//--------------------------------------------------------------------------------------------------

// Forward 'operator delete []' to 'operator delete'.
void operator delete [] (
    void *	ptr)			// pointer to to-be-freed memory
    throw()				// never throw
{
    operator delete(ptr);
}

//--------------------------------------------------------------------------------------------------

// Forward non-throwing 'operator new' to standard 'operator new'.
void * operator new(
    size_t	n,			// number of _bytes_ to allocate
    const std::nothrow_t &)		// dummy argument required by standard
    throw()				// never throw
{
    return operator new(n);
}

//--------------------------------------------------------------------------------------------------

// Forward non-throwing 'operator new []' to standard 'operator new'.
void * operator new [] (
    size_t	n,			// number of _bytes_ to allocate
    const std::nothrow_t &)		// dummy argument required by standard
    throw()				// never throw
{
    return operator new(n);
}

#if !defined(_SGI_COMPILER_VERSION)	// The Irix CC doesn't support
					// placement delete (a.k.a. "delete
					// with additional parameters"), so we
					// don't need to overload these
					// functions.

//--------------------------------------------------------------------------------------------------

// Forward non-throwing 'operator delete' to standard 'operator delete'.
void operator delete(
    void *	ptr, 			// pointer to to-be-freed memory
    const std::nothrow_t &)		// dummy argument required by standard
    throw()				// never throw
{
    operator delete(ptr);
}

//--------------------------------------------------------------------------------------------------

// Forward non-throwing 'operator delete []' to standard 'operator delete'.
void operator delete [] (
    void *	ptr,			// pointer to to-be-freed memory
    const std::nothrow_t &)		// dummy argument required by standard
    throw()				// never throw
{
    operator delete(ptr);
}

#endif // _SGI_COMPILER_VERSION

#endif // USE_DBG_ALLOC
