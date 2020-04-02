/***************************************************************************************************
 * Copyright (c) 2011-2020, NVIDIA CORPORATION. All rights reserved.
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

/*
Copyright (c) 2002, 2008 Curtis Bartley
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

- Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the
distribution.

- Neither the name of Curtis Bartley nor the names of any other
contributors may be used to endorse or promote products derived
from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef BASE_LIB_MEM_MEMTRACK_H
#define BASE_LIB_MEM_MEMTRACK_H

/// Usage
/// =====
///
/// To enable this memory tracking tool you need to define MI_MEM_TRACKER and to make sure that
/// this file is included first such that it can redefine new as macro. One way to achieve this
/// is to add a file pch.h somewhere with the content
///
///     #define MI_MEM_TRACKER
///     #include <base/lib/mem/mem_memtrack.h>
///
/// and to make sure that the include directive for this directory is the first one. Furthermore,
/// add mem_memtrack.cpp to SRC in base/lib/mem/Makefile.
///
/// The methods list_memory_usage() and dump_blocks() can be used to query the current memory
/// usage (summary and explicit listing of all blocks).
///
/// Code changes
/// ============
///
/// The macro for new works fine with placement new, but not with custom definitions of the new
/// operator. Small local changes are usually enough to make it work.
///
/// Limitations
/// ===========
///
/// - Relies on overloading the global operators new and delete. Hence, it does not work in devsl
///   builds, and might not work for plugins.
/// - Memory allocated with malloc() and deallocated via free() is not tracked (can be done, but
///   requires manual code changes everywhere).
/// - Works only with g++ since a function from the internal g++ API is used to demangle the type
///   names.
/// - Does not work with some parts of the code base that redefines "new" as macro itself.
///   libneuray.so and libdice.so is supposed to work.
/// - Some allocations are tracked, but not stamped (probably because the macro "new" is not
///   defined). I suppose this happens for 3rd party code which does not include "pch.h" first.
/// - Some allocations are stamped as "void". This happens at least for STLport because the
///   actual allocation uses void* instead of T*. Stamping later is possible but requires some
///   patching inside STLport. Or, in general, the new operator in invoked directly, with a size
///   argument.

#ifdef MI_MEM_TRACKER

/// Note: avoid other includes which might pull in code where "new" is not yet defined as macro.

#include <typeinfo>

namespace MI {

namespace MEM {

/// Summarizes current memory usage.
void list_memory_usage();

/// Dumps all currently allocated blocks.
void dump_blocks();

/// Manually tracks an allocation not tracked otherwise (e.g. allocated with malloc()).
void* track_allocation (long unsigned int size); // Avoid size_t which needs an additional include.

/// Manually tracks a deallocation not tracked otherwise (e.g. deallocated with free()).
void track_deallocation (void* p);

/// Helper class to collect the file name and line number.
class Context
{
public:
    Context (const char* file_name, int line_number)
      : m_file_name (file_name), m_line_number (line_number) { }

    const char* const m_file_name;
    const int m_line_number;
};

/// Records the meta information (file name, line number, type name) in the memory block.
void stamp (void* p, const Context& context, const char* type_name);

/// operator* (const Context&, T*)
///
/// Calls the stamp() function above with the context and type name.
template <class T>
inline T* operator* (const Context& context, T* p)
{
    stamp (p, context, typeid(T).name());
    return p;
}

/// Helper macro that calls the (overloaded) new operator to track the allocation and
/// invokes the operator above to stamp the memory block.
#define MI_MEM_TRACKER_NEW MI::MEM::Context(__FILE__, __LINE__) * new

/// The defintion of "new" as macro.
#define new MI_MEM_TRACKER_NEW

} // namespace MEM

} // namespace MI

#endif // MI_MEM_TRACKER

#endif // BASE_LIB_MEM_MEMTRACK_H
