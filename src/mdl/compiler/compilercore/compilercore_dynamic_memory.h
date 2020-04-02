/******************************************************************************
 * Copyright (c) 2012-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_COMPILERCORE_DYNAMIC_MEMORY_H
#define MDL_COMPILERCORE_DYNAMIC_MEMORY_H 1

#include "compilercore_cc_conf.h"

#include <mi/base/handle.h>
#include <base/lib/mem/i_mem_consumption.h>

#if MDL_STD_HAS_UNORDERED
#include <unordered_map>
#include <unordered_set>
#endif

namespace mi {
namespace mdl {

// inside MDL pointers never have dynamic memory
template<typename T>
inline bool has_dynamic_memory_consumption(T const *) { return false; }

template<typename T>
inline size_t dynamic_memory_consumption(T const *) { return 0; }

template<typename T>
inline bool has_dynamic_memory_consumption(mi::base::Handle<T> const &) { return false; }

template<typename T>
inline size_t dynamic_memory_consumption(mi::base::Handle<T> const &) { return 0; }

}  // mdl
}  // mi

#if MDL_STD_HAS_UNORDERED

namespace std {

template <class T1, class T2, class T3, class T4, class T5>
inline bool has_dynamic_memory_consumption(
    unordered_map<T1, T2, T3, T4, T5> const &)
{
    return true;
}

template <class T1 ,class T2, class T3, class T4, class T5>
inline size_t dynamic_memory_consumption(
    unordered_map<T1, T2, T3, T4, T5> const &the_map)
{
    typedef unordered_map<T1, T2, T3, T4, T5> Map_type;

    // bucket size is missing
    size_t total = sizeof(Map_type);

    // additional dynamic size of the map elements
    if (the_map.size() > 0) {
        bool dynamic_memory_T1 = has_dynamic_memory_consumption(the_map.begin()->first);
        bool dynamic_memory_T2 = has_dynamic_memory_consumption(the_map.begin()->second);
        if (dynamic_memory_T1 || dynamic_memory_T2) {
            typename Map_type::const_iterator it  = the_map.begin();
            typename Map_type::const_iterator end = the_map.end();
            for (; it != end; ++it) {
                if (dynamic_memory_T1) total += dynamic_memory_consumption(it->first);
                if (dynamic_memory_T2) total += dynamic_memory_consumption(it->second);
            }
        }
    }
    return total;
}

}  // MISTD
#endif

#endif
