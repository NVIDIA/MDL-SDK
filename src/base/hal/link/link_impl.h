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
/// \brief

#ifndef BASE_HAL_LINK_LINK_IMPL_H
#define BASE_HAL_LINK_LINK_IMPL_H

#include "i_link.h"

#include <set>
#include <mi/base/interface_implement.h>
#include <base/lib/mem/i_mem_allocatable.h>

namespace MI {

namespace LINK {

/// Implementation of the Link_module interface
class Link_module_impl : public Link_module, public MEM::Allocatable
{
public:
    ILibrary* load_library( const char* path);

    // methods of SYSTEM::IModule

    bool init() { return true; }

    void exit() { }
};

/// Implementation of the Library interface
class Library_impl
  : public mi::base::Interface_implement<ILibrary>, public MEM::Allocatable
{
public:
    /// Constructor
    Library_impl( void* handle);

    // interface methods

    ~Library_impl();

    void* get_symbol( const char* symbol_name);

    std::string get_filename( const char* symbol_name);

private:
    /// The OS-specific handle the the dynamic library.
    void* m_handle;
};

} // namespace LINK

} // namespace MI

#endif // BASE_HAL_LINK_LINK_IMPL_H
