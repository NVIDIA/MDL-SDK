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

#ifndef BASE_LIB_MEM_MEM_H
#define BASE_LIB_MEM_MEM_H

/// \file
/// \brief The memory module header.

#include <base/system/main/i_module.h>

namespace MI {
namespace SYSTEM { class Module_registration_entry; }
namespace MEM {

/// The module interface.
class Mem_module : public SYSTEM::IModule
{
public:
    /// Constructor.
    Mem_module() : m_exit_memory_callback(0) { }

    /// Destructor.
    ~Mem_module() { }

    /// Register a callback to be called from exit() of this module.
    /// \note This currently accepts only one callback function.
    /// \param cb the callback function, 0 can be used to uninstall the previous callback
    void set_exit_cb(void (*cb)());

    // methods of IModule
    
    static const char* get_name() { return "MEM"; }

    static SYSTEM::Module_registration_entry* get_instance();

    bool init();

    void exit();

private:
    void (*m_exit_memory_callback)();
};

} // namespace MEM
} // namespace MI


#endif // BASE_LIB_MEM_MEM_H
