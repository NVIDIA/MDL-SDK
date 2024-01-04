/******************************************************************************
 * Copyright (c) 2007-2023, NVIDIA CORPORATION. All rights reserved.
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
/// \brief  common module class for all regression tests

#ifndef BASE_SYSTEM_MAIN_TEST_MODULE_API_H
#define BASE_SYSTEM_MAIN_TEST_MODULE_API_H

#include "access_module.h"
#include <iosfwd>
#include <iostream>

namespace MI
{
namespace SYSTEM
{
class Module_registration_entry;
}
}

struct Test_module : public MI::SYSTEM::IModule
{
    virtual void run() = 0;

    static unsigned int n_impl1_runs;
    // Allow link time detection.
    static MI::SYSTEM::Module_registration_entry* get_instance();
};

namespace MI { namespace SYSTEM {

inline std::ostream & operator<< (std::ostream & os, MI::SYSTEM::Module_state st)
{
    using namespace MI::SYSTEM;
    switch (st)
    {
    case MODULE_STATUS_UNINITIALIZED:   return os << "uninitialized";
    case MODULE_STATUS_STARTING:        return os << "initialization starting";
    case MODULE_STATUS_INITIALIZED:     return os << "initialized";
    case MODULE_STATUS_EXITED:          return os << "exited";
    case MODULE_STATUS_FAILED:          return os << "failed";
    default:                            return os << "unknown module state '" << static_cast<int>(st) << "'";
    }
};

}}

#endif
