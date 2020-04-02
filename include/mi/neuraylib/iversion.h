/***************************************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief Interface for accessing version information.

#ifndef MI_NEURAYLIB_IVERSION_H
#define MI_NEURAYLIB_IVERSION_H

#include <mi/base/interface_declare.h>

namespace mi {

namespace neuraylib {

/// Abstract interface for accessing version information.
class IVersion : public
    mi::base::Interface_declare<0xe8f929df,0x6c1e,0x4ed5,0xa6,0x17,0x29,0xa6,0xb,0x12,0xdb,0x48>
{
public:

    /// Returns the product name.
    virtual const char* get_product_name() const = 0;

    /// Returns the product version.
    virtual const char* get_product_version() const = 0;

    /// Returns the build number.
    virtual const char* get_build_number() const = 0;
    
    /// Returns the build date.
    virtual const char* get_build_date() const = 0;
    
    /// Returns the platform the library was built on.
    virtual const char* get_build_platform() const = 0;

    /// Returns the full version string.
    virtual const char* get_string() const = 0;

    /// Returns the neuray interface id.
    virtual base::Uuid get_neuray_iid() const = 0;
};

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_IVERSION_H
