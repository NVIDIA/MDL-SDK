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

/** \file
 ** \brief Implementation of IVersion
 **
 ** Implements the IVersion interface
 **/

#ifndef API_API_NEURAY_VERSION_IMPL_H
#define API_API_NEURAY_VERSION_IMPL_H

#include <mi/neuraylib/iversion.h>
#include <mi/base/interface_implement.h>

#include <string>

namespace MI {

namespace NEURAY {

class Version_impl
  : public mi::base::Interface_implement<mi::neuraylib::IVersion>
{
public:
    /// Constructor of Version_impl
    ///
    Version_impl();

    // public API methods

    const char* get_product_name() const;

    const char* get_product_version() const;

    const char* get_build_number() const;

    const char* get_build_date() const;

    const char* get_build_platform() const;

    const char* get_string() const;

    mi::base::Uuid get_neuray_iid() const;

private:

    std::string m_version;
    std::string m_stripped_platform;
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_VERSION_IMPL_H
