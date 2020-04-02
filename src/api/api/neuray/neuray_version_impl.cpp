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

#include "pch.h"

#include "neuray_version_impl.h"

#include <base/system/version/i_version.h>
#include <mi/neuraylib/version.h>
#include <mi/neuraylib/ineuray.h>
#include <mi/base/config.h>


namespace {

std::string strip_compiler_suffix(const std::string& platform)
{
#if defined(MI_PLATFORM_WINDOWS)
    const std::string suffix = "-vc";
#elif defined(MI_PLATFORM_MACOSX)
    const std::string suffix = "-clang";
#elif defined(MI_PLATFORM_LINUX)
    const std::string suffix = "-gcc";
#else
#warning Unknown platform
    const std::string suffix = "";
    return platform;
#endif
    size_t p = platform.find(suffix);
    if(p == std::string::npos)
        return platform;
    return platform.substr(0, p);
}
}

namespace MI {

namespace NEURAY {

Version_impl::Version_impl()
{
    m_stripped_platform = strip_compiler_suffix(VERSION::get_platform_os());

    m_version = get_product_name() + std::string(" ") + get_product_version();
    m_version += std::string( ", build ") + VERSION::get_platform_version();
    m_version += std::string( ", ") + VERSION::get_platform_date();
    m_version += std::string( ", ") + m_stripped_platform;
}

const char* Version_impl::get_product_name() const
{
    return "MDL SDK"; 
}

const char* Version_impl::get_product_version() const
{
    return MI_NEURAYLIB_PRODUCT_VERSION_STRING;
}

const char* Version_impl::get_build_number() const
{
    return VERSION::get_platform_version();
}

const char* Version_impl::get_build_date() const
{
    return VERSION::get_platform_date();
}

const char* Version_impl::get_build_platform() const
{
    return m_stripped_platform.c_str();
}

const char* Version_impl::get_string() const
{
    return m_version.c_str();
}

mi::base::Uuid Version_impl::get_neuray_iid() const
{
    return mi::neuraylib::INeuray::IID();
}

} // namespace NEURAY

} // namespace MI

