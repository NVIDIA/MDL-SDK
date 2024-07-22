/***************************************************************************************************
 * Copyright (c) 2010-2024, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Header for the IString implementation.
 **/

#ifndef API_API_NEURAY_NEURAY_STRING_IMPL_H
#define API_API_NEURAY_NEURAY_STRING_IMPL_H

#include <mi/base/interface_implement.h>
#include <mi/neuraylib/istring.h>

#include <string>

#include <boost/core/noncopyable.hpp>

namespace MI {

namespace NEURAY {

/// Convenience implementation of IString.
///
/// Exists for code that wants to do "return new String_impl(...)" without invoking the class
/// factory.
class String_impl : public mi::base::Interface_implement<mi::IString>, public boost::noncopyable
{
public:
    String_impl( const char* str = nullptr) {  set_c_str( str); }
    const char* get_type_name() const { return "String"; }
    const char* get_c_str() const { return m_storage.c_str(); }
    void set_c_str( const char* str) { m_storage = str ? str : ""; }

private:
    std::string m_storage;
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_NEURAY_STRING_IMPL_H
