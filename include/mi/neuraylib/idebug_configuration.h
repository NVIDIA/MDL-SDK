/***************************************************************************************************
 * Copyright (c) 2008-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief API component for debugging settings.

#ifndef MI_NEURAYLIB_IDEBUG_CONFIGURATION_H
#define MI_NEURAYLIB_IDEBUG_CONFIGURATION_H

#include <mi/base/interface_declare.h>

namespace mi {

class IString;

namespace neuraylib {

/** \addtogroup mi_neuray_configuration
@{
*/

/// This interface represents an interface to set debug options.
class IDebug_configuration : public 
    mi::base::Interface_declare<0x7938887b,0x57c6,0x422f,0x84,0x03,0xdc,0x06,0xf2,0x26,0xd6,0x04>
{
public:
    /// Sets a particular debug option.
    ///
    /// \param option    The option to be set in the form \c key=value.
    /// \return
    ///                  -  0: Success.
    ///                  - -1: The option could not be successfully parsed. This happens for example
    ///                        if the option is not of the form key=value.
    virtual Sint32 set_option( const char* option) = 0;

    /// Returns the value of a particular debug option.
    ///
    /// \param key       The key of the debug option.
    /// \return          The value of the debug option, or \c NULL if the option is not set.
    virtual const IString* get_option( const char* key) const = 0;
};

/*@}*/ // end group mi_neuray_configuration

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_IDEBUG_CONFIGURATION_H
