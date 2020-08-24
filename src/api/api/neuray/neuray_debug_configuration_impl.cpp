/***************************************************************************************************
 * Copyright (c) 2009-2020, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Implementation of IDebug_configuration
 **
 ** Implements the IDebug_configuration interface
 **/

#include "pch.h"

#include <base/lib/config/config.h>
#include <base/lib/log/i_log_module.h>

#include "neuray_debug_configuration_impl.h"
#include "neuray_string_impl.h"

namespace MI {

namespace NEURAY {

Debug_configuration_impl::Debug_configuration_impl()
{
    m_config_module.set();
}

Debug_configuration_impl::~Debug_configuration_impl()
{
    m_config_module.reset();
}

mi::Sint32 Debug_configuration_impl::set_option( const char* option)
{
    if( !option)
        return 0;

    return m_config_module->override( option) ? 0 : -1;
}

const mi::IString* Debug_configuration_impl::get_option( const char* key) const
{
    if( !key)
        return nullptr;
    std::string value = m_config_module->get_config_value_as_string( key);
    if( value.empty())
        return nullptr;
    mi::IString* istring = new String_impl();
    istring->set_c_str( value.c_str());
    return istring;
}

mi::Sint32 Debug_configuration_impl::start()
{
    // Since neuraylib does not parse the command line via m_config_module.read_commandline(),
    // we have to tell the CONFIG module manually that it is completely initialized now.
    m_config_module->set_initialization_complete( true);
    // The LOG module has already been started (before the CONFIG module was available).
    // The following call ensures that it obtains its config variables from the CONFIG module.
    m_log_module.set();
    m_log_module->configure();

    return 0;
}

mi::Sint32 Debug_configuration_impl::shutdown()
{
    m_log_module.reset();
    return 0;
}

} // namespace NEURAY

} // namespace MI
