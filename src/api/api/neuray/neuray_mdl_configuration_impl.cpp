/***************************************************************************************************
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Implementation of IMdl_configuration
 **
 ** Implements the IMdl_configuration interface
 **/

#include "pch.h"

#include "neuray_mdl_configuration_impl.h"

#include <mdl/integration/mdlnr/i_mdlnr.h>


namespace MI {

namespace NEURAY {

Mdl_configuration_impl::Mdl_configuration_impl( mi::neuraylib::INeuray* neuray)
  : m_neuray( neuray)
  , m_mdlc_module( false)
{
}

Mdl_configuration_impl::~Mdl_configuration_impl()
{
}

mi::Sint32 Mdl_configuration_impl::set_implicit_cast_enabled( bool value)
{
    const mi::neuraylib::INeuray::Status status = m_neuray->get_status();
    if(    (status != mi::neuraylib::INeuray::PRE_STARTING)
        && (status != mi::neuraylib::INeuray::SHUTDOWN))
        return -1;

    m_mdlc_module->set_implicit_cast_enabled( value);
    return 0;
}

bool Mdl_configuration_impl::get_implicit_cast_enabled() const
{
    return m_mdlc_module->get_implicit_cast_enabled();
}

mi::Sint32 Mdl_configuration_impl::set_expose_names_of_let_expressions( bool value)
{
   const mi::neuraylib::INeuray::Status status = m_neuray->get_status();
    if(    (status != mi::neuraylib::INeuray::PRE_STARTING)
        && (status != mi::neuraylib::INeuray::SHUTDOWN))
        return -1;

    m_mdlc_module->set_expose_names_of_let_expressions(value);
    return 0;
}

bool Mdl_configuration_impl::get_expose_names_of_let_expressions() const
{
    return m_mdlc_module->get_expose_names_of_let_expressions();
}

mi::Sint32 Mdl_configuration_impl::start()
{
    return 0;
}

mi::Sint32 Mdl_configuration_impl::shutdown()
{
    m_mdlc_module.reset();
    return 0;
}

} // namespace NEURAY

} // namespace MI
