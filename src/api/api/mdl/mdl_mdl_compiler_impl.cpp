/***************************************************************************************************
 * Copyright (c) 2012-2025, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Implementation of IMdl_compiler
 **
 ** Implements the IMdl_compiler interface
 **/

#include "pch.h"

#include "mdl_mdl_compiler_impl.h"
#include "mdl_neuray_impl.h"

#include "neuray_impexp_utilities.h"

#include "neuray_module_impl.h"
#include "neuray_transaction_impl.h"
#include "neuray_mdl_execution_context_impl.h"

#include <mi/neuraylib/imdl_backend_api.h>
#include <mi/neuraylib/imdl_impexp_api.h>

#include <mi/mdl/mdl_modules.h>
#include <mi/mdl/mdl_mdl.h>
#include <mi/mdl/mdl_entity_resolver.h>

#include <base/hal/disk/disk_memory_reader_writer_impl.h>
#include <base/lib/log/i_log_logger.h>
#include <base/lib/path/i_path.h>
#include <base/lib/plug/i_plug.h>
#include <mdl/integration/mdlnr/i_mdlnr.h>
#include <io/scene/mdl_elements/i_mdl_elements_module.h>


namespace MI {

namespace MDL {

Mdl_compiler_impl::Mdl_compiler_impl( Neuray_impl* neuray_impl)
  : m_neuray_impl( neuray_impl),
    m_attr_module( true),
    m_mdlc_module( true),
    m_mem_module( false),
    m_path_module( false),
    m_plug_module( false)
{
}

Mdl_compiler_impl::~Mdl_compiler_impl()
{
    m_neuray_impl = nullptr;
}

mi::Sint32 Mdl_compiler_impl::add_builtin_module(
    const char* module_name, const char* module_source)
{
    if (!module_name || !module_source)
        return -1;

    mi::base::Handle<mi::mdl::IMDL> compiler(m_mdlc_module->get_mdl());
    bool success = compiler->add_builtin_module(
        module_name,
        module_source,
        strlen(module_source),
        /*is_encoded*/ false,
        /*is_native*/ true);
    return success ? 0 : -1;
}

mi::Sint32 Mdl_compiler_impl::start()
{
    m_mdlc_module.set();
    m_attr_module.set();

    return 0;
}

mi::Sint32 Mdl_compiler_impl::shutdown()
{
    m_attr_module.reset();
    m_mdlc_module.reset();

    return 0;
}

} // namespace MDL



} // namespace MI


