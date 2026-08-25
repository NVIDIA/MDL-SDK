/***************************************************************************************************
 * Copyright (c) 2010-2026, NVIDIA CORPORATION. All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
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
 ** \brief Source for the IExtension_api implementation.
 **/

#include "pch.h"

#include "neuray_class_factory.h"
#include "neuray_class_registration.h"
#include "neuray_extension_api_impl.h"


#include <mi/neuraylib/ineuray.h>


namespace MI {

namespace NEURAY {

static const char* valid_decl_characters
    = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_";

Extension_api_impl::Extension_api_impl(
    mi::neuraylib::INeuray* neuray, Class_factory* class_factory)
  : m_neuray( neuray),
    m_class_factory( class_factory)
{
}

Extension_api_impl::~Extension_api_impl()
{
    if( m_class_factory) {
        m_class_factory->unregister_structure_decls();
        m_class_factory->unregister_enum_decls();
    }

    m_neuray = nullptr;
    m_class_factory = nullptr;
}

mi::Sint32 Extension_api_impl::start()
{
    return 0;
}

mi::Sint32 Extension_api_impl::shutdown()
{
    return 0;
}

mi::Sint32 Extension_api_impl::register_structure_decl(
    const char* structure_name, const mi::IStructure_decl* decl)
{
    if( !structure_name || !decl)
        return -2;

    const std::string structure_name_str( structure_name);
    if( structure_name_str.empty() || (structure_name_str[0] == '_'))
        return -4;
    if( structure_name_str.find_first_not_of( valid_decl_characters) != std::string::npos)
        return -4;

    return m_class_factory->register_structure_decl( structure_name, decl);
}

mi::Sint32 Extension_api_impl::unregister_structure_decl( const char* structure_name)
{
    if( !structure_name)
        return -2;

    const std::string structure_name_str( structure_name);
    if( structure_name_str.empty() || (structure_name_str[0] == '_'))
        return -4;
    if( structure_name_str.find_first_not_of( valid_decl_characters) != std::string::npos)
        return -4;
    if( Class_registration::is_predefined_structure_declaration( structure_name))
        return -6;

    return m_class_factory->unregister_structure_decl( structure_name);
}

mi::Sint32 Extension_api_impl::register_enum_decl(
    const char* enum_name, const mi::IEnum_decl* decl)
{
    if( !enum_name || !decl)
        return -2;

    const std::string enum_name_str( enum_name);
    if( enum_name_str.empty() || (enum_name_str[0] == '_'))
        return -4;
    if( enum_name_str.find_first_not_of( valid_decl_characters) != std::string::npos)
        return -4;

    return m_class_factory->register_enum_decl( enum_name, decl);
}

mi::Sint32 Extension_api_impl::unregister_enum_decl( const char* enum_name)
{
    if( !enum_name)
        return -2;

    const std::string enum_name_str( enum_name);
    if( enum_name_str.empty() || (enum_name_str[0] == '_'))
        return -4;
    if( enum_name_str.find_first_not_of( valid_decl_characters) != std::string::npos)
        return -4;

    return m_class_factory->unregister_enum_decl( enum_name);
}

mi::Sint32 Extension_api_impl::register_class(
    const char* class_name, mi::base::Uuid uuid, mi::neuraylib::IUser_class_factory* factory)
{
    (void) class_name;
    (void) uuid;
    (void) factory;
    return -1;
}

mi::Sint32 Extension_api_impl::unregister_class(
    const char* class_name, mi::base::Uuid uuid)
{
    (void) class_name;
    (void) uuid;
    return -1;
}

mi::Sint32 Extension_api_impl::register_importer( mi::neuraylib::IImporter* importer)
{
    return -1;
}

mi::Sint32 Extension_api_impl::register_exporter( mi::neuraylib::IExporter* exporter)
{
    return -1;
}

mi::Sint32 Extension_api_impl::unregister_importer( mi::neuraylib::IImporter* importer)
{
    return -1;
}

mi::Sint32 Extension_api_impl::unregister_exporter( mi::neuraylib::IExporter* exporter)
{
    return -1;
}

} // namespace NEURAY

} // namespace MI

