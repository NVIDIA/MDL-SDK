/***************************************************************************************************
 * Copyright (c) 2010-2025, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Implementation of IFactory
 **
 ** Implements the IFactory interface
 **/

#include "pch.h"

#include "neuray_class_factory.h"
#include "neuray_factory_impl.h"
#include "neuray_type_utilities.h"

#include <mi/base/handle.h>
#include <mi/neuraylib/iarray.h>
#include <mi/neuraylib/icompound.h>
#include <mi/neuraylib/idata.h>
#include <mi/neuraylib/idynamic_array.h>
#include <mi/neuraylib/ienum.h>
#include <mi/neuraylib/imap.h>
#include <mi/neuraylib/imatrix.h>
#include <mi/neuraylib/iref.h>
#include <mi/neuraylib/istring.h>
#include <mi/neuraylib/inumber.h>
#include <mi/neuraylib/ipointer.h>
#include <mi/neuraylib/istructure.h>
#include <mi/neuraylib/itransaction.h>
#include <mi/neuraylib/iuuid.h>
#include <mi/neuraylib/ivector.h>

#include <mi/neuraylib/ibbox.h>
#include <mi/neuraylib/icolor.h>
#include <mi/neuraylib/ispectrum.h>
#include "neuray_expression_impl.h"
#include "neuray_type_impl.h"
#include "neuray_value_impl.h"

#include <sstream>

#include <base/util/string_utils/i_string_lexicographic_cast.h>
#include <base/data/idata/i_idata_factory.h>

namespace MI {

namespace NEURAY {

mi::neuraylib::IFactory* s_factory = nullptr;

Factory_impl::Factory_impl( Class_factory* class_factory)
  : m_class_factory( class_factory)
{
}

Factory_impl::~Factory_impl()
{
    m_class_factory = nullptr;
}

mi::base::IInterface* Factory_impl::create(
    const char* type_name,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    return m_class_factory->create_type_instance( nullptr, type_name, argc, argv);
}

mi::Uint32 Factory_impl::assign_from_to(
    const mi::IData* source, mi::IData* target, mi::Uint32 options)
{
    IDATA::Factory* idata_factory = m_class_factory->get_idata_factory();
    return idata_factory->assign_from_to( source, target, options);
}

mi::IData* Factory_impl::clone( const mi::IData* source, mi::Uint32 options)
{
    IDATA::Factory* idata_factory = m_class_factory->get_idata_factory();
    return idata_factory->clone( source, options);
}

mi::Sint32 Factory_impl::compare( const mi::IData* lhs, const mi::IData* rhs)
{
    IDATA::Factory* idata_factory = m_class_factory->get_idata_factory();
    return idata_factory->compare( lhs, rhs);
}

const mi::IString* Factory_impl::dump( const mi::IData* data, const char* name, mi::Size depth)
{
    if( !data)
        return nullptr;

    std::ostringstream s;
    IDATA::Factory* idata_factory = m_class_factory->get_idata_factory();
    idata_factory->dump( name, data, depth, s);

    auto* result = create<mi::IString>( "String");
    result->set_c_str( s.str().c_str());
    return result;
}

const mi::IString* Factory_impl::dump(
    mi::neuraylib::ITransaction* transaction,
    const mi::IData* data,
    const char* name,
    mi::Size depth)
{
    return dump( data, name, depth);
}

const mi::IStructure_decl* Factory_impl::get_structure_decl( const char* structure_name) const
{
   if( !structure_name)
        return nullptr;

    IDATA::Factory* idata_factory = m_class_factory->get_idata_factory();
    return idata_factory->get_structure_decl( structure_name);
}

const mi::IEnum_decl* Factory_impl::get_enum_decl( const char* enum_name) const
{
   if( !enum_name)
        return nullptr;

    IDATA::Factory* idata_factory = m_class_factory->get_idata_factory();
    return idata_factory->get_enum_decl( enum_name);
}

mi::Sint32 Factory_impl::start()
{
    return 0;
}

mi::Sint32 Factory_impl::shutdown()
{
    return 0;
}

} // namespace NEURAY

} // namespace MI

