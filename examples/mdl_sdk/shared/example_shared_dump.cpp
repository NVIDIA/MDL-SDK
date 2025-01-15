/******************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION. All rights reserved.
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
 *****************************************************************************/

#include <iomanip>

#include "example_shared_dump.h"

namespace mi { namespace examples { namespace mdl {

// Utility function to dump the structure of a compiled material.
void dump_compiled_material(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_factory* mdl_factory,
    const mi::neuraylib::ICompiled_material* cm,
    std::ostream& s)
{
    mi::base::Handle<mi::neuraylib::IValue_factory> value_factory(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> expression_factory(
        mdl_factory->create_expression_factory( transaction));

    mi::base::Uuid hash = cm->get_hash();
    char buffer[36];
    snprintf( buffer, sizeof( buffer),
        "%08x %08x %08x %08x", hash.m_id1, hash.m_id2, hash.m_id3, hash.m_id4);
    s << "    hash overall = " << buffer << std::endl;

    for( mi::Uint32 i = mi::neuraylib::SLOT_FIRST; i <= mi::neuraylib::SLOT_LAST; ++i) {
        hash = cm->get_slot_hash( mi::neuraylib::Material_slot( i));
        snprintf( buffer, sizeof( buffer),
            "%08x %08x %08x %08x", hash.m_id1, hash.m_id2, hash.m_id3, hash.m_id4);
        s << "    hash slot " << std::setw( 2) << i << " = " << buffer << std::endl;
    }

    mi::Size parameter_count = cm->get_parameter_count();
    for( mi::Size i = 0; i < parameter_count; ++i) {
        mi::base::Handle<const mi::neuraylib::IValue> argument( cm->get_argument( i));
        std::stringstream name;
        name << i;
        mi::base::Handle<const mi::IString> result(
            value_factory->dump( argument.get(), name.str().c_str(), 1));
        s << "    argument " << result->get_c_str() << std::endl;
    }

    mi::Size temporary_count = cm->get_temporary_count();
    for( mi::Size i = 0; i < temporary_count; ++i) {
        mi::base::Handle<const mi::neuraylib::IExpression> temporary( cm->get_temporary( i));
        std::stringstream name;
        name << i;
        mi::base::Handle<const mi::IString> result(
            expression_factory->dump( temporary.get(), name.str().c_str(), 1));
        s << "    temporary " << result->get_c_str() << std::endl;
    }

    mi::base::Handle<const mi::neuraylib::IExpression> body( cm->get_body());
    mi::base::Handle<const mi::IString> result( expression_factory->dump( body.get(), 0, 1));
    s << "    body " << result->get_c_str() << std::endl;

    s << std::endl;
}

}}}
