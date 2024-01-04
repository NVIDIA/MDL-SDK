/******************************************************************************
 * Copyright (c) 2010-2023, NVIDIA CORPORATION. All rights reserved.
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

/// \file
/// \brief regression test public/mi/base components
///
/// See \ref mi_base_uuid.
///

#include "pch.h"
#include <base/system/test/i_test_auto_case.h>

#include <mi/base/uuid.h>

mi_static_assert(sizeof(mi::base::Uuid) == 16);

namespace mi { namespace base {
std::ostringstream& operator<<( std::ostringstream& out, const mi::base::Uuid& id) {
    out << "Uuid(" << id.m_id1 << ','
                   << id.m_id2 << ','
                   << id.m_id3 << ','
                   << id.m_id4 << ')';
    return out;
}
}}

typedef mi::base::Uuid_t<0x33244daf,0xd131,0x4f29,0x98,0xcf,0x83,0xe1,0xdc,0x7c,0xdf,0xe0> ID1;
typedef mi::base::Uuid_t<0x841522b0,0xdf6f,0x4d84,0x97,0xa2,0x0b,0xdd,0x60,0x97,0x36,0x9b> ID2;

mi::Uint32 uuid_switch_test( const mi::base::Uuid& id) {
    switch ( uuid_hash32( id)) {
    case ID1::hash32:
        return 1;
    case ID2::hash32:
        return 2;
    default:
        ;
    }
    return 0;
}

MI_TEST_AUTO_FUNCTION( test_uuid )
{
    mi::base::Uuid id1 = ID1();
    mi::base::Uuid id2 = ID2();

    MI_CHECK_EQUAL( id1, id1);
    MI_CHECK(  !(id1 == id2));
    MI_CHECK_NOT_EQUAL( id1, id2);
    MI_CHECK( ! (id1 != id1));
    MI_CHECK_LESS( id1, id2);
    MI_CHECK( ! (id2 < id1));
    MI_CHECK( ! (id1 < id1));
    MI_CHECK_GREATER( id2, id1);
    MI_CHECK( ! (id1 > id2));
    MI_CHECK( ! (id1 > id1));
    MI_CHECK_LESS_OR_EQUAL( id1, id2);
    MI_CHECK_LESS_OR_EQUAL( id1, id1);
    MI_CHECK( ! (id2 <= id1));
    MI_CHECK_GREATER_OR_EQUAL( id2, id1);
    MI_CHECK_GREATER_OR_EQUAL( id1, id1);
    MI_CHECK( ! (id1 >= id2));

    MI_CHECK_EQUAL( 1, uuid_switch_test(id1));
    MI_CHECK_EQUAL( 2, uuid_switch_test(id2));
}

