/***************************************************************************************************
 * Copyright (c) 2017-2025, NVIDIA CORPORATION. All rights reserved.
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
/// \brief Tests for markers and marker helpers.

#include "serial_marker_helpers.h"
#include "serial.h"
#include "i_serializer.h"
#include "i_serial_buffer_serializer.h"
#include "test_function.h"
#include "test_serializable.h"

#include <base/lib/zlib/i_zlib.h>
#include <base/system/test/i_test_auto_case.h>

using namespace MI;
using namespace MI::SERIAL;

namespace std {
template <typename... Tp>
std::ostream& operator<<(std::ostream& str, const std::variant<Tp...>& var)
{
#ifdef MI_PLATFORM_MACOSX
    return str << "variant(" << var.index() << ')';
#else
    std::visit([&str](const auto& el){ str << el; },var);
    return str;
#endif
}
}

/// Serialize object with extension. Expect correct deserialization.
MI_TEST_AUTO_FUNCTION( test_extension_marker )
{
    Deserialization_manager_holder dmh;
    dmh.get()->register_class(
        ID_TEST_TYPE_2_CLASS_ID, test_type_2_factory);

    Buffer_serializer serializer;

    Test_type_2 src;
    src.m_int = 11;
    src.m_ext = 12;

    serializer.serialize(&src, false);

    Buffer_deserializer deserializer(dmh.get());
    deserializer.reset(serializer.get_buffer(), serializer.get_buffer_size());
    Test_type_2 *dst = dynamic_cast<Test_type_2*>(deserializer.deserialize(false));

    MI_REQUIRE_EQUAL(src, *dst);
}

/// Serialize object without extension. Deserialize without it.
MI_TEST_AUTO_FUNCTION( test_extension_marker_no_serialized_extension )
{
    Deserialization_manager_holder dmh;
    dmh.get()->register_class(
        ID_TEST_TYPE_2_CLASS_ID, test_type_2_factory);

    Buffer_serializer serializer;

    Test_type_2 src(false);
    src.m_int = 11;

    serializer.serialize(&src, false);

    Buffer_deserializer deserializer(dmh.get());
    deserializer.reset(serializer.get_buffer(), serializer.get_buffer_size());
    Test_type_2 *dst = dynamic_cast<Test_type_2*>(deserializer.deserialize(false));

    MI_REQUIRE_EQUAL(src, *dst);
}

/// Serialize object with extension without deserializer knowing about it.
/// Expect partial deserialization.
MI_TEST_AUTO_FUNCTION( test_extension_marker_no_deserialized_extension )
{
    Deserialization_manager_holder dmh;
    dmh.get()->register_class(
        ID_TEST_TYPE_2_CLASS_ID, test_type_2_no_ext_factory);

    Buffer_serializer serializer;

    Test_type_2 src(false);
    src.m_int = 22;
    serializer.serialize(&src, false);

    Buffer_deserializer deserializer(dmh.get());
    deserializer.reset(serializer.get_buffer(), serializer.get_buffer_size());
    Test_type_2 *dst = dynamic_cast<Test_type_2*>(deserializer.deserialize(false));

    MI_REQUIRE_EQUAL(src, *dst);
    MI_REQUIRE(!dst->m_ext_found);
}

/// Serialize object with end marker inside payload. Expect correct deserialization.
MI_TEST_AUTO_FUNCTION( test_extension_marker_with_end_marker_in_payload )
{
    Deserialization_manager_holder dmh;
    dmh.get()->register_class(
        ID_TEST_TYPE_2_CLASS_ID, test_type_2_factory);

    Buffer_serializer serializer;

    Test_type_2 src;
    src.m_int = END_MARKER;
    src.m_ext = 22;
    serializer.serialize(&src, false);

    Buffer_deserializer deserializer(dmh.get());
    deserializer.reset(serializer.get_buffer(), serializer.get_buffer_size());
    Test_type_2 *dst = dynamic_cast<Test_type_2*>(deserializer.deserialize(false));

    MI_REQUIRE_EQUAL(src, *dst);
}

/// Serialize object with extension marker in payload
MI_TEST_AUTO_FUNCTION( test_extension_marker_with_ext_marker_in_payload )
{
    Deserialization_manager_holder dmh;
    dmh.get()->register_class(
        ID_TEST_TYPE_2_CLASS_ID, test_type_2_factory);

    Buffer_serializer serializer;

    Test_type_2 src;
    src.m_int = EXTENSION_MARKER;
    src.m_ext = 22;
    serializer.serialize(&src, false);

    Buffer_deserializer deserializer(dmh.get());
    deserializer.reset(serializer.get_buffer(), serializer.get_buffer_size());
    Test_type_2 *dst = dynamic_cast<Test_type_2*>(deserializer.deserialize(false));

    MI_REQUIRE_EQUAL(src.m_int, dst->m_int);
    MI_REQUIRE_EQUAL(src.m_ext, dst->m_ext);
    MI_REQUIRE_EQUAL(src, *dst);
}

/// Serialize multiple objects with various options.
MI_TEST_AUTO_FUNCTION( test_multiple_mixed_objects )
{
    Deserialization_manager_holder dmh;
    dmh.get()->register_class(
        ID_TEST_TYPE_0_CLASS_ID, test_type_0_factory);
    dmh.get()->register_class(
        ID_TEST_TYPE_2_CLASS_ID, test_type_2_factory);

    Buffer_serializer serializer;

    std::vector<Serializable*> src_objects;
    src_objects.push_back(new Test_type_0());
    src_objects.push_back(new Test_type_2(true));
    src_objects.push_back(new Test_type_0());
    src_objects.push_back(new Test_type_2());
    src_objects.push_back(new Test_type_2(true));
    src_objects.push_back(new Test_type_2());

    for (int i = 0; i < src_objects.size(); i++)
        serializer.serialize(src_objects[i], false);

    Buffer_deserializer deserializer(dmh.get());
    deserializer.reset(serializer.get_buffer(), serializer.get_buffer_size());

    for (int i = 0; i < src_objects.size(); i++)
    {
        Serializable *dst = deserializer.deserialize(false);
        if (dynamic_cast<Test_type_0*>(dst))
            MI_REQUIRE_EQUAL(*dynamic_cast<Test_type_0*>(src_objects[i]), *dynamic_cast<Test_type_0*>(dst));
        else if (dynamic_cast<Test_type_2*>(dst))
            MI_REQUIRE_EQUAL(*dynamic_cast<Test_type_2*>(src_objects[i]), *dynamic_cast<Test_type_2*>(dst));
        else
            MI_REQUIRE_MSG(false, "Unknown type.");
    }

    for (int i = 0; i < src_objects.size(); i++)
        delete src_objects[i];
    src_objects.clear();
}


/// Serialize std::variant
MI_TEST_AUTO_FUNCTION( test_variant )
{
    Deserialization_manager_holder dmh;
    dmh.get()->register_class(
        ID_TEST_TYPE_0_CLASS_ID, test_type_0_factory);
    dmh.get()->register_class(
        ID_TEST_TYPE_2_CLASS_ID, test_type_2_factory);

    Buffer_serializer serializer;

    using Variant = std::variant<int,double,std::string>;
    Variant v1{1};
    Variant v2{"foo"};
    SERIAL::write(&serializer,v1);
    SERIAL::write(&serializer,v2);

    Buffer_deserializer deserializer(dmh.get());
    deserializer.reset(serializer.get_buffer(), serializer.get_buffer_size());

    Variant tmp;
    SERIAL::read(&deserializer,&tmp);
    MI_REQUIRE_EQUAL(v1,tmp);
    SERIAL::read(&deserializer,&tmp);
    MI_REQUIRE_EQUAL(v2,tmp);
}

/// Serialize serializable object with a nested serializable object.
MI_TEST_AUTO_FUNCTION( test_extension_marker_with_nested_object )
{
    Deserialization_manager_holder dmh;
    dmh.get()->register_class(
        ID_TEST_TYPE_2_CLASS_ID, test_type_2_factory);
    dmh.get()->register_class(
        ID_TEST_TYPE_3_CLASS_ID, test_type_3_factory);

    Buffer_serializer serializer;

    Test_type_2 s2;
    s2.m_int = 11;
    s2.m_ext = 22;

    Test_type_3 s3;
    s3.m_data = 33;
    s3.m_ptr = &s2;

    serializer.serialize(&s3, false);

    mi::Uint8* buf= serializer.get_buffer();
    size_t buf_sz = serializer.get_buffer_size();

    // Layout should be:
    //
    //    0-11: id of TT3 (8 bytes id, 4 bytes class id)
    //      12: data of TT3
    //   16-17: id of TT2 (8 bytes id, 4 bytes class id)
    //      28: data of TT2
    // ...
    // sz - 16: END_MARKER
    // sz - 12: CRC_TT2
    // sz -  8: END_MARKER
    // sz -  4: CRC_TT3
    // sz     : end of buffer

    // Test for checksum of the outer object (Test_type_3)
    mi::Uint32 outer_crc_serialized = *(reinterpret_cast<mi::Uint32*>(&buf[buf_sz - 4]));
    mi::Uint32 outer_crc_computed = ZLIB::crc32(&buf[12], buf_sz - 12 - 4);
    MI_REQUIRE_EQUAL(outer_crc_serialized, outer_crc_computed);

    // Test for checksum of the outer object (Test_type_2)
    mi::Uint32 inner_crc_serialized = *(reinterpret_cast<mi::Uint32*>(&buf[buf_sz - 3*4]));
    mi::Uint32 inner_crc_computed = ZLIB::crc32(&buf[28], buf_sz - 28 - 3*4);
    MI_REQUIRE_EQUAL(inner_crc_serialized, inner_crc_computed);

    // And check deserialization, just in case.
    Buffer_deserializer deserializer(dmh.get());
    deserializer.reset(buf, buf_sz);
    Test_type_3 *dst = dynamic_cast<Test_type_3*>(deserializer.deserialize(false));
    MI_REQUIRE_EQUAL(s3, *dst);
}

/// Serialize serializable object with a nested serializable object. This time use
/// an object that does recursive serialization/deserialization.
MI_TEST_AUTO_FUNCTION( test_extension_marker_with_recursive_nesting )
{
    Deserialization_manager_holder dmh;
    dmh.get()->register_class(
        ID_TEST_TYPE_4_CLASS_ID, test_type_4_factory);

    Test_type_4 src(20);

    // Serialize
    Buffer_serializer serializer;
    serializer.serialize(&src);

    // Deserialize
    Buffer_deserializer deserializer(dmh.get());
    deserializer.reset(serializer.get_buffer(), serializer.get_buffer_size());
    deserializer.deserialize(false);
}

MI_TEST_AUTO_FUNCTION( test_serializer_checksummer_simple )
{
    Serializer_checksummer cs;
    const char *buf = "Foo Bar";

    cs.start();
    cs.update(buf, sizeof(buf));
    MI_REQUIRE_EQUAL(cs.get(), ZLIB::crc32(buf, sizeof(buf)));
    cs.end();

    MI_REQUIRE_EQUAL(cs.get(), 0);
}

MI_TEST_AUTO_FUNCTION( test_serializer_checksummer_nested )
{
    Serializer_checksummer cs;
    const char *buf = "Foo Bar";
    int pos = 3;

    cs.start();
    // update for first 3 bytes and check
    cs.update(buf, pos);
    MI_REQUIRE_EQUAL(cs.get(), ZLIB::crc32(buf, 3));

        cs.start();
        // start a scope for the remaining bytes
        size_t len = sizeof(buf) - pos;
        cs.update(&buf[pos], len);
        MI_REQUIRE_EQUAL(cs.get(), ZLIB::crc32(&buf[pos], len));
        cs.end();

    // And the outer crc should have been updated.
    MI_REQUIRE_EQUAL(cs.get(), ZLIB::crc32(buf, sizeof(buf)));
    cs.end();

    MI_REQUIRE_EQUAL(cs.get(), 0);
}

class Deserializer_test_error_handler:
        public IDeserializer_error_handler<Serializable>
{
public:
    void handle(Marker_status status, const Serializable* serializable)
    {

        m_last_status = status;
        m_last_class_id = serializable->get_class_id();
    }

    Marker_status m_last_status;
    Class_id m_last_class_id;
};

/// Serialize object, patch a byte. Install error handler in deserializer, expect
/// it to be called.
MI_TEST_AUTO_FUNCTION( test_deserializer_bad_checksum )
{
    mi::base::Handle<Deserializer_test_error_handler> err_handler(
                new Deserializer_test_error_handler());

    Deserialization_manager_holder dmh;
    dmh.get()->register_class(
        ID_TEST_TYPE_2_CLASS_ID, test_type_2_factory);

    Buffer_serializer serializer;

    Test_type_2 src;
    src.m_int = 11;
    src.m_ext = 12;

    serializer.serialize(&src, false);
    // Corrupt byte in data to insert error
    serializer.get_buffer()[15] = ~serializer.get_buffer()[15];

    Buffer_deserializer deserializer(dmh.get());
    deserializer.set_error_handler(err_handler.get());

    deserializer.reset(serializer.get_buffer(), serializer.get_buffer_size());
    deserializer.deserialize(false);

    MI_REQUIRE_EQUAL(err_handler->m_last_status, MARKER_BAD_CHECKSUM);
    MI_REQUIRE_EQUAL(err_handler->m_last_class_id, ID_TEST_TYPE_2_CLASS_ID);
}
