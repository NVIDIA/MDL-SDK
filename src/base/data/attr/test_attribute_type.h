/***************************************************************************************************
 * Copyright (c) 2013-2023, NVIDIA CORPORATION. All rights reserved.
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
/// \brief The tests for the combination of \c ATTR::Attribute and \c ATTR::Type

#include "pch.h"

#include "attr.h"
#include "i_attr_utilities.h"

#include <base/system/main/access_module.h>
#include <base/system/test/i_test_auto_driver.h>
#include <base/lib/log/i_log_logger.h>

#include <string>
#include <utility>
#include <vector>

using namespace MI;
using namespace ATTR;

namespace {

/// Utility functions for TYPE_ENUM \c Attributes. Note that these versions don't work in general,
/// only for the case, where the top level \c Type is TYPE_ENUM, but not for nested TYPE_ENUM
/// \c Types.
//@{
/// Retrieve the index of the given enum's value in the enum collection.
/// \return index of the enum's value or -1 else
int retrieve_enum_value_index(
    const Attribute& attr);
/// Retrieve the string of the given enum's value of the \c Attribute \p attr.
/// \return string of the given enum's value or 0 else
const char* retrieve_enum_string_value(
    const Attribute& attr);
//}@

int retrieve_enum_value_index(
    const Attribute& attr)
{
    if (attr.get_type().get_typecode() != TYPE_ENUM)
        return -1;
    vector<pair<int, string> >* values = attr.get_type().get_enum();
    if (!values)
        return -1;

    // the given enum value
    int value = attr.get_value_int();

    vector<pair<int, string> >::const_iterator it, end=values->end();
    Sint32 index = 0;
    for (it=values->begin(); it != end; ++it, ++index) {
        if (it->first == value)
            return index;
    }
    return -1;
}


const char* retrieve_enum_string_value(
    const Attribute& attr)
{
    if (attr.get_type().get_typecode() != TYPE_ENUM)
        return 0;
    vector<pair<int, string> >* values = attr.get_type().get_enum();
    if (!values)
        return 0;

    // the given enum value
    int value = attr.get_value_int();

    vector<pair<int, string> >::const_iterator it, end=values->end();
    Sint32 index = 0;
    for (it=values->begin(); it != end; ++it, ++index) {
        if (it->first == value)
            return it->second.c_str();
    }
    return 0;
}

}


class Attribute_type_test_suite : public TEST::Test_suite
{
public:
    Attribute_type_test_suite() : TEST::Test_suite("ATTR::Attribute Type Test Suite")
    {
        m_attr_module.set();
        m_log_module.set();
        m_log_module->set_severity_limit(LOG::ILogger::S_ALL);
        // configure mod_log
        m_log_module->set_severity_by_category(LOG::ILogger::C_DATABASE, LOG::ILogger::S_ALL);

        add(MI_TEST_METHOD(Attribute_type_test_suite, test_enums));
    }

    ~Attribute_type_test_suite()
    {
        m_log_module.reset();
        m_attr_module.reset();
    }

    void test_enums()
    {
        {
        const char* name = "Equals_less_10";
        Type enum_type(ATTR::TYPE_ENUM, name);
        std::vector<std::pair<int, std::string> >**raw_ptr = enum_type.set_enum();
        *raw_ptr = new std::vector<std::pair<int, std::string> >;
        std::vector<std::pair<int, std::string> >* enum_collection = *raw_ptr;
        enum_collection->push_back(make_pair(0, "zero"));
        enum_collection->push_back(make_pair(2, "two"));
        enum_collection->push_back(make_pair(4, "four"));
        enum_collection->push_back(make_pair(6, "six"));
        enum_collection->push_back(make_pair(8, "eight"));

        Attribute_id id = Attribute::id_create(enum_type.get_name());
        Attribute attr(id, enum_type);

        // the actual attribute values are NOT the indices but the enum int values
        attr.set_value_int(2);
        MI_CHECK_EQUAL(retrieve_enum_value_index(attr), 1);
        const char* str = retrieve_enum_string_value(attr);
        MI_CHECK(str && string(str) == string("two"));

        attr.set_value_int(8);
        MI_CHECK_EQUAL(retrieve_enum_value_index(attr), 4);
        str = retrieve_enum_string_value(attr);
        MI_CHECK(str && string(str) == string("eight"));
        }
    }

    SYSTEM::Access_module<Attr_module> m_attr_module;
    SYSTEM::Access_module<Log_module> m_log_module;
};

MI_TEST_AUTO_CASE( new Attribute_type_test_suite );
