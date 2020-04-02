/***************************************************************************************************
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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
#include "version.h"
#include "util.h"
#include "errors.h"

using namespace i18n;
using std::string;
using std::ostream;

Version::Version(const string & version)
    : m_major(0), m_minor(0), m_patch(0)
{
    mi::Sint32 rtn = parse_version(version, m_major, m_minor, m_patch, m_label);
    if (0 != rtn)
    {
        Util::log_error("Error parsing version information: " + version);
    }
    check_success(0 == rtn);
}

bool Version::Test()
{
    Version v;
    {
        v = Version("1.0.0-alpha.1");
        v = Version("1");
        v = Version("1.");
        v = Version("1.1");
        v = Version("1.1.");
        v = Version("1.1.1");
        v = Version("1.1.1.");
        v = Version("1.1.1-label");
        v = Version("1.1.1-label garbage");
        v = Version("1.2.3");
    }
    Version v1("1");
    Version v2("2");
    check_success(v1 < v2);
    v1 = Version("1.0.1");
    v2 = Version("1.0.2");
    check_success(v1 < v2);
    v1 = Version("1.0.0");
    v2 = Version("1.1.0");
    check_success(v1 < v2);
    v1 = Version("1.1.0");
    v2 = Version("1.1.0");
    check_success(v1 <= v2);
    check_success(v1 >= v2);
    check_success(v1 == v2);
    v2 = Version("1.0.2-label");
    Util::log_info("Test log Version: " + i18n::to_string(v2));
    return true;
}

ostream& i18n::operator<<(ostream& os, const Version& dt)
{
    os << dt.major() << '.' << dt.minor() << '.' << dt.patch();
    if (!dt.label().empty())
    {
        os << '-' << dt.label();
    }
    return os;
}

bool i18n::operator< (const Version & lhs, const Version & rhs)
{
    for (int i = 0; i < 3; i++)
    {
        mi::Uint32 lhsv = lhs[i];
        mi::Uint32 rhsv = rhs[i];
        if (lhsv != rhsv)
        {
            return(lhsv < rhsv);
        }
    }
    return false;
}
bool i18n::operator> (const Version & lhs, const Version & rhs)
{
    return i18n::operator<(rhs, lhs);
}
bool i18n::operator<=(const Version & lhs, const Version & rhs)
{
    return !i18n::operator>(lhs, rhs);
}
bool i18n::operator>=(const Version & lhs, const Version & rhs)
{
    return !i18n::operator<(lhs, rhs);
}

bool i18n::operator==(const Version & lhs, const Version & rhs)
{
    if (lhs.major() != rhs.major()
        || lhs.minor() != rhs.minor()
        || lhs.patch() != rhs.patch()
        || lhs.label() != rhs.label()
        )
    {
        return false;
    }
    return true;
}
bool i18n::operator!=(const Version & lhs, const Version & rhs)
{
    return !i18n::operator==(lhs, rhs);
}
