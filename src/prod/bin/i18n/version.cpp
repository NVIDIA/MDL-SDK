/******************************************************************************
* Copyright 2018 NVIDIA Corporation. All rights reserved.
*****************************************************************************/
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
