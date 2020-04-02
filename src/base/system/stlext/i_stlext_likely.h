/***************************************************************************************************
 * Copyright (c) 2008-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief The definition of the Likely<> template class.

#ifndef BASE_SYSTEM_STLEXT_I_STLEXT_LIKELY_H
#define BASE_SYSTEM_STLEXT_I_STLEXT_LIKELY_H

#include <base/lib/log/i_log_assert.h>

namespace MI {
namespace STLEXT {

/// Return a result from a function which most likely will be a valid T. But sometimes, it
/// might be an invalid T - how can you find out such cases w/o having a proper failure value
/// for a T or w/o having an additional bool return value? The common example is atoi(), where
/// it is not possible to tell from the result whether the input was invalid or "0". A Likely<>
/// object can implicitly converted to a T or its value can be retrieved via its members.
/// \code
/// int i_val = 7789;
/// Likely<string> s_val = lexicographic_cast_s<string>(i_val);
/// if (s_val.get_status())
///     cout << *s_val.get_ptr() endl;
/// assert(*lexicographic_cast_s<int>(static_cast<string>(s_val)).get_value() == i_val);
/// \endcode
template <typename T>
class Likely
{
  public:
    /// Constructor.
    explicit Likely(const T& value=T(), bool status=true)
      : m_value(value), m_status(status)
    {}
    /// Is the current state valid?
    /// \return current status
    bool get_status() const     { return m_status; }
    /// Conversion operator. A real Likely<> implementation would disallow the retrieval
    /// of the value whenever the state is invalid. In systems without EH we could add
    /// some poor man's EH via error log messages - left out for simplicity here.
    /// \return const reference to the current value
    operator const T&() const   { ASSERT(SYSTEM::M_MAIN, m_status); return m_value; }
    /// Return the stored value as a pointer.
    /// \return the stored value if valid, 0 else
    const T* get_ptr() const  { return m_status? &m_value : 0; }
  private:
    /// the actual value
    T m_value;
    /// the current state's validity
    bool m_status;
};

}
}

#endif
