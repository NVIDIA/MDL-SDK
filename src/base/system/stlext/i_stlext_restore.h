/******************************************************************************
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
 *****************************************************************************/
/// \file
/// \brief Utility class to backup and restore temporarily overwritten values.

#ifndef BASE_SYSTEM_STLEXT_RESTORE_H
#define BASE_SYSTEM_STLEXT_RESTORE_H

#include <base/system/main/platform.h>
#include "i_stlext_concepts.h"

namespace MI {
namespace STLEXT {


/** \brief Stores and restores a value.

 This class stores a value on construction and restores it when the object
 goes out of scope. This allows saving and restoring values that are
 temporarily overwritten.
 */
template <typename T>
class Store : private Non_copyable
{
public:
    /// Copy the original value.
    MI_FORCE_INLINE Store(T& t)
    : m_addr(t), m_val(t)
    {}

    /// Copy the original value and overwrite it.
    MI_FORCE_INLINE Store(T& t, const T& new_value)
    : m_addr(t), m_val(t)
    {
        t = new_value;
    }

    /// Restore the original value.
    MI_FORCE_INLINE void restore()
    { m_addr = m_val; }

    /// Restore the original value.
    MI_FORCE_INLINE ~Store()
    { restore(); }

private:
    T&  m_addr; ///< address
    T   m_val;  ///< value
};


}}


#endif //BASE_SYSTEM_STLEXT_RESTORE_H
