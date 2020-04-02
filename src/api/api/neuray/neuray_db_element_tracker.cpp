/***************************************************************************************************
 * Copyright (c) 2010-2020, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Source for the Db_element_tracker.
 **/

#include "pch.h"

#include "neuray_db_element_tracker.h"

#include "neuray_class_factory.h"
#include "neuray_db_element_impl.h"

#include <sstream>
#include <mi/base/handle.h>
#include <base/lib/log/i_log_assert.h>


namespace MI {

namespace NEURAY {

Db_element_tracker::Db_element_tracker()
  : m_initialized( false)
{
}

Db_element_tracker::~Db_element_tracker()
{
    ASSERT( M_NEURAY_API, m_elements.empty());
}

void Db_element_tracker::add_element( const Db_element_impl_base* db_element)
{
    mi::base::Lock::Block block(&m_lock);

    // Note: we store a pointer to a reference-counted object without calling retain() here.
    // This is not a problem since this method will only be called from the constructor of that
    // object (and the corresponding method from the destructor of that object).
    //
    // Reference counting as usual is not possible since that would increase the reference count,
    // and the object would never go out of scope.
    m_elements.insert( db_element);
}

void Db_element_tracker::remove_element( const Db_element_impl_base* db_element)
{
    // Note: we remove a pointer to reference-counted object without calling release() here.
    // This is not a problem since this method will only be called from the destructor of that
    // object (and the corresponding method from the constructor of that object).
    //
    // Reference counting as usual is not possible since that would increase the reference count,
    // and the object would never go out of scope.
    mi::base::Lock::Block block( &m_lock);
    m_elements.erase( db_element);
}


} // namespace NEURAY

} // namespace MI

