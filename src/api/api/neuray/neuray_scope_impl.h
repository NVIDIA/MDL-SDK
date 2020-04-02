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

/** \file
 ** \brief Implementation of IScope
 **
 ** Implementation of IScope
 **/

#ifndef API_API_NEURAY_SCOPE_IMPL_H
#define API_API_NEURAY_SCOPE_IMPL_H

#include <mi/base/handle.h>
#include <mi/base/lock.h>
#include <mi/base/interface_implement.h>
#include <mi/neuraylib/iscope.h>

#include <string>
#include <base/data/db/i_db_scope.h>
#include <boost/core/noncopyable.hpp>

namespace MI {

namespace DB { class Transaction; }

namespace NEURAY {

class Class_factory;

class Scope_impl
  : public mi::base::Interface_implement<mi::neuraylib::IScope>,
    public boost::noncopyable
{
public:
    /// Constructs a Scope_impl
    Scope_impl( DB::Scope* scope, const Class_factory* class_factory);

    /// Destructs a Scope_impl
    ~Scope_impl();

    // public API methods

    mi::neuraylib::ITransaction* create_transaction();

    const char* get_id() const;

    const char* get_name() const;

    mi::Uint8 get_privacy_level() const;

    DB::Scope* get_scope() const;

    mi::neuraylib::IScope* get_parent() const;

    // internal methods

private:
    /// The wrapped scope.
    DB::Scope* m_scope;

    /// The scope name. Cached here for methods with const char* return value.
    std::string m_name;

    /// The scope ID. Cached here for methods with const char* return value.
    std::string m_id;

    /// Pointer to the class factory.
    const Class_factory* m_class_factory;
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_SCOPE_IMPL_H
