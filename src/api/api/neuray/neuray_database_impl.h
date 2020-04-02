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
 ** \brief Implementation of IDatabase
 **
 ** Implements the IDatabase interface
 **/

#ifndef API_API_NEURAY_DATABASE_IMPL_H
#define API_API_NEURAY_DATABASE_IMPL_H

#include <mi/base/interface_implement.h>
#include <mi/neuraylib/idatabase.h>
#include <mi/neuraylib/ineuray.h>

#include <boost/core/noncopyable.hpp>

namespace MI {

namespace DB { class Database; }

namespace NEURAY {

class Class_factory;

/// This API component is shared between Neuray_impl and Cluster_impl. It must not use
/// Access_module<DATA::Data_module> or DATA::mod_data directly.
class Database_impl
  : public mi::base::Interface_implement<mi::neuraylib::IDatabase>,
    public boost::noncopyable
{
public:
    /// Constructor of Database_impl
    ///
    /// \param status           The status of the interface this API component belongs to.
    Database_impl( mi::neuraylib::INeuray::Status& status);

    /// Destructor of Database_impl
    ~Database_impl();

    // public API methods

    mi::neuraylib::IScope* get_global_scope() const;

    mi::neuraylib::IScope* create_scope( mi::neuraylib::IScope* parent, mi::Uint8 level, bool temp);

    mi::neuraylib::IScope* create_or_get_named_scope(
        const char* name, mi::neuraylib::IScope* parent, mi::Uint8 privacy_level);

    mi::neuraylib::IScope* get_scope( const char* id) const;

    mi::neuraylib::IScope* get_named_scope( const char* name) const;

    mi::Sint32 remove_scope( const char* id) const;

    void lock( mi::Uint32 lock_id);

    mi::Sint32 unlock( mi::Uint32 lock_id);

    void garbage_collection();

    // internal methods

    /// Starts this API component.
    /// 
    /// The implementation of INeuray::start() calls the #start() method of each API component.
    /// This method performs the API component's specific part of the library start.
    ///
    /// \param database    The database to be used by this API component.
    /// \return            0, in case of success, -1 in case of failure.
    mi::Sint32 start( DB::Database* database);

    /// Shuts down this API component.
    ///
    /// The implementation of INeuray::shutdown() calls the #shutdown() method of each API
    /// component. This method performs the API component's specific part of the library shutdown.
    ///
    /// \return           0, in case of success, -1 in case of failure
    mi::Sint32 shutdown();

private:
    /// The status of the interface this API component belongs to (INeuray/ICluster).
    mi::neuraylib::INeuray::Status& m_status;
    
    /// The database to be used by this API component.
    DB::Database* m_database;
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_DATABASE_IMPL_H
