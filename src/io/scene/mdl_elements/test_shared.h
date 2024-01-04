/******************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

#ifndef IO_SCENE_MDL_ELEMENTS_TEST_SHARED_H
#define IO_SCENE_MDL_ELEMENTS_TEST_SHARED_H

#include <base/system/main/access_module.h>
#include <base/lib/mem/mem.h>
#include <base/lib/log/i_log_module.h>
#include <base/data/db/i_db_database.h>
#include <base/data/db/i_db_transaction.h>
#include <mdl/integration/mdlnr/i_mdlnr.h>

#include <base/data/dblight/i_dblight.h>
#include <io/scene/scene/i_scene_mdl_sdk.h>

namespace MI {

class Unified_database_access
{
public:
    Unified_database_access();
    ~Unified_database_access();
    DB::Database* get_database() { return m_database; }
private:
    DB::Database* m_database;

    SYSTEM::Access_module<LOG::Log_module> m_log_module;
    SYSTEM::Access_module<MDLC::Mdlc_module> m_mdlc_module;
};

Unified_database_access::Unified_database_access()
{
    m_log_module.set();
    // For IMDL::is_valid_mdl_identifier() in IType_factory::create_deferred_sized_array().
    m_mdlc_module.set();
    m_database = MI::DBLIGHT::factory();
    SCENE::register_db_elements( m_database);
}

Unified_database_access::~Unified_database_access()
{
    m_database->close();
    m_database = nullptr;
}

} // namespace MI

#endif // IO_SCENE_MDL_ELEMENTS_TEST_SHARED_H

