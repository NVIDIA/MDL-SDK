/******************************************************************************
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

 // examples/mdl_sdk/dxr/mdl_d3d12/mdl_sdk.h

#ifndef MDL_D3D12_MDL_SDK_H
#define MDL_D3D12_MDL_SDK_H

#include "common.h"
#include <mi/mdl_sdk.h>

namespace mdl_d3d12
{
    class Base_application;
    class Mdl_transaction;
    class Mdl_material_library;

    enum class Mdl_resource_kind
    {
        Texture,
        // Light_profile,
        // Bsdf_measurement,
        _Count
    };

    class Mdl_sdk
    {
    public:
        explicit Mdl_sdk(Base_application* app);
        virtual ~Mdl_sdk();

        bool is_running() const { return m_hlsl_backend.is_valid_interface(); }

        /// logs errors, warnings, infos, ... and returns true in case the was NO error
        bool log_messages(const mi::neuraylib::IMdl_execution_context* context);

        mi::neuraylib::INeuray& get_neuray() { return *m_neuray; }
        mi::neuraylib::IDatabase& get_database() { return *m_database; }
        mi::neuraylib::IMdl_compiler& get_compiler() { return *m_mdl_compiler; }
        mi::neuraylib::IImage_api& get_image_api() { return *m_image_api; }
        mi::neuraylib::IMdl_backend& get_backend() { return *m_hlsl_backend; }

        // Creates a new execution context. At least one per thread is required.
        // This means you can share the context for multiple calls from the same thread.
        // However, sharing is not required. Creating a context for each call is valid too but
        // slightly more expensive.
        // Use a neuray handle to hold the pointer returned by this function.
        mi::neuraylib::IMdl_execution_context* create_context();

        size_t get_num_texture_results() const { return 16; }

        /// access point to the database 
        Mdl_transaction& get_transaction() { return *m_transaction; }

        /// keeps all materials that are loaded by the application
        Mdl_material_library* get_library() { return m_library; }

        /// enable or disable MDL class compilation mode
        bool use_class_compilation;

    private:
        Base_application* m_app;

        mi::base::Handle<mi::neuraylib::INeuray> m_neuray;
        mi::base::Handle<mi::neuraylib::IDatabase> m_database;
        mi::base::Handle<mi::neuraylib::IMdl_compiler> m_mdl_compiler;
        mi::base::Handle<mi::neuraylib::IImage_api> m_image_api;
        mi::base::Handle<mi::neuraylib::IMdl_factory> m_mdl_factory;
        mi::base::Handle<mi::neuraylib::IMdl_backend> m_hlsl_backend;

        Mdl_transaction* m_transaction;
        Mdl_material_library* m_library;
    };


    class Mdl_transaction
    {
        // make sure there is only one transaction
        friend class Mdl_sdk;

        explicit Mdl_transaction(Mdl_sdk* sdk);
    public:
        virtual ~Mdl_transaction();

        // runs an operation on the database.
        // concurrent calls are executed in sequence using a lock.
        template<typename TRes>
        TRes execute(std::function<TRes(mi::neuraylib::ITransaction* t)> action)
        {
            std::lock_guard<std::mutex> lock(m_transaction_mtx);
            return action(m_transaction.get());
        }

        template<>
        void execute<void>(std::function<void(mi::neuraylib::ITransaction* t)> action)
        {
            std::lock_guard<std::mutex> lock(m_transaction_mtx);
            action(m_transaction.get());
        }

        // locked database access function
        template<typename TIInterface>
        const TIInterface* access(const char* db_name)
        {
            return execute<const TIInterface*>(
                [&](mi::neuraylib::ITransaction* t)
            {
                return t->access<TIInterface>(db_name);
            });
        }

        // locked database access function
        template<typename TIInterface>
        TIInterface* edit(const char* db_name)
        {
            return execute<TIInterface*>(
                [&](mi::neuraylib::ITransaction* t)
            {
                return t->edit<TIInterface>(db_name);
            });
        }

        // locked database create function
        template<typename TIInterface>
        TIInterface* create(const char* type_name)
        {
            return execute<TIInterface*>(
                [&](mi::neuraylib::ITransaction* t)
            {
                return t->create<TIInterface>(type_name);
            });
        }

        // locked database store function
        template<typename TIInterface>
        mi::Sint32 store(
            TIInterface* db_element, 
            const char* name)
        {
            return execute<mi::Sint32>(
                [&](mi::neuraylib::ITransaction* t)
            {
                return t->store(db_element, name);
            });
        }

        // locked database commit function.
        // For that, all handles to neuray objects have to be released.
        // Initializes for further actions afterwards.
        void commit();

        mi::neuraylib::ITransaction* get() { return m_transaction.get(); }

    private:
        mi::base::Handle<mi::neuraylib::ITransaction> m_transaction;
        std::mutex m_transaction_mtx;
        Mdl_sdk* m_sdk;
    };


}

#endif
