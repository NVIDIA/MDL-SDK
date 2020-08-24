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

// examples/mdl_sdk/dxr/mdl_d3d12/shader.h

#ifndef MDL_D3D12_SHADER_H
#define MDL_D3D12_SHADER_H

#include "common.h"

namespace mi { namespace examples { namespace mdl_d3d12
{
    class Base_application;
    class Raytracing_pipeline;

    // --------------------------------------------------------------------------------------------

    class Shader_compiler
    {
    public:
        IDxcBlob* compile_shader_library(
            const std::string& file_name,
            const std::map<std::string, std::string>* defines = nullptr);

        IDxcBlob* compile_shader_library_from_string(
            const std::string& shader_source,
            const std::string& debug_name,
            const std::map<std::string, std::string>* defines = nullptr);
    };

    // --------------------------------------------------------------------------------------------

    struct DxcBlobFromMemory : public IDxcBlob
    {
        explicit DxcBlobFromMemory(
            const char* blob_data,
            size_t blob_data_size)
            : m_buffer(blob_data, blob_data + blob_data_size)
            , m_cRef(0)
        {
        }

        LPVOID STDMETHODCALLTYPE GetBufferPointer(void) final { return m_buffer.data(); }
        SIZE_T STDMETHODCALLTYPE GetBufferSize(void) final { return m_buffer.size(); }

        // ---------------------------------------------------------------------------------------------

        HRESULT QueryInterface(REFIID riid, LPVOID* ppvObj)
        {
            if (!ppvObj) return E_INVALIDARG;
            *ppvObj = NULL;
            if (riid == IID_IUnknown || riid == __uuidof(IDxcBlob))
            {
                *ppvObj = (LPVOID)this;
                AddRef();
                return NOERROR;
            }
            return E_NOINTERFACE;
        }
        // ----------------------------------------------------------------------------------------

        ULONG AddRef()
        {
            InterlockedIncrement(&m_cRef);
            return m_cRef;
        }

        // ----------------------------------------------------------------------------------------

        ULONG Release()
        {
            ULONG ulRefCount = InterlockedDecrement(&m_cRef);
            if (0 == m_cRef)
                delete this;
            return ulRefCount;
        }

    private:
        std::vector<char> m_buffer;
        LONG m_cRef;
    };

    // --------------------------------------------------------------------------------------------

    class Shader
    {
    public:
        explicit Shader(Base_application* app);
        virtual ~Shader();

    private:
        Base_application* m_app;
    };

    // --------------------------------------------------------------------------------------------

    class Descriptor_table
    {
        friend class Root_signature;
    public:
        Descriptor_table() = default;
        Descriptor_table(const Descriptor_table& to_copy);
        Descriptor_table(Descriptor_table&& to_move);

        void register_cbv(size_t slot, size_t space, size_t heap_offset, size_t count = 1);
        void register_srv(size_t slot, size_t space, size_t heap_offset, size_t count = 1);
        void register_uav(size_t slot, size_t space, size_t heap_offset, size_t count = 1);

        size_t get_size() const { return m_descriptor_ranges.size(); }
        void clear() { return m_descriptor_ranges.clear(); }

    private:
        std::vector<CD3DX12_DESCRIPTOR_RANGE1> m_descriptor_ranges;
    };

    // --------------------------------------------------------------------------------------------

    class Root_signature
    {
        // --------------------------------------------------------------------

        struct Element
        {
            explicit Element();

            enum class Kind
            {
                None = 0,
                CBV,
                Constant,
                SRV,
                UAV,
                DescriptorTable,
            };

            Kind kind;
            size_t size_in_word;
            size_t root_signature_index;
        };

        // --------------------------------------------------------------------

    public:
        /// Constructor.
        explicit Root_signature(Base_application* app, const std::string& debug_name);

        /// Destructor.
        virtual ~Root_signature();

        /// Initialize a root signature entry for a constant that is directly placed in the
        /// signature.
        ///
        /// \param slot     The slot that the constant is bound to.
        ///                 Available in the shader as register(b<slot>).
        /// \return         True in case of success.
        template<typename T>
        bool register_constants(size_t slot)
        {
            return register_constants(slot, sizeof(T));
        }
        /// Initialize a root signature entry for a constant buffer view.
        ///
        /// \param slot     The slot that the CBV is bound to.
        ///                 Available in the shader as register(b<slot>).
        /// \return         True in case of success.
        bool register_cbv(size_t slot);

        /// Initialize a root signature entry for a unordered access view.
        ///
        /// \param slot     The slot that the UAV is bound to.
        ///                 Available in the shader as register(u<slot>).
        /// \return         True in case of success.
        bool register_uav(size_t slot);

        /// Initialize a root signature entry for a shader resource view.
        ///
        /// \param slot     The slot that the SRV is bound to.
        ///                 Available in the shader as register(t<slot>).
        /// \return         True in case of success.
        bool register_srv(size_t slot);

        /// Initialize a root signature entry for a descriptor table.
        ///
        /// \param descriptor_table     The table to register.
        /// \return                     True in case of success.
        bool register_dt(const Descriptor_table& descriptor_table);

        bool register_static_sampler(const D3D12_STATIC_SAMPLER_DESC& sampler_desc);

        /// Add a flag to the signature.
        ///
        /// \param flag     The flag to add using a logical or.
        /// \return         The resulting flag combination after adding.
        D3D12_ROOT_SIGNATURE_FLAGS add_flag(D3D12_ROOT_SIGNATURE_FLAGS flag);

        /// When the setup is complete the object has to be finalized in order to be used for
        /// rendering. Afterwards, no changes to the object are allowed anymore.
        bool finalize();

        /// Get the d3d root signature (available after finalize() has been called)
        ///
        /// \return         The signature used for setting up the pipeline or
        ///                 nullptr when called before finalize().
        ID3D12RootSignature* get_signature();

        /// Get the number of root signature entries,
        /// which is required for creating a corresponding
        size_t get_root_parameter_count() const { return m_root_parameters.size(); }

    private:
        bool register_constants(size_t slot, size_t size_in_byte);

        Base_application* m_app;
        const std::string m_debug_name;
        bool m_is_finalized;

        // Maximum 64 DWORDS divided up amongst all root parameters.
        // Root constants = 1 DWORD * NumConstants
        // Root descriptor (CBV, SRV, or UAV) = 2 DWORDs each
        // Descriptor table pointer = 1 DWORD
        // Static samplers = 0 DWORDS (compiled into shader)
        std::vector<CD3DX12_ROOT_PARAMETER1> m_root_parameters;
        std::unordered_map<size_t, Element> m_root_elements_b;
        std::unordered_map<size_t, Element> m_root_elements_t;
        std::unordered_map<size_t, Element> m_root_elements_u;
        std::vector<D3D12_STATIC_SAMPLER_DESC> m_static_samplers;
        std::vector<Element> m_root_elements_dt;
        D3D12_ROOT_SIGNATURE_FLAGS m_flags;
        ComPtr<ID3D12RootSignature> m_root_signature;
    };

    // --------------------------------------------------------------------------------------------

    class Shader_binding_tables
    {
        // --------------------------------------------------------------------

    private:
        struct Shader_record
        {
            Shader_record();

            void* m_shader_id;
            std::vector<uint8_t> m_local_root_arguments;

            uint8_t* m_mapped_table_pointer;                // used directly after finalize()
        };

        // --------------------------------------------------------------------

    public:
        struct Shader_handle
        {
            enum class Kind
            {
                invalid = -1,
                ray_generation = 0,
                miss = 1,
                hit_group = 2
            };
            friend class Shader_binding_tables;
            static const Shader_handle invalid;
            bool is_valid() const { return m_kind != Kind::invalid; }
            Kind get_kind() const { return m_kind; }

            explicit Shader_handle()
                : m_shader_binding_table(nullptr), m_kind(Kind::invalid), m_shader_id(nullptr) {}

            virtual ~Shader_handle() = default;
        private:
            explicit Shader_handle(
                Shader_binding_tables* binding_table, Kind kind, void* shader_id);

            Shader_binding_tables* m_shader_binding_table;
            void* m_shader_id;
            Kind m_kind;
        };

        // --------------------------------------------------------------------

        /// Constructor.
        explicit Shader_binding_tables(
            Raytracing_pipeline* pipeline,
            size_t ray_type_count,
            size_t hit_record_count,
            const std::string& debug_name);

        /// Destructor.
        virtual ~Shader_binding_tables();

        const Shader_handle add_ray_generation_program(const std::string& symbol_name);
        const Shader_handle add_miss_program(size_t ray_type, const std::string& symbol_name);
        const Shader_handle add_hit_group(size_t ray_type, const std::string& group_name);

        template<typename T>
        bool set_shader_record(
            size_t index,
            const Shader_handle& shader_handle,
            const T* local_root_arguments)
        {
            return set_shader_record(
                index, shader_handle,
                reinterpret_cast<const uint8_t*>(local_root_arguments),
                sizeof(T));
        }

        bool finalize();

        void upload(D3DCommandList* command_list);

        D3D12_DISPATCH_RAYS_DESC get_dispatch_description() const;

    private:

        const Shader_handle add_shader(Shader_handle::Kind kind, const std::string& name);
        bool set_shader_record(
            size_t index,
            const Shader_handle& shader_handle,
            const uint8_t* local_root_arguments,
            size_t size_in_byte);

        Base_application* m_app;
        const std::string m_debug_name;
        bool m_is_finalized;

        Raytracing_pipeline* m_pipeline;
        size_t m_ray_type_count;
        size_t m_hit_record_count;

        std::unordered_set<std::string> m_added_symbol_names;
        std::vector<Shader_record> m_shader_records[3];

        ComPtr<ID3D12Resource> m_binding_table_buffer_upload;
        ComPtr<ID3D12Resource> m_binding_table_buffer;
        uint8_t* m_mapped_binding_table;
        D3D12_DISPATCH_RAYS_DESC m_prefilled_dispatch_description;
    };

}}} // mi::examples::mdl_d3d12
#endif
