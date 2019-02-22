/******************************************************************************
 * Copyright (c) 2012-2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_GENERATOR_JIT_GENERATED_CODE
#define MDL_GENERATOR_JIT_GENERATED_CODE 1

#include <mi/base/handle.h>
#include <mi/mdl/mdl_generated_executable.h>

#include <mdl/compiler/compilercore/compilercore_cc_conf.h>
#include <mdl/compiler/compilercore/compilercore_allocator.h>
#include <mdl/compiler/compilercore/compilercore_messages.h>

#include <mdl/codegenerators/generator_dag/generator_dag_lambda_function.h>

#include <llvm/ADT/OwningPtr.h>
#include <llvm/IR/LLVMContext.h>

#include "generator_jit_llvm.h"
#include "generator_jit_res_manager.h"

namespace llvm {
    class Module;
}

namespace mi {
namespace mdl {

/// Layout structure.
typedef struct {
    unsigned char  kind;                 ///< mi::mdl::IValue::Kind
    unsigned char  alloc_size_padding;   ///< number of padding bytes after the element
    unsigned short _unused;
    unsigned       element_size;         ///< the size of the element without alloc_size_padding
    unsigned       element_offset;       ///< for compound elements, aligned offset of first element
                                         ///< relative to start of parent
    unsigned short num_children;         ///< only non-zero for compound elements
    unsigned short children_state_offs;  ///< state offset for children
} Layout_struct;


/// Represents the layout of an argument value block with support for nested elements.
/// Implementation of #mi::mdl::IGenerated_code_value_layout.
class Generated_code_value_layout : public
    Allocator_interface_implement<IGenerated_code_value_layout>
{
    typedef Allocator_interface_implement<IGenerated_code_value_layout> Base;

public:
    /// Constructor.
    ///
    /// \param alloc     The allocator.
    /// \param code_gen  The LLVM code generator providing the information about the captured
    ///                  arguments and the type mapper.
    Generated_code_value_layout(
        IAllocator          *alloc,
        LLVM_code_generator *code_gen);

    /// Constructor from layout data block.
    ///
    /// \param alloc                  The allocator.
    /// \param layout_data_block      The data block containing the layout data.
    /// \param layout_data_size       The size of the layout data block.
    /// \param strings_mapped_to_ids  True, if strings are mapped to IDs.
    Generated_code_value_layout(
        IAllocator *alloc,
        char const *layout_data_block,
        size_t     layout_data_size,
        bool       strings_mapped_to_ids);

    /// Returns the size of the target argument block.
    size_t get_size() const MDL_FINAL;

    /// Returns the number of arguments / elements at the given layout state.
    ///
    /// \param state  The layout state representing the current nesting within the
    ///               argument value block. The default value is used for the top-level.
    size_t get_num_elements(State state = State()) const MDL_FINAL;

    /// Get the offset, the size and the kind of the argument / element inside the argument
    /// block at the given layout state.
    ///
    /// \param[out]  kind      Receives the kind of the argument.
    /// \param[out]  arg_size  Receives the size of the argument.
    /// \param       state     The layout state representing the current nesting within the
    ///                        argument value block. The default value is used for the top-level.
    ///
    /// \returns the offset of the requested argument / element or ~size_t  (0) if the state
    ///          is invalid.
    size_t get_layout(
        mi::mdl::IValue::Kind &kind,
        size_t                &arg_size,
        State                 state = State()) const MDL_FINAL;

    /// Get the layout state for the i'th argument / element inside the argument value block
    /// at the given layout state.
    ///
    /// \param i      The index of the argument / element.
    /// \param state  The layout state representing the current nesting within the argument
    ///               value block. The default value is used for the top-level.
    ///
    /// \returns the layout state for the nested element or a state with m_state_offs set to
    ///          ~mi::Uint32(0) if the element is atomic.
    IGenerated_code_value_layout::State get_nested_state(
        size_t i,
        State  state = State()) const MDL_FINAL;

    /// Set the value inside the given block at the given layout state.
    ///
    /// \param[inout] block           The argument value block buffer to be modified.
    /// \param[in]    value           The value to be set. It has to match the expected kind.
    /// \param[in]    value_callback  Callback for retrieving resource indices for resource values.
    /// \param[in]    state           The layout state representing the current nesting within the
    ///                               argument value block.
    ///                               The default value is used for the top-level.
    ///
    /// \return
    ///                      -  0: Success.
    ///                      - -1: Invalid parameters, block or value is a \c NULL pointer.
    ///                      - -2: Invalid state provided.
    ///                      - -3: Value kind does not match expected kind.
    ///                      - -4: Size of compound value does not match expected size.
    ///                      - -5: Unsupported value type.
    int set_value(
        char                           *block,
        mi::mdl::IValue const          *value,
        IGenerated_code_value_callback *value_callback,
        State                          state = State()) const MDL_FINAL;

    // Non-API methods

    /// Get the layout data buffer and its size.
    char const *get_layout_data(size_t &size);

private:
    /// The layout data buffer.
    vector<char>::Type m_layout_data;

    /// If true, string argument values are mapped to string identifiers.
    bool m_strings_mapped_to_ids;
};


///
/// Implementation of generated executable code.
///
class Generated_code_jit : public Allocator_interface_implement<IGenerated_code_executable>
{
    typedef Allocator_interface_implement<IGenerated_code_executable> Base;

    friend class JIT_code_printer;
public:
    // The code generator produces compiler errors.
    static char const MESSAGE_CLASS = 'C';

    /// Constructor.
    ///
    /// \param alloc         The allocator.
    /// \param jitted_code   The jitted code singleton.
    /// \param filename      The file name for generated messages.
    explicit Generated_code_jit(
        IAllocator  *alloc,
        Jitted_code *jitted_code,
        char const  *filename);

    /// Get the kind of code generated.
    /// \returns    The kind of generated code.
    IGenerated_code::Kind get_kind() const MDL_FINAL;

    /// Get the target language.
    /// \returns    The name of the target language for which this code was generated.
    char const *get_target_language() const MDL_FINAL;

    /// Check if the code contents are valid.
    bool is_valid() const MDL_FINAL;

    /// Access messages.
    Messages const &access_messages() const MDL_FINAL;

    /// Acquires a const interface.
    ///
    /// If this interface is derived from or is the interface with the passed
    /// \p interface_id, then return a non-\c NULL \c const #mi::base::IInterface* that
    /// can be casted via \c static_cast to an interface pointer of the interface type
    /// corresponding to the passed \p interface_id. Otherwise return \c NULL.
    ///
    /// In the case of a non-\c NULL return value, the caller receives ownership of the 
    /// new interface pointer, whose reference count has been retained once. The caller 
    /// must release the returned interface pointer at the end to prevent a memory leak.
    mi::base::IInterface const *get_interface(
        mi::base::Uuid const &interface_id) const MDL_FINAL;

    /// Returns the source code of the module if available.
    ///
    /// \param size  will be assigned to the length of the source code
    /// \returns the source code or NULL if no source is available.
    ///
    /// \note The source code might be generated lazily.
    char const *get_source_code(size_t &size) const MDL_FINAL;

    /// Get the data for the read-only data segment if available.
    ///
    /// \param size  will be assigned to the length of the RO data segment
    /// \returns the data segment or NULL if no RO data segment is available.
    char const *get_ro_data_segment(size_t &size) const MDL_FINAL;

    /// Get the used state properties of  the generated lambda function code.
    State_usage get_state_usage() const MDL_FINAL;

    /// Get the number of captured argument block layouts.
    size_t get_captured_argument_layouts_count() const MDL_FINAL;

    /// Get a captured arguments block layout if available.
    ///
    /// \param i   the index of the block layout
    ///
    /// \returns the layout or NULL if no arguments were captured or the index is invalid.
    IGenerated_code_value_layout const *get_captured_arguments_layout(size_t i)
        const MDL_FINAL;

    /// Get the number of mapped string constants used inside the generated code.
    size_t get_string_constant_count() const MDL_FINAL;

    /// Get the mapped string constant for a given id.
    ///
    /// \param id  the string id (as used in the generated code)
    ///
    /// \return the MDL string constant that corresponds to the given id or NULL
    ///         if id is out of range
    ///
    /// \note that the id 0 is ALWAYS mapped to the empty string ""
    char const *get_string_constant(size_t id) const MDL_FINAL;

    // non-interface methods

    /// Compile a whole MDL module into LLVM-IR.
    ///
    /// \param module             The MDL module to generate code from.
    /// \param options            The backend options.
    void compile_module_to_llvm(
        mi::mdl::IModule const *module,
        Options_impl const     &options);

    /// Compile a whole MDL module into PTX.
    ///
    /// \param module             The MDL module to generate code from.
    /// \param options            The backend options.
    void compile_module_to_ptx(
        mi::mdl::IModule const *module,
        Options_impl const     &options);

    /// Retrieve the LLVM context of this jitted code.
    llvm::LLVMContext &get_llvm_context() { return m_llvm_context; }

    /// Add a mapped string.
    ///
    /// \param s   the string constant
    /// \param id  the assigned id for this constant
    void add_mapped_string(char const *s, size_t id);

private:

    /// Destructor.
    ~Generated_code_jit() MDL_OVERRIDE;

    // no copy constructor.
    Generated_code_jit(Generated_code_jit const &) MDL_DELETED_FUNCTION;

    // no assignment operator
    Generated_code_jit const &operator=(Generated_code_jit const &) MDL_DELETED_FUNCTION;

    /// Retrieve the LLVM module.
    llvm::Module const *get_llvm_module() const { return m_llvm_module.get(); }

    /// Retrieve the PTX code.
    char const *get_ptx_code() const { return m_ptx_code.c_str(); }

private:
    /// The allocator builder.
    mutable Allocator_builder m_builder;

    /// The context of the module, its life time must include the module ...
    llvm::LLVMContext m_llvm_context;

    /// The llvm module.
    llvm::OwningPtr<llvm::Module> m_llvm_module;

    /// A Reference to the jitted code singleton.
    mi::base::Handle<Jitted_code> m_jitted_code;

    /// The messages if any.
    Messages_impl m_messages;

    /// Generated PTX code if any.
    string m_ptx_code;

    /// The render state usage.
    State_usage m_render_state_usage;

    typedef vector<string>::Type Mappend_string_vector;

    /// The mapped strings
    Mappend_string_vector m_mappend_strings;
};

/// The implementation of a source code, used for PTX or LLVM-IR.
class Generated_code_source :
    public Allocator_interface_implement<IGenerated_code_executable>
{
    typedef Allocator_interface_implement<IGenerated_code_executable> Base;
    friend class Allocator_builder;
public:
    typedef vector<size_t>::Type Offset_vec;

    /// The resource manager for generated source code.
    class Source_res_manag : public IResource_manager {
        typedef mi::mdl::hash_map<unsigned, size_t>::Type         Tag_index_map;
        typedef mi::mdl::hash_map<
            mi::mdl::string, size_t, string_hash<string> >::Type  String_index_map;
    public:
        /// Constructor.
        ///
        /// \param alloc         The allocator.
        /// \param resource_map  If non-NULL, use this map to resolve resources
        Source_res_manag(
            IAllocator              *alloc,
            Resource_attr_map const *resource_map);

        /// Register the given resource value and return its 1-based index in the resource table.
        /// Index 0 represents an invalid resource reference.
        ///
        /// \param resource  the MDL resource to register
        size_t get_resource_index(IValue_resource const *resource) MDL_FINAL;

        /// Register a string constant and return its 1 based index in the string table.
        ///
        /// \param string  the MDL string value to register
        size_t get_string_index(IValue_string const *string) MDL_FINAL;

        /// Sets a new resource attribute map.
        ///
        /// \param resource_map  if non-NULL, the new map to be used
        void set_resource_attribute_map(Resource_attr_map const *resource_map) {
            m_resource_map = resource_map;
        }

    private:
        /// The current allocator.
        IAllocator              *m_alloc;

        /// The resource-attribute-map if given.
        Resource_attr_map const *m_resource_map;

        /// Lookup-table for resource indexes.
        Tag_index_map           m_res_indexes;

        /// Lookup-table for string indexes;
        String_index_map        m_string_indexes;

        /// The current resource index.
        size_t m_curr_res_idx;

        /// The current string index.
        size_t m_curr_string_idx;
    };

public:
    // -------------------from IGenerated_code -------------------

    /// Get the kind of code generated.
    IGenerated_code::Kind get_kind() const MDL_FINAL;

    /// Get the target language.
    char const *get_target_language() const MDL_FINAL;

    /// Check if the code contents are valid.
    bool is_valid() const MDL_FINAL;

    /// Access messages.
    Messages const &access_messages() const MDL_FINAL;

    /// Returns the source code of the module if available.
    ///
    /// \param size  will be assigned to the length of the assembler code
    /// \returns the source code or NULL if no source code is available.
    ///
    /// \note The source code might be generated lazily.
    char const *get_source_code(size_t &size) const MDL_FINAL;

    /// Get the data for the read-only data segment if available.
    ///
    /// \param size  will be assigned to the length of the RO data segment
    /// \returns the data segment or NULL if no RO data segment is available.
    char const *get_ro_data_segment(size_t &size) const MDL_FINAL;

    /// Get the used state properties of  the generated lambda function code.
    State_usage get_state_usage() const MDL_FINAL;

    /// Get the number of captured argument block layouts.
    size_t get_captured_argument_layouts_count() const MDL_FINAL;

    /// Get a captured arguments block layout if available.
    ///
    /// \param i   the index of the block layout
    ///
    /// \returns the layout or NULL if no arguments were captured or the index is invalid.
    IGenerated_code_value_layout const *get_captured_arguments_layout(
        size_t i) const MDL_FINAL;

    /// Get the number of mapped string constants used inside the generated code.
    size_t get_string_constant_count() const MDL_FINAL;

    /// Get the mapped string constant for a given id.
    ///
    /// \param id  the string id (as used in the generated code)
    ///
    /// \return the MDL string constant that corresponds to the given id or NULL
    ///         if id is out of range
    ///
    /// \note that the id 0 is ALWAYS mapped to the empty string ""
    char const *get_string_constant(size_t id) const MDL_FINAL;

    // -------------------- non-interface methods --------------------

    /// Write access to the source code.
    string &access_src_code() { return m_src_code; }

    /// Write access to the messages.
    Messages_impl &access_messages() { return m_messages; }

    /// Set the Read-Only data segment.
    void set_ro_segment(char const *data, size_t size) {
        m_ro_segment.assign(data, data + size);
    }

    /// Set the render state usage.
    void set_render_state_usage(IGenerated_code_lambda_function::State_usage usage)
    {
        m_render_state_usage = usage;
    }

    /// Add a captured arguments layout.
    void add_captured_arguments_layout(IGenerated_code_value_layout const *layout)
    {
        m_captured_arguments_layouts.push_back(mi::base::make_handle_dup(layout));
    }

    /// Add a mapped string.
    ///
    /// \param s   the string constant
    /// \param id  the assigned id for this constant
    void add_mapped_string(char const *s, size_t id);

private:
    /// Constructor.
    ///
    /// \param alloc  the allocator
    /// \param kind   the kind, either CK_PTX or CK_LLVM_IR
    explicit Generated_code_source(
        IAllocator            *alloc,
        IGenerated_code::Kind kind);

    /// Destructor.
    ~Generated_code_source() MDL_OVERRIDE;

    // no copy constructor
    Generated_code_source(Generated_code_source const &) MDL_DELETED_FUNCTION;

    // no assignment operator
    Generated_code_source const &operator=(Generated_code_source const &) MDL_DELETED_FUNCTION;

private:
    /// The kind of this source.
    IGenerated_code::Kind const m_kind;

    /// The render state usage.
    State_usage m_render_state_usage;

    /// The Messages.
    Messages_impl m_messages;

    /// The source code.
    string m_src_code;

    typedef vector<char>::Type Byte_vec;

    /// The RO data segment.
    Byte_vec m_ro_segment;

    typedef vector<mi::base::Handle<IGenerated_code_value_layout const> >::Type Layout_vec;

    /// The list of captured arguments block layouts.
    Layout_vec m_captured_arguments_layouts;

    typedef vector<string>::Type Mappend_string_vector;

    /// The mapped strings
    Mappend_string_vector m_mappend_strings;
};

/// The implementation of a compiled lambda function.
class Generated_code_lambda_function :
    public Allocator_interface_implement<IGenerated_code_lambda_function>
{
    typedef Allocator_interface_implement<IGenerated_code_lambda_function> Base;
    friend class Allocator_builder;

    /// Helper value class to handle resource entries.
    class Resource_entry {
    public:
        /// Constructor.
        Resource_entry(mi::Uint32 tag, IType_resource const *type)
        : m_tag(tag)
        , m_type(type)
        , m_gamma_mode(IValue_texture::gamma_default)
        {
        }

        /// Constructor for texture entries.
        Resource_entry(
            mi::Uint32                 tag,
            IType_texture const        *tex_type,
            IValue_texture::gamma_mode gamma_mode)
        : m_tag(tag)
        , m_type(tex_type)
        , m_gamma_mode(gamma_mode)
        {
        }

        /// Get the tag of this resource entry.
        mi::Uint32 get_tag() const { return m_tag; }

        /// Get the type of this resource entry.
        IType_resource const *get_type() const { return m_type; }

        /// Get the gamma mode of this texture entry.
        IValue_texture::gamma_mode get_gamma_mode() const { return m_gamma_mode; }

    private:
        /// The tag.
        mi::Uint32                 m_tag;
        /// The resource type of the tag.
        IType_resource const       *m_type;
        /// If the type is a texture type, the MDL gamma mode of this texture.
        IValue_texture::gamma_mode m_gamma_mode;
    };

public:
    /// The resource manager for lambda functions.
    ///
    /// This version registers all resource values with the compiled lambda function, so
    /// resource-callbacks can be used.
    class Lambda_res_manag : public IResource_manager {
        typedef mi::mdl::hash_map<unsigned, size_t>::Type        Tag_index_map;
        typedef mi::mdl::hash_map<
            mi::mdl::string, size_t, string_hash<string> >::Type String_index_map;
    public:
        /// Constructor.
        ///
        /// \param lambda        The compiled lambda function.
        /// \param resource_map  If non-NULL, use this map to resolve resources
        Lambda_res_manag(
            Generated_code_lambda_function &lambda,
            Resource_attr_map const       *resource_map);

        /// Register the given resource value and return its 1-based index in the resource table.
        /// Index 0 represents an invalid resource reference.
        ///
        /// \param resource  the MDL resource to register
        size_t get_resource_index(IValue_resource const *resource) MDL_FINAL;

        /// Register a string constant and return its 1 based index in the string table.
        ///
        /// \param string  the MDL string value to register
        size_t get_string_index(IValue_string const *string) MDL_FINAL;

        /// Registers all resources in the given resource map in the order of the associated
        /// indices.
        ///
        /// \param resource_map  The resource attribute map to import.
        void import_from_resource_attribute_map(Resource_attr_map const *resource_map);

    private:
        /// The compiled lambda function.
        Generated_code_lambda_function &m_lambda;

        /// The resource-attribute-map if given.
        Resource_attr_map const        *m_resource_map;

        /// Lookup-table for resource indexes.
        Tag_index_map                  m_res_indexes;

        /// Lookup-table for string indexes.
        String_index_map               m_string_indexes;
    };

    /// The resource data helper class.
    class Res_data {
        friend class Generated_code_lambda_function;
    public:
        /// Default constructor.
        Res_data()
            : m_obj_size(0)
            , m_res_arr(NULL)
            , m_resource_handler(NULL)
        {
        }

        /// The size of one resource_data entry.
        size_t get_obj_size() const { return m_obj_size; }

        /// Get the blob entry for the given resource index.
        char const *get_resource_store(size_t idx) const { return m_res_arr + idx * m_obj_size; }

        /// Get the current resource handler.
        IResource_handler const *get_resource_handler() const { return m_resource_handler; }

        // Clear the data.
        void clear() { m_obj_size = 0; m_res_arr = NULL; m_resource_handler = NULL; }

    private:
        /// The size of one resource_data entry.
        size_t m_obj_size;

        /// The resource_data blob.
        char *m_res_arr;

        /// The current resource handler.
        IResource_handler *m_resource_handler;
    };

    /// The resource data pair helper class.
    class Res_data_pair {
        friend class Type_mapper;
    public:
        /// Constructor.
        ///
        /// \param res_data     the resource data, shared by ALL threads
        /// \param thread_data  additional "per thread" data, passed to the resource handler
        Res_data_pair(Res_data const &res_data, void *thread_data)
            : m_shared_data(&res_data), m_thread_data(thread_data)
        {
        }

        /// Get the shared data.
        Res_data const *get_shared_data() const { return m_shared_data; }

        /// Get the thread data.
        void *get_thread_data() const { return m_thread_data; }

    private:
        /// The read-only texture data, shared between ALL threads.
        Res_data const *m_shared_data;

        /// Per-thread data.
        void           *m_thread_data;
    };

public:
    // -------------------from IGenerated_code -------------------

    /// Get the kind of code generated.
    IGenerated_code::Kind get_kind() const MDL_FINAL;

    /// Get the target language.
    char const *get_target_language() const MDL_FINAL;

    /// Check if the code contents are valid.
    bool is_valid() const MDL_FINAL;

    /// Access messages.
    Messages const &access_messages() const MDL_FINAL;

    /// Returns the source code of the module if available.
    ///
    /// \param size  will be assigned to the length of the assembler code
    /// \returns the source code or NULL if no source code is available.
    ///
    /// \note The source code might be generated lazily.
    char const *get_source_code(size_t &size) const MDL_FINAL;

    /// Get the data for the read-only data segment if available.
    ///
    /// \param size  will be assigned to the length of the RO data segment
    /// \returns the data segment or NULL if no RO data segment is available.
    char const *get_ro_data_segment(size_t &size) const MDL_FINAL;

    /// Get the used state properties of  the generated lambda function code.
    State_usage get_state_usage() const MDL_FINAL;

    /// Get the number of captured argument block layouts.
    size_t get_captured_argument_layouts_count() const MDL_FINAL;

    /// Get a captured arguments block layout if available.
    ///
    /// \param i   the index of the block layout
    ///
    /// \returns the layout or NULL if no arguments were captured or the index is invalid.
    IGenerated_code_value_layout const *get_captured_arguments_layout(size_t i)
        const MDL_FINAL;

    /// Get the number of mapped string constants used inside the generated code.
    size_t get_string_constant_count() const MDL_FINAL;

    /// Get the mapped string constant for a given id.
    ///
    /// \param id  the string id (as used in the generated code)
    ///
    /// \return the MDL string constant that corresponds to the given id or NULL
    ///         if id is out of range
    ///
    /// \note that the id 0 is ALWAYS mapped to the empty string ""
    char const *get_string_constant(size_t id) const MDL_FINAL;

    // ------------------- own methods -------------------

    /// Write access to the messages.
    Messages_impl &access_messages() { return m_messages; }

    /// Initialize a JIT compiled lambda function.
    ///
    /// \param[in] ctx          a used defined context parameter
    /// \param[in] exc_handler  the handler for MDL exceptions or NULL
    /// \param[in] res_handler  the handler for resources or NULL
    ///
    /// exc_handler and res_handler are currently only used in CPU mode.
    /// If exc_handler is NULL, no exceptions are reported, but the function is still aborted
    /// if an exception occurs.
    /// If res_handler is set to NULL, iray-style resource handling is used.
    /// The context ctx is only passed to methods of the res_handler interface and otherwise
    /// unused.
    void init(
        void                   *ctx,
        IMDL_exception_handler *exc_handler,
        IResource_handler      *res_handler) MDL_FINAL;

    /// Terminates the resource handling.
    void term() MDL_FINAL;

    /// Run a compiled lambda functions as an environment function on the CPU.
    ///
    /// \param[in]  index     the index of the function to execute
    /// \param[out] result    out: the result will be written to
    /// \param[in]  state     the state of the shader
    /// \param[in]  tex_data  extra thread data for the texture handler
    ///
    /// \returns false if execution was aborted by runtime error, true otherwise
    ///
    /// Executes a compiled environment function.
    bool run_environment(
        size_t                          index,
        RGB_color                       *result,
        Shading_state_environment const *state,
        void                            *tex_data) MDL_FINAL;

    /// Run this compiled lambda functions as a uniform function returning bool on the CPU.
    ///
    /// \param[out] result  the result will be written to
    ///
    /// \returns false if execution was aborted by runtime error, true otherwise
    ///
    /// Executes a compiled uniform function returning bool.
    bool run(bool &result) MDL_FINAL;

    /// Run this compiled lambda functions as a uniform function returning int on the CPU.
    ///
    /// \param[out] result  the result will be written to
    ///
    /// \returns false if execution was aborted by runtime error, true otherwise
    ///
    /// Executes a compiled uniform function returning int.
    bool run(int &result) MDL_FINAL;

    /// Run this compiled lambda functions as a uniform function returning unsigned on the CPU.
    ///
    /// \param[out] result  the result will be written to
    ///
    /// \returns false if execution was aborted by runtime error, true otherwise
    ///
    /// Executes a compiled uniform function returning unsigned.
    bool run(unsigned &result) MDL_FINAL;

    /// Run this compiled lambda functions as a uniform function returning float on the CPU.
    ///
    /// \param[out] result  the result will be written to
    ///
    /// \returns false if execution was aborted by runtime error, true otherwise
    ///
    /// Executes a compiled uniform function returning float.
    bool run(float &result) MDL_FINAL;

    /// Run this compiled lambda functions as a uniform function returning float2 on the CPU.
    ///
    /// \param[out] result  the result will be written to
    ///
    /// \returns false if execution was aborted by runtime error, true otherwise
    ///
    /// Executes a compiled uniform function returning float2.
    bool run(Float2_struct &result) MDL_FINAL;

    /// Run this compiled lambda functions as a uniform function returning float3 on the CPU.
    ///
    /// \param[out] result  the result will be written to
    ///
    /// \returns false if execution was aborted by runtime error, true otherwise
    ///
    /// Executes a compiled uniform function returning float3.
    bool run(Float3_struct &result) MDL_FINAL;

    /// Run this compiled lambda functions as a uniform function returning float4 on the CPU.
    ///
    /// \param[out] result  the result will be written to
    ///
    /// \returns false if execution was aborted by runtime error, true otherwise
    ///
    /// Executes a compiled uniform function returning float4.
    bool run(Float4_struct &result) MDL_FINAL;

    /// Run this compiled lambda functions as a uniform function returning float3x3 on the CPU.
    ///
    /// \param[out] result  the result will be written to
    ///
    /// \returns false if execution was aborted by runtime error, true otherwise
    ///
    /// Executes a compiled uniform function returning float3x3.
    bool run(Matrix3x3_struct &result) MDL_FINAL;

    /// Run this compiled lambda functions as a uniform function returning float4x4 on the CPU.
    ///
    /// \param[out] result  the result will be written to
    ///
    /// \returns false if execution was aborted by runtime error, true otherwise
    ///
    /// Executes a compiled uniform function returning float4x4.
    bool run(Matrix4x4_struct &result) MDL_FINAL;

    /// Run this compiled lambda functions as a uniform function returning string on the CPU.
    ///
    /// \param[out] result  the result will be written to
    ///
    /// \returns false if execution was aborted by runtime error, true otherwise
    ///
    /// Executes a compiled uniform function returning string.
    ///
    /// \note It is not possible in MDL to dynamically create a string. Hence all possible
    ///       return values are statically known and embedded into the compiled code.
    ///       The returned pointer is valid as long as this compiled lambda function is not
    ///       destroyed.
    bool run(char const *&result) MDL_FINAL;

    /// Run a this compiled lambda switch function on the CPU.
    ///
    /// \param[in]  proj      the projection index of the lambda tuple to compute
    /// \param[out] result    the result will be written to
    /// \param[in]  state     the MDL state for the evaluation
    /// \param[in]  tex_data  extra thread data for the texture handler
    /// \param[in]  cap_args  the captured arguments block, if arguments were captured
    ///
    /// \returns false if execution was aborted by runtime error, true otherwise
    ///
    /// \note This is the typical entry point for varying functions. Attached to a material
    ///       the only possible return value is float3 or float, which is automatically converted
    ///       to a float3 by the compiled code.
    bool run_core(
        unsigned                     proj,
        Float3_struct                &result,
        Shading_state_material const *state,
        void                         *tex_data,
        void const                   *cap_args) MDL_FINAL;

    /// Run a this compiled lambda switch function on the CPU.
    ///
    /// \param[in]  index     the index of the function to execute
    /// \param[out] result    the result will be written to
    /// \param[in]  state     the core state
    /// \param[in]  tex_data  extra thread data for the texture handler
    /// \param[in]  cap_args  the captured arguments block, if arguments were captured
    ///
    /// \returns false if execution was aborted by runtime error, true otherwise
    ///
    /// \note This allows to execute any compiled function on the CPU. The result must be
    ///       big enough to take the functions result.
    ///       It can be used as an replacement to run_core() if this funciton is NOT a
    ///       switch function.
    bool run_generic(
        size_t                       index,
        void                         *result,
        Shading_state_material const *state,
        void                         *tex_data,
        void const                   *cap_args) MDL_FINAL;


    /// Run a compiled init function on the CPU. This may modify the texture results buffer
    /// of the given state.
    ///
    /// \param[in]  index     the index of the function to execute
    /// \param[in]  state     the core state
    /// \param[in]  tex_data  extra thread data for the texture handler
    /// \param[in]  cap_args  the captured arguments block, if arguments were captured
    ///
    /// \returns false if execution was aborted by runtime error, true otherwise
    bool run_init(
        size_t                 index,
        Shading_state_material *state,
        void                   *tex_data,
        void const             *cap_args) MDL_FINAL;

    /// Returns the index of the given resource for use as an parameter to a resource-related
    /// function in the generated CPU code.
    ///
    /// \param tag  the resource tag
    ///
    /// \returns the resource index or 0 if the resource is unknown.
    unsigned get_known_resource_index(unsigned tag) const MDL_FINAL;

    // -------------------- non-interface methods --------------------

    /// Get the LLVM context.
    llvm::LLVMContext &get_llvm_context() { return m_context; }

    /// Get the LLVM module.
    llvm::Module *get_llvm_module() { return m_module; }

    /// Set the LLVM module.
    ///
    /// \param module  the LLVM module
    void set_llvm_module(llvm::Module *module) { m_module = module; }

    /// Add an entry point of a JIT compiled function.
    ///
    /// \param address  the function address
    void add_entry_point(void *address);

    /// Set the Read-Only data segment.
    void set_ro_segment(char const *data, size_t size);

    /// Set the render state usage.
    void set_render_state_usage(IGenerated_code_lambda_function::State_usage usage)
    {
        m_render_state_usage = usage;
    }

    /// Add a captured arguments layout.
    void add_captured_arguments_layout(IGenerated_code_value_layout const *layout)
    {
        m_captured_arguments_layouts.push_back(mi::base::make_handle_dup(layout));
    }

    /// Add a mapped string.
    ///
    /// \param s   the string constant
    /// \param id  the assigned id for this constant
    void add_mapped_string(char const *s, size_t id);

private:
    /// Register a new non-texture resource tag.
    ///
    /// \param tag   the tag
    /// \param type  the type of this resource
    size_t register_resource_tag(
        unsigned tag,
        IType_resource const *type);

    /// Register a new texture resource tag.
    ///
    /// \param tag         the texture tag
    /// \param gamma_mode  the MDl gamma mode
    /// \param tex_type    the type of this texture
    size_t register_texture_tag(
        unsigned                   tag,
        IType_texture const        *tex_type,
        IValue_texture::gamma_mode gamma_mode);

    /// Register a new string.
    ///
    /// \param s   the string
    size_t register_string(
        char const *s);

private:
    /// Constructor.
    ///
    /// \param jitted_code  The jitted_code.
    Generated_code_lambda_function(Jitted_code *jitted_code);

    /// Destructor.
    ~Generated_code_lambda_function() MDL_OVERRIDE;

    // no copy constructor.
    Generated_code_lambda_function(
        Generated_code_lambda_function const &) MDL_DELETED_FUNCTION;

    // no assignment operator
    Generated_code_lambda_function const &operator=(
        Generated_code_lambda_function const &) MDL_DELETED_FUNCTION;

private:
    /// The jitted code singleton.
    mi::base::Handle<Jitted_code> m_jitted_code;

    /// The LLVM context of the LLVM module.
    llvm::LLVMContext m_context;

    /// The LLVM module of the environment function.
    llvm::Module      *m_module;

    typedef void (Jitted_func)();

    /// The signature of the JIT compiled environment function.
    ///
    /// \param result         the result color will be written here
    /// \param state          the MDL state
    /// \param res_data_pair  the resource data helper object, shared and thread parts
    /// \param exc_state      the exception state helper
    /// \param cap_args       the captured arguments block, should be NULL for environment
    ///                       functions
    typedef void (Env_func)(
        RGB_color                       *result,
        Shading_state_environment const *state,
        Res_data_pair const             &res_data_pair,
        LLVM_code_generator::Exc_state  &exc_state,
        void const                      *cap_args);

    /// The signature of the JIT compiled lambda function returning a boolean.
    ///
    /// \param result         the result will be written here
    /// \param res_data_pair  the resource data helper object, shared and thread parts
    /// \param exc_state      the exception state helper
    /// \param cap_args       the captured arguments block, should be NULL for const functions
    typedef void (Lambda_func_bool)(
        bool                           &result,
        Res_data_pair const            &res_data_pair,
        LLVM_code_generator::Exc_state &exc_state,
        void const                     *cap_args);

    /// The signature of the JIT compiled lambda function returning an int.
    ///
    /// \param result         the result will be written here
    /// \param res_data_pair  the resource data helper object, shared and thread parts
    /// \param exc_state      the exception state helper
    /// \param cap_args       the captured arguments block, should be NULL for const functions
    typedef void (Lambda_func_int)(
        int                            &result,
        Res_data_pair const            &res_data_pair,
        LLVM_code_generator::Exc_state &exc_state,
        void const                     *cap_args);

    /// The signature of the JIT compiled lambda function returning an unsigned.
    ///
    /// \param result         the result will be written here
    /// \param res_data_pair  the resource data helper object, shared and thread parts
    /// \param exc_state      the exception state helper
    /// \param cap_args       the captured arguments block, should be NULL for const functions
    typedef void (Lambda_func_unsigned)(
        unsigned                       &result,
        Res_data_pair const            &res_data_pair,
        LLVM_code_generator::Exc_state &exc_state,
        void const                     *cap_args);

    /// The signature of the JIT compiled lambda function returning a float.
    ///
    /// \param result         the result will be written here
    /// \param res_data_pair  the resource data helper object, shared and thread parts
    /// \param exc_state      the exception state helper
    /// \param cap_args       the captured arguments block, should be NULL for const functions
    typedef void (Lambda_func_float)(
        float                          &result,
        Res_data_pair const            &res_data_pair,
        LLVM_code_generator::Exc_state &exc_state,
        void const                     *cap_args);

    /// The signature of the JIT compiled lambda function returning a float2 vector.
    ///
    /// \param result         the result will be written here
    /// \param res_data_pair  the resource data helper object, shared and thread parts
    /// \param exc_state      the exception state helper
    /// \param cap_args       the captured arguments block, should be NULL for const functions
    typedef void (Lambda_func_float2)(
        Float2_struct                  &result,
        Res_data_pair const            &res_data_pair,
        LLVM_code_generator::Exc_state &exc_state,
        void const                     *cap_args);

    /// The signature of the JIT compiled lambda function returning a float3 vector.
    ///
    /// \param result        the result will be written here
    /// \param res_data_pair  the resource data helper object, shared and thread parts
    /// \param exc_state     the exception state helper
    /// \param cap_args       the captured arguments block, should be NULL for const functions
    typedef void (Lambda_func_float3)(
        Float3_struct                  &result,
        Res_data_pair const            &res_data_pair,
        LLVM_code_generator::Exc_state &exc_state,
        void const                     *cap_args);

    /// The signature of the JIT compiled lambda function returning a float4 vector.
    ///
    /// \param result         the result will be written here
    /// \param res_data_pair  the resource data helper object, shared and thread parts
    /// \param exc_state      the exception state helper
    /// \param cap_args       the captured arguments block, should be NULL for const functions
    typedef void (Lambda_func_float4)(
        Float4_struct                  &result,
        Res_data_pair const            &res_data_pair,
        LLVM_code_generator::Exc_state &exc_state,
        void const                     *cap_args);

    /// The signature of the JIT compiled lambda function returning a float3x3 matrix.
    ///
    /// \param result         the result will be written here
    /// \param res_data_pair  the resource data helper object, shared and thread parts
    /// \param exc_state      the exception state helper
    /// \param cap_args       the captured arguments block, should be NULL for const functions
    typedef void (Lambda_func_float3x3)(
        Matrix3x3_struct               &result,
        Res_data_pair const            &res_data_pair,
        LLVM_code_generator::Exc_state &exc_state,
        void const                     *cap_args);

    /// The signature of the JIT compiled lambda function returning a float4x4 matrix.
    ///
    /// \param result         the result will be written here
    /// \param res_data_pair  the resource data helper object, shared and thread parts
    /// \param exc_state      the exception state helper
    /// \param cap_args       the captured arguments block, should be NULL for const functions
    typedef void (Lambda_func_float4x4)(
        Matrix4x4_struct               &result,
        Res_data_pair const            &res_data_pair,
        LLVM_code_generator::Exc_state &exc_state,
        void const                     *cap_args);

    /// The signature of the JIT compiled lambda function returning a string.
    ///
    /// \param result         the result will be written here
    /// \param res_data_pair  the resource data helper object, shared and thread parts
    /// \param exc_state      the exception state helper
    /// \param cap_args       the captured arguments block, should be NULL for const functions
    typedef char const *(Lambda_func_string)(
        char const *                   &result,
        Res_data_pair const            &res_data_pair,
        LLVM_code_generator::Exc_state &exc_state,
        void const                     *cap_args);

    /// The signature of the JIT compiled lambda function.
    ///
    /// \param[in]  state          the core state
    /// \param[in]  res_data_pair  the resource data helper object, shared and thread parts
    /// \param[in]  exc_state      the exception state helper
    /// \param[in]  cap_args       the captured arguments block, if arguments were captured
    /// \param[out] result         the result
    /// \param[in]  proj           the projection index of the tuple to compute
    typedef bool (Core_func)(
        Shading_state_material const   *state,
        Res_data_pair const            &res_data_pair,
        LLVM_code_generator::Exc_state &exc_state,
        void const                     *cap_args,
        Float3_struct                  &result,
        unsigned                       proj);

    /// The generic signature of the JIT compiled lambda function.
    ///
    /// \param[out] result         the result
    /// \param[in]  state          the core state
    /// \param[in]  res_data_pair  the resource data helper object, shared and thread parts
    /// \param[in]  exc_state      the exception state helper
    /// \param[in]  cap_args       the captured arguments block, if arguments were captured
    typedef void (Gen_func)(
        void                           *result,
        Shading_state_material const   *state,
        Res_data_pair const            &res_data_pair,
        LLVM_code_generator::Exc_state &exc_state,
        void const                     *cap_args);

    /// The signature of a JIT compiled BSDF init function.
    ///
    /// \param[inout] state          the core state
    /// \param[in]    res_data_pair  the resource data helper object, shared and thread parts
    /// \param[in]    exc_state      the exception state helper
    /// \param[in]    cap_args       the captured arguments block, if arguments were captured
    typedef void (Init_func)(
        Shading_state_material         *state,
        Res_data_pair const            &res_data_pair,
        LLVM_code_generator::Exc_state &exc_state,
        void const                     *cap_args);

    /// The list of JIT compiled functions.
    mi::mdl::vector<Jitted_func *>::Type m_jitted_funcs;

    /// Collected resource entries used for the IResource_handler interface
    mi::mdl::vector<Resource_entry>::Type m_res_entries;

    /// Collected string entries used for the IResource_handler interface
    mi::mdl::vector<mi::mdl::string>::Type m_string_entries;

    /// The Messages.
    Messages_impl m_messages;

    /// The resource helper objects for this function.
    Res_data m_res_data;

    /// The exception handler if any.
    IMDL_exception_handler *m_exc_handler;

    /// If set non-zero, this function will not be executed anymore.
    mi::base::Atom32        m_aborted;

    /// The RO data segment if any.
    char *m_ro_segment;

    /// The length of the RO segment.
    size_t m_ro_length;

    /// The potential render state usage of the generated code.
    IGenerated_code_lambda_function::State_usage m_render_state_usage;

    typedef vector<mi::base::Handle<IGenerated_code_value_layout const> >::Type Layout_vec;

    /// The list of captured arguments block layouts.
    Layout_vec m_captured_arguments_layouts;

    typedef vector<string>::Type Mappend_string_vector;

    /// The mapped strings
    Mappend_string_vector m_mappend_strings;
};


}  // mdl
}  // mi

#endif // MDL_GENERATOR_JIT_GENERATED_CODE
