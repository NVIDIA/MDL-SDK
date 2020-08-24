/******************************************************************************
 * Copyright (c) 2012-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_GENERATOR_JIT_H
#define MDL_GENERATOR_JIT_H 1

#include <mdl/compiler/compilercore/compilercore_cc_conf.h>
#include <mdl/compiler/compilercore/compilercore_allocator.h>
#include <mdl/codegenerators/generator_code/generator_code.h>

#include "generator_jit_type_map.h"
#include "generator_jit_llvm.h"

namespace mi {
namespace mdl {

class MDL;
class IModule;
class Jitted_code;

///
/// Implementation of the Link unit for the JIT code generator
///
class Link_unit_jit : public Allocator_interface_implement<ILink_unit>
{
    typedef Allocator_interface_implement<ILink_unit> Base;
    friend class Allocator_builder;
public:
    /// Possible targets for the generated code.
    enum Target_kind {
        TK_PTX,      ///< Generate CUDA PTX code.
        TK_LLVM_IR,  ///< Generate LLVM IR (LLVM 7.0 compatible)
        TK_NATIVE,   ///< Generate native code
        TK_HLSL      ///< Generate HLSL code.
    };

    typedef Type_mapper::Type_mapping_mode Type_mapping_mode;

    /// Add a lambda function to this link unit.
    ///
    /// \param lambda               the lambda function to compile
    /// \param name_resolver        the call name resolver
    /// \param kind                 the kind of the lambda function
    /// \param arg_block_index      on success, this parameter will receive the index of the target
    ///                             argument block used for added entity or ~0 if none is used
    /// \param function_index       the index of the callable function in the created target code.
    ///                             This parameter is option, provide NULL if not required.
    ///
    /// \return true on success
    bool add(
        ILambda_function const                    *lambda,
        ICall_name_resolver const                 *name_resolver,
        IGenerated_code_executable::Function_kind  kind,
        size_t                                    *arg_block_index,
        size_t                                    *function_index) MDL_FINAL;

    /// Add a distribution function to this link unit.
    ///
    /// Currently only BSDFs are supported.
    /// For a BSDF it results in four functions, with their names built from the name of the
    /// main DF function of \p dist_func suffixed with \c "_init", \c "_sample", \c "_evaluate"
    /// and \c "_pdf", respectively.
    ///
    /// \param dist_func            the distribution function to compile
    /// \param name_resolver        the call name resolver
    /// \param arg_block_index      this variable will receive the index of the target argument
    ///                             block used for this distribution function or ~0 if none is used
    /// \param function_index       the index of the callable function in the created target code.
    ///                             This parameter is option, provide NULL if not required.
    ///
    /// \return true on success
    bool add(
        IDistribution_function const *dist_func,
        ICall_name_resolver const    *name_resolver,
        size_t                       *arg_block_index,
        size_t                       *function_index) MDL_FINAL;

    /// Get the number of functions in this link unit.
    size_t get_function_count() const MDL_FINAL;

    /// Get the name of the i'th function inside this link unit.
    ///
    /// \param i  the index of the function
    ///
    /// \return the name of the i'th function or NULL if the index is out of bounds
    char const *get_function_name(size_t i) const MDL_FINAL;

    /// Returns the distribution kind of the i'th function inside this link unit.
    ///
    /// \param i  the index of the function
    ///
    /// \return The distribution kind of the i'th function or \c FK_INVALID if \p i was invalid.
    IGenerated_code_executable::Distribution_kind get_distribution_kind(size_t i) const MDL_FINAL;

    /// Returns the function kind of the i'th function inside this link unit.
    ///
    /// \param i  the index of the function
    ///
    /// \return The function kind of the i'th function or \c FK_INVALID if \p i was invalid.
    IGenerated_code_executable::Function_kind get_function_kind(size_t i) const MDL_FINAL;

    /// Get the index of the target argument block layout for the i'th function inside this link
    /// unit if used.
    ///
    /// \param i  the index of the function
    ///
    /// \return The index of the target argument block layout or ~0 if not used or \p i is invalid.
    size_t get_function_arg_block_layout_index(size_t i) const MDL_FINAL;

    /// Get the number of target argument block layouts used by this link unit.
    size_t get_arg_block_layout_count() const MDL_FINAL;

    /// Get the i'th target argument block layout used by this link unit.
    ///
    /// \param i  the index of the target argument block layout
    ///
    /// \return The target argument block layout or \c NULL if \p i is invalid.
    IGenerated_code_value_layout const *get_arg_block_layout(size_t i) const MDL_FINAL;

    /// Access messages.
    Messages const &access_messages() const MDL_FINAL;

    /// Returns the prototype of the i'th function inside this link unit.
    ///
    /// \param index   the index of the function.
    /// \param lang    the language to use for the prototype.
    ///
    /// \return The prototype or NULL if \p index is out of bounds or \p lang cannot be used
    ///         for this target code.
    const char* get_function_prototype(
        size_t index,
        IGenerated_code_executable::Prototype_language lang) const MDL_FINAL;

    /// Get the target kind.
    Target_kind get_target_kind() const { return m_target_kind; }

    /// Get the LLVM module.
    llvm::Module const *get_llvm_module() const;

    /// The arrow operator accesses the code generator.
    LLVM_code_generator *operator->() const {
        return &m_code_gen;
    }

    /// Get the LLVM function of the i'th function inside this link unit.
    ///
    /// \param i  the index of the function
    ///
    /// \return The LLVM function.
    llvm::Function *get_function(size_t i) const;

    /// Get the code object of this link unit.
    IGenerated_code_executable *get_code_object() const {
        m_code->retain();
        return m_code.get();
    }

    /// Get write access to the messages of the generated code.
    Messages_impl &access_messages();

    typedef vector<Resource_tag_tuple>::Type Resource_tag_map;

    /// Get the resource tag map of this unit.
    Resource_tag_map const *get_resource_tag_map() const { return &m_resource_tag_map; }

private:
    /// Constructor.
    ///
    /// \param alloc                the allocator
    /// \param jitted_code          the jitted code singleton
    /// \param compiler             the MDL compiler
    /// \param target_kind          the kind of targeted code
    /// \param tm_mode              if target is not PTX, the type mapping mode
    /// \param sm_version           if target is PTX, the SM_version we compile for
    /// \param num_texture_spaces   the number of supported texture spaces
    /// \param num_texture_results  the number of texture result entries
    /// \param options              the backend options
    /// \param state_mapping        how to map the MDL state
    /// \param enable_debug         if true, generate debug info
    Link_unit_jit(
        IAllocator         *alloc,
        Jitted_code        *jitted_code,
        MDL                *compiler,
        Target_kind        target_kind,
        Type_mapping_mode  tm_mode,
        unsigned           sm_version,
        unsigned           num_texture_spaces,
        unsigned           num_texture_results,
        Options_impl const *options,
        unsigned           state_mapping,
        bool               enable_debug);

    // non copyable
    Link_unit_jit(Link_unit_jit const &) MDL_DELETED_FUNCTION;
    Link_unit_jit &operator=(Link_unit_jit const &) MDL_DELETED_FUNCTION;

    /// Destructor.
    ~Link_unit_jit();

    /// Creates the code object to be used with this link unit.
    ///
    /// \param jitted_code  the jitted code object required for native targets
    IGenerated_code_executable *create_code_object(
        Jitted_code *jitted_code);

    /// Creates the resource manager to be used with this link unit.
    ///
    /// \param icode  the generated code object
    IResource_manager *create_resource_manager(
        IGenerated_code_executable *icode);

    /// Update the resource attribute maps for the current lambda function to be compiled.
    ///
    /// \param lambda  the current lambda function to be compiled
    void update_resource_attribute_map(
        Lambda_function const *lambda);

    /// Update the resource to tag map for the current lambda function to be compiled.
    ///
    /// \param lambda  the current lambda function to be compiled
    void update_resource_tag_map(
        Lambda_function const *lambda);

    /// Find the assigned tag for a resource in the resource map.
    ///
    /// \param kind     the kind of the resource
    /// \param url      the url of the resource
    int find_resource_tag(
        Resource_tag_tuple::Kind kind,
        char const               *url) const;

    /// Add a new entry in the resource to tag map.
    ///
    /// \param kind     the kind of the resource
    /// \param url      the url of the resource
    /// \param tag      the assigned tag
    void add_resource_tag_mapping(
        Resource_tag_tuple::Kind kind,
        char const               *url,
        int                      tag);

    /// Get the LLVM context to use with this link unit.
    llvm::LLVMContext *get_llvm_context();

private:
    /// Memory arena for storing strings.
    Memory_arena m_arena;

    /// The kind of targeted code.
    Target_kind m_target_kind;

    /// The used LLVM context for source-only targets.
    llvm::LLVMContext m_source_only_llvm_context;

    /// The code object which will contain the result.
    /// For native JIT, this will also contain the used LLVM context.
    mi::base::Handle<IGenerated_code_executable> m_code;

    /// The code generator.
    mutable LLVM_code_generator m_code_gen;

    /// The resource manager for the unit.
    IResource_manager *m_res_manag;

    typedef vector<mi::base::Handle<mi::mdl::IGenerated_code_value_layout> >::Type Layout_vec;

    /// The target argument block layouts used by the functions inside the link unit.
    Layout_vec m_arg_block_layouts;

    /// The added lambda functions.
    /// Must be held to avoid invalid entries in the context data map of m_code_gen.
    vector<mi::base::Handle<ILambda_function const> >::Type m_lambdas;

    /// The added distribution functions.
    /// Must be held to avoid invalid entries in the context data map of m_code_gen.
    vector<mi::base::Handle<IDistribution_function const> >::Type m_dist_funcs;

    /// The resource to tag map for this link unit, mapping resource values to tags.
    Resource_tag_map m_resource_tag_map;
};

///
/// Implementation of the code generator for executable code.
///
class Code_generator_jit : public Code_generator<ICode_generator_jit>
{
    typedef Code_generator<ICode_generator_jit> Base;
    friend class Allocator_builder;
public:
    /// Creates a JIT code generator.
    ///
    /// \param alloc        The allocator.
    /// \param mdl          The compiler.
    static Code_generator_jit *create_code_generator(
        IAllocator *alloc,
        MDL        *mdl);

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

    /// Get the name of the target language.
    char const *get_target_language() const MDL_FINAL;

    /// Compile a whole module.
    ///
    /// \param module  The module to compile.
    /// \param mode    The compilation mode
    ///
    /// \note This method is not used currently for code generation, just
    ///       by the unit tests to test various aspects of the code generator.
    ///
    /// \returns The generated code.
    IGenerated_code_executable *compile(
        IModule const    *module,
        Compilation_mode mode) MDL_FINAL;

    /// Compile a lambda function using the JIT into an environment (shader) of a scene.
    ///
    /// \param lambda         the lambda function to compile
    /// \param name_resolver  the call name resolver
    ///
    /// \return the compiled function or NULL on compilation errors
    IGenerated_code_lambda_function *compile_into_environment(
        ILambda_function const    *lambda,
        ICall_name_resolver const *name_resolver) MDL_FINAL;

    /// Compile a lambda function using the JIT into a constant function.
    ///
    /// \param lambda           the lambda function to compile
    /// \param name_resolver    the call name resolver
    /// \param attr             an interface to retrieve resource attributes
    /// \param world_to_object  the world-to-object transformation matrix for this function
    /// \param object_to_world  the object-to_world transformation matrix for this function
    /// \param object_id        the result of state::object_id() for this function
    ///
    /// \return the compiled function or NULL on compilation errors
    IGenerated_code_lambda_function *compile_into_const_function(
        ILambda_function const    *lambda,
        ICall_name_resolver const  *name_resolver,
        ILambda_resource_attribute *attr,
        Float4_struct const        world_to_object[4],
        Float4_struct const        object_to_world[4],
        int                        object_id) MDL_FINAL;

    /// Compile a lambda switch function having several roots using the JIT into a
    /// function computing one of the root expressions.
    ///
    /// \param lambda               the lambda function to compile
    /// \param name_resolver        the call name resolver
    /// \param num_texture_spaces   the number of supported texture spaces
    /// \param num_texture_results  the number of texture result entries
    ///
    /// \return the compiled function or NULL on compilation errors
    IGenerated_code_lambda_function *compile_into_switch_function(
        ILambda_function const    *lambda,
        ICall_name_resolver const *name_resolver,
        unsigned                  num_texture_spaces,
        unsigned                  num_texture_results) MDL_FINAL;

    /// Compile a lambda switch function having several roots using the JIT into a
    /// function computing one of the root expressions for execution on the GPU.
    ///
    /// \param code_cache           If non-NULL, a code cache
    /// \param lambda               the lambda function to compile
    /// \param name_resolver        the call name resolver
    /// \param num_texture_spaces   the number of supported texture spaces
    /// \param num_texture_results  the number of texture result entries
    /// \param sm_version           the target architecture of the GPU
    ///
    /// \return the compiled function or NULL on compilation errors
    IGenerated_code_executable *compile_into_switch_function_for_gpu(
        ICode_cache               *code_cache,
        ILambda_function const    *lambda,
        ICall_name_resolver const *name_resolver,
        unsigned                  num_texture_spaces,
        unsigned                  num_texture_results,
        unsigned                  sm_version) MDL_FINAL;

    /// Compile a lambda function into a generic function using the JIT.
    ///
    /// \param lambda               the lambda function to compile
    /// \param name_resolver        the call name resolver
    /// \param num_texture_spaces   the number of supported texture spaces
    /// \param num_texture_results  the number of texture result entries
    /// \param transformer          an optional transformer for calls in the lambda expression
    ///
    /// \return the compiled function or NULL on compilation errors
    ///
    /// \note the lambda function must have only one root expression.
    IGenerated_code_lambda_function *compile_into_generic_function(
        ILambda_function const    *lambda,
        ICall_name_resolver const *name_resolver,
        unsigned                  num_texture_spaces,
        unsigned                  num_texture_results,
        ILambda_call_transformer  *transformer) MDL_FINAL;

    /// Compile a lambda function into a LLVM-IR using the JIT.
    ///
    /// \param lambda               the lambda function to compile
    /// \param name_resolver        the call name resolver
    /// \param num_texture_spaces   the number of supported texture spaces
    /// \param num_texture_results  the number of texture result entries
    /// \param enable_simd          if true, SIMD instructions will be generated
    ///
    /// \return the compiled function or NULL on compilation errors
    IGenerated_code_executable *compile_into_llvm_ir(
        ILambda_function const    *lambda,
        ICall_name_resolver const *name_resolver,
        unsigned                  num_texture_spaces,
        unsigned                  num_texture_results,
        bool                      enable_simd) MDL_FINAL;

    /// Fill a code object from a code cache entry.
    ///
    /// \param code   the code object to fill
    /// \param entry  the code cache entry
    void fill_code_from_cache(Generated_code_source *code, ICode_cache::Entry const *entry);

    /// Enter a code object into the code cache.
    ///
    /// \param code        the code object
    /// \param code_cache  the code cache where a new entry shall be inserted
    /// \param cache_key   the key to use when entering the code object into the cache
    void enter_code_into_cache(
        Generated_code_source *code,
        ICode_cache *code_cache,
        unsigned char const cache_key[16]);

    /// Compile a lambda function into PTX or HLSL using the JIT.
    ///
    /// The generated function will have the signature #mi::mdl::Lambda_generic_function or
    /// #mi::mdl::Lambda_switch_function depending on the type of the lambda.
    ///
    /// \param code_cache           If non-NULL, a code cache
    /// \param lambda               the lambda function to compile
    /// \param name_resolver        the call name resolver
    /// \param num_texture_spaces   the number of supported texture spaces
    /// \param num_texture_results  the number of texture result entries
    /// \param sm_version           the target architecture of the GPU
    /// \param comp_mode            the compilation mode deciding the target language,
    ///                             must be PTX or HLSL
    /// \param llvm_ir_output       if true generate LLVM-IR (prepared for the target language)
    ///
    /// \return the compiled function or NULL on compilation errors
    virtual IGenerated_code_executable *compile_into_source(
        ICode_cache               *code_cache,
        ILambda_function const    *lambda,
        ICall_name_resolver const *name_resolver,
        unsigned                  num_texture_spaces,
        unsigned                  num_texture_results,
        unsigned                  sm_version,
        Compilation_mode          comp_mode,
        bool                      llvm_ir_output) MDL_FINAL;

    /// Compile a distribution function into native code using the JIT.
    ///
    /// Currently only BSDFs are supported.
    /// For a BSDF, it results in four functions, with their names built from the name of the
    /// main DF function of \p dist_func suffixed with \c "_init", \c "_sample", \c "_evaluate"
    /// and \c "_pdf", respectively.
    ///
    /// \param dist_func            the distribution function to compile
    /// \param name_resolver        the call name resolver
    /// \param num_texture_spaces   the number of supported texture spaces
    /// \param num_texture_results  the number of texture result entries
    ///
    /// \return the compiled distribution function or NULL on compilation errors
    IGenerated_code_executable *compile_distribution_function_cpu(
        IDistribution_function const *dist_func,
        ICall_name_resolver const    *name_resolver,
        unsigned                     num_texture_spaces,
        unsigned                     num_texture_results) MDL_FINAL;

    /// Compile a distribution function into PTX or HLSL using the JIT.
    ///
    /// Currently only BSDFs are supported.
    /// For a BSDF, it results in four functions, with their names built from the name of the
    /// main DF function of \p dist_func suffixed with \c "_init", \c "_sample", \c "_evaluate"
    /// and \c "_pdf", respectively.
    ///
    /// \param dist_func            the distribution function to compile
    /// \param name_resolver        the call name resolver
    /// \param num_texture_spaces   the number of supported texture spaces
    /// \param num_texture_results  the number of texture result entries
    /// \param sm_version           the target architecture of the GPU
    /// \param comp_mode            the compilation mode deciding the target language,
    ///                             must be PTX or HLSL
    /// \param llvm_ir_output       if true generate LLVM-IR (prepared for the target language)
    ///
    /// \return the compiled distribution function or NULL on compilation errors
    IGenerated_code_executable *compile_distribution_function_gpu(
        IDistribution_function const *dist_func,
        ICall_name_resolver const    *name_resolver,
        unsigned                     num_texture_spaces,
        unsigned                     num_texture_results,
        unsigned                     sm_version,
        Compilation_mode             comp_mode,
        bool                         llvm_ir_output) MDL_FINAL;

    /// Get the device library for PTX compilation.
    ///
    /// \param[out] size        the size of the library
    ///
    /// \return the library as LLVM bitcode representation
    unsigned char const *get_libdevice_for_gpu(
        size_t   &size) MDL_FINAL;
    
    /// Get the resolution of the libbsdf multi-scattering lookup table data.
    ///
    /// \param bsdf_data_kind   the kind of the BSDF data, has to be a multiscatter kind
    /// \param out_theta        will contain the number of theta values when data is available
    /// \param out_roughness    will contain the number of roughness values when data is available
    /// \param out_ior          will contain the number of IOR values when data is available
    /// \returns                true if there is data for this semantic (BSDF)
    bool get_libbsdf_multiscatter_data_resolution(
        IValue_texture::Bsdf_data_kind bsdf_data_kind,
        size_t &out_theta,
        size_t &out_roughness,
        size_t &out_ior) const MDL_FINAL;

    /// Get access to the libbsdf multi-scattering lookup table data.
    ///
    /// \param bsdf_data_kind  the kind of the BSDF data, has to be a multiscatter kind
    /// \param[out] size       the size of the data
    ///
    /// \returns               the lookup data if available for this semantic (BSDF), NULL otherwise
    unsigned char const *get_libbsdf_multiscatter_data(
        IValue_texture::Bsdf_data_kind bsdf_data_kind,
        size_t                         &size) const MDL_FINAL;

    /// Create a link unit.
    ///
    /// \param mode                 the compilation mode
    /// \param enable_simd          if LLVM-IR is targeted, specifies whether to use SIMD
    ///                             instructions
    /// \param sm_version           if PTX is targeted, the SM version we compile for
    /// \param num_texture_spaces   the number of supported texture spaces
    /// \param num_texture_results  the number of texture result entries
    ///
    /// \return  a new empty link unit.
    Link_unit_jit *create_link_unit(
        Compilation_mode  mode,
        bool              enable_simd,
        unsigned          sm_version,
        unsigned          num_texture_spaces,
        unsigned          num_texture_results) MDL_FINAL;

    /// Compile a link unit into a LLVM-IR, PTX or native code using the JIT.
    ///
    /// \param unit  the link unit to compile
    ///
    /// \return the compiled function or NULL on compilation errors
    IGenerated_code_executable *compile_unit(
        ILink_unit const *unit) MDL_FINAL;

    /// Create a blank layout used for deserialization of target codes.
    IGenerated_code_value_layout* create_value_layout() const MDL_FINAL;

private:
    /// Calculate the state mapping mode from options.
    unsigned get_state_mapping() const;

private:
    /// Constructor.
    ///
    /// \param alloc        The allocator.
    /// \param mdl          The compiler.
    /// \param jitted_code  The jitted code singleton.
    explicit Code_generator_jit(
        IAllocator  *alloc,
        MDL         *mdl,
        Jitted_code *jitted_code);

private:
    /// The builder for objects.
    mutable Allocator_builder m_builder;

    /// The jitted code.
    mi::base::Handle<Jitted_code> m_jitted_code;
};

}  // mdl
}  // mi

#endif // MDL_GENERATOR_JIT_H
