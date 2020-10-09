/******************************************************************************
 * Copyright (c) 2013-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_GENERATOR_DAG_LAMBDA_FUNCTION_H
#define MDL_GENERATOR_DAG_LAMBDA_FUNCTION_H 1

#include <mi/base/handle.h>
#include <mi/base/atom.h>

#include <mi/mdl/mdl_code_generators.h>

#include "mdl/compiler/compilercore/compilercore_cc_conf.h"
#include "mdl/compiler/compilercore/compilercore_memory_arena.h"
#include "mdl/compiler/compilercore/compilercore_factories.h"
#include "mdl/compiler/compilercore/compilercore_mdl.h"
#include "mdl/compiler/compilercore/compilercore_cstring_hash.h"

#include "generator_dag_derivatives.h"
#include "generator_dag_ir.h"

namespace mi {
namespace mdl {

class Derivative_infos;
class Function_context;
class IMDL;
class IValue_resource;

/// A value entry in the resource attribute map.
struct Resource_attr_entry {
    size_t index;               ///< The "index" value of this resource.
    bool valid;                 ///< True if this resource is valid.
    union {
        struct {
            unsigned width;              ///< texture width
            unsigned height;             ///< texture height
            unsigned depth;              ///< texture depth
            IType_texture::Shape shape;  ///< texture shape
        } tex;
        struct {
            float power;        ///< light profile power
            float maximum;      ///< light profile maximum
        } lp;
    } u;
};

struct Resource_hasher {
    size_t operator()(Resource_tag_tuple const &p) const {
        cstring_hash cstring_hasher;

        return size_t(p.m_kind) ^ cstring_hasher(p.m_url) ^ p.m_tag;
    }
};

struct Resource_equal_to {
    bool operator()(Resource_tag_tuple const &a, Resource_tag_tuple const &b) const {
        if (a.m_kind != b.m_kind)
            return false;
        if (a.m_tag != b.m_tag)
            return false;

        cstring_equal_to cstring_cmp;

        return cstring_cmp(a.m_url, b.m_url);
    }
};

typedef hash_map<Resource_tag_tuple, Resource_attr_entry, Resource_hasher, Resource_equal_to>::Type
    Resource_attr_map;

/// This class handles the creation and compilation of lambda functions.
///
/// The body (expression) of this lambda function is expression as a DAG to simplify the
/// reuse of DAG components.
/// Once the expression is build, in can be compiled into a function.
class Lambda_function : public Allocator_interface_implement<ILambda_function>
{
    typedef Allocator_interface_implement<ILambda_function> Base;
    friend class Allocator_builder;

public:
    /// Get the type factory of this function.
    Type_factory *get_type_factory() MDL_FINAL;

    /// Get the value factory of this function.
    Value_factory *get_value_factory() MDL_FINAL;

    /// Create a constant.
    /// \param  value       The value of the constant.
    /// \returns            The created constant.
    ///
    /// \note Use this method to create arguments of the instance.
    DAG_constant const *create_constant(IValue const *value) MDL_FINAL;

    /// Create a call.
    /// \param  name            The absolute name of the called function.
    /// \param  sema            The semantic of the called function.
    /// \param  call_args       The call arguments of the called function.
    /// \param  num_call_args   The number of call arguments.
    /// \param  ret_type        The return type of the called function.
    /// \returns                The created call.
    ///
    /// \note Use this method to create arguments of the instance.
    DAG_node const *create_call(
        char const                    *name,
        IDefinition::Semantics        sema,
        DAG_call::Call_argument const call_args[],
        int                           num_call_args,
        mi::mdl::IType const          *ret_type) MDL_FINAL;

    /// Create a parameter reference.
    /// \param  type        The type of the parameter
    /// \param  index       The index of the parameter.
    /// \returns            The created parameter reference.
    DAG_parameter const *create_parameter(
        IType const *type,
        int         index) MDL_FINAL;

    /// Get the body of this function.
    ///
    /// \return The body expression or NULL if this is a switch function.
    DAG_node const *get_body() const MDL_FINAL;

    /// Set the body of this function.
    ///
    /// \param expr   the body expression
    void set_body(DAG_node const *expr) MDL_FINAL;

    /// Import (i.e. deep-copy) a DAG expression into this lambda function.
    ///
    /// \param expr  the DAG expression to import
    ///
    /// \return the imported expression
    DAG_node const *import_expr(DAG_node const *expr) MDL_FINAL;

    /// Store a DAG (root) expression and returns an index for it.
    ///
    /// \param expr  the expression to remember, must be owned by this lambda function
    ///
    /// \return the index of this expression
    ///
    /// \note The same index will be assigned to identical (in the sense of CSE) expressions.
    size_t store_root_expr(DAG_node const *expr) MDL_FINAL;

    /// Remove a root expression.
    ///
    /// \param idx  the index of the root expression to be removed
    ///
    /// \return true on success, false if idx is invalid
    ///
    /// \note The freed index can be reused.
    bool remove_root_expr(size_t idx) MDL_FINAL;

    /// Run garbage collection AFTER a root expression was removed.
    ///
    /// \returns a cleaned copy or NULL if all was deleted
    Lambda_function *garbage_collection() MDL_FINAL;

    /// Get the remembered expression for a given index.
    ///
    /// \param idx  the index of the root expression
    DAG_node const *get_root_expr(size_t idx) const MDL_FINAL;

    /// Get the number of root expressions.
    size_t get_root_expr_count() const MDL_FINAL;

    /// Enumerate all used texture resources of this lambda function.
    ///
    /// \param resolver    a call name resolver
    /// \param enumerator  the enumerator interface
    /// \param root        if non-NULL, the root expression to enumerate, else enumerate
    ///                    all roots of a switch function
    void enumerate_resources(
        ICall_name_resolver const   &resolver,
        ILambda_resource_enumerator &enumerator,
        DAG_node const              *root = NULL) const MDL_FINAL;

    /// Register a texture resource mapping.
    ///
    /// \param res_kind        the kind of the resource (texture or invalid reference)
    /// \param res_url         the URL of the texture resource if any
    /// \param gamma           the gamma mode of this resource
    /// \param bsdf_data_kind  the kind of BSDF data in case of BSDF data textures
    /// \param shape           the shape of this resource
    /// \param res_tag         the tag of the texture resource
    /// \param idx             the mapped index value representing the resource in a lookup table
    /// \param valid           true if this is a valid resource, false otherwise
    /// \param width           the width of the texture
    /// \param height          the height of the texture
    /// \param depth           the depth of the texture
    void map_tex_resource(
        IValue::Kind                   res_kind,
        char const                     *res_url,
        IValue_texture::gamma_mode     gamma,
        IValue_texture::Bsdf_data_kind bsdf_data_kind,
        IType_texture::Shape           shape,
        int                            res_tag,
        size_t                         idx,
        bool                           valid,
        int                            width,
        int                            height,
        int                            depth) MDL_FINAL;

    /// Register a light profile resource mapping.
    ///
    /// \param res_kind  the kind of the resource (texture or invalid reference)
    /// \param res_url   the URL of the texture resource if any
    /// \param res_tag   the tag of the texture resource
    /// \param idx       the mapped index value representing the resource in a lookup table
    /// \param valid     true if this is a valid resource, false otherwise
    /// \param power     the power of this light profile
    /// \param maximum   the maximum of this light profile
    void map_lp_resource(
        IValue::Kind res_kind,
        char const   *res_url,
        int          res_tag,
        size_t       idx,
        bool         valid,
        float        power,
        float        maximum) MDL_FINAL;

    /// Register a bsdf measurement resource mapping.
    ///
    /// \param res_kind  the kind of the resource (texture or invalid reference)
    /// \param res_url   the URL of the texture resource if any
    /// \param res_tag   the tag of the texture resource
    /// \param idx       the mapped index value representing the resource in a lookup table
    /// \param valid     true if this is a valid resource, false otherwise
    void map_bm_resource(
        IValue::Kind res_kind,
        char const   *res_url,
        int          res_tag,
        size_t       idx,
        bool         valid) MDL_FINAL;

    /// Analyze one root of a lambda function.
    ///
    /// \param[in]  proj           the root number
    /// \param[in]  name_resolver  a call name resolver
    /// \param[out] result         the analysis result
    ///
    /// \return true on success, false if proj is out of bounds
    bool analyze(
        size_t                    proj,
        ICall_name_resolver const *name_resolver,
        Analysis_result           &result) const MDL_FINAL;

    /// Optimize the lambda function.
    ///
    /// \param[in]  name_resolver   a call name resolver for inlining functions
    /// \param[in]  call_evaluator  a call evaluator for handling some intrinsic functions
    void optimize(
        ICall_name_resolver const *name_resolver,
        ICall_evaluator           *call_evaluator) MDL_FINAL;

    /// Returns true if a switch function was "modified", by adding a new
    /// root expression.
    ///
    /// \param reset  if true, reset the modify flag
    ///
    /// \note Deleting a root expression does not set the modify flag.
    ///       The idea is, that even with deleted entries an already compiled
    ///       function can be reused (some roots will be never called), but
    ///       adding a new root must trigger recompilation.
    bool is_modified(bool reset = true) MDL_FINAL;

    /// Returns true if a switch function was "modified" by removing a
    /// root expression.
    bool has_dead_code() const MDL_FINAL;

    /// Pass the uniform context for a given call node.
    ///
    /// \param name_resolver    the call name resolver
    /// \param expr             the lambda expression
    /// \param world_to_object  the world-to-object transformation matrix for this function
    /// \param object_to_world  the object-to-world transformation matrix for this function
    /// \param object_id        the result of state::object_id() for this function
    ///
    /// \return expr if the uniform state is not used, otherwise a modified call
    DAG_node const *set_uniform_context(
        ICall_name_resolver const *name_resolver,
        DAG_node const            *expr,
        Float4_struct const       world_to_object[4],
        Float4_struct const       object_to_world[4],
        int                       object_id) MDL_FINAL;

    /// Get a "serial version" number of this lambda function.
    ///
    /// The serial number can be used to distinguish different lambda functions.
    /// In is increased, whenever the modified flag was set or a new lambda function
    /// is created.
    unsigned get_serial_number() const MDL_FINAL;

    /// Set the name of the lambda function.
    ///
    /// The default name is "lambda".
    void set_name(char const *name) MDL_FINAL;

    /// Get the name of the lambda function.
    char const *get_name() const MDL_FINAL;

    /// Get the hash value of this lambda function.
    ///
    /// \note: the hash value is computed on demand
    DAG_hash const *get_hash() const MDL_FINAL;

    /// Returns the number of parameters of this lambda function.
    size_t get_parameter_count() const MDL_FINAL;

    /// Return the type of the i'th parameter.
    ///
    /// \param i  the parameter index
    IType const *get_parameter_type(size_t i) const MDL_FINAL;

    /// Return the name of the i'th parameter.
    ///
    /// \param i  the parameter index
    char const *get_parameter_name(size_t i) const MDL_FINAL;

    /// Add a new "captured" parameter.
    ///
    /// \param type  the parameter type
    /// \param name  the name of the parameter
    ///
    /// \return  the parameter index
    size_t add_parameter(
        IType const *type,
        char const  *name) MDL_FINAL;

    /// Map material parameter i to lambda parameter j
    ///
    void set_parameter_mapping(size_t i, size_t j) MDL_FINAL;

    /// Initialize the derivative information for this lambda function.
    /// This rewrites the body/sub-expressions with derivative types.
    ///
    /// \param resolver  the call name resolver
    void initialize_derivative_infos(ICall_name_resolver const *resolver) MDL_FINAL;

    /// Returns true, if the attributes in the resource attribute table are valid.
    /// If false, only the indices are valid.
    bool has_resource_attributes() const MDL_FINAL;

    /// Sets whether the resource attribute table contains valid attributes.
    void set_has_resource_attributes(bool avail) MDL_FINAL;

    /// Set a tag, version pair for a resource value that might be reachable from this
    /// function.
    ///
    /// \param res_kind        the resource kind
    /// \param res_url         the resource url
    /// \param tag             the tag value
    void set_resource_tag(
        Resource_tag_tuple::Kind const res_kind,
        char const                     *res_url,
        int                            tag) MDL_FINAL;

    /// Remap a resource value according to the resource map.
    ///
    /// \param r  the resource
    int get_resource_tag(IValue_resource const *r) const MDL_FINAL;

    /// Get the number of entires in the resource map.
    size_t get_resource_entries_count() const MDL_FINAL;

    /// Get the i'th entry of the resource map.
    Resource_tag_tuple const *get_resource_entry(size_t index) const MDL_FINAL;

    // --------------- non-interface members ---------------

    typedef vector<Resource_tag_tuple>::Type Resource_tag_map;

    /// Get the resource tag map.
    Resource_tag_map const &get_resource_tag_map() const { return m_resource_tag_map; }

    /// Get the derivative information if they have been initialized.
    Derivative_infos const *get_derivative_infos() const;

    /// Get the MDL compiler used to create this lambda.
    MDL *get_compiler() const {
        m_mdl->retain();
        return m_mdl.get();
    }

    /// Get the return type of the lambda function.
    mi::mdl::IType const *get_return_type() const;

    /// Returns true if this lambda function is an entry point.
    bool is_entry_point() const { return true; }

    /// Returns true if this lambda function uses the render state.
    bool uses_render_state() const { return m_uses_render_state; }

    /// Sets whether this lambda function uses the render state.
    void set_uses_render_state(bool uses_render_state) { m_uses_render_state = uses_render_state; }

    /// Returns true if this lambda function uses lambda_results.
    bool uses_lambda_results() const { return m_uses_lambda_results; }

    /// Returns true if this lambda function uses resources.
    bool uses_resources() const { return true; }

    /// Returns true if this lambda function can throw.
    bool can_throw() const { return true; }

    /// Check if the given DAG expression may use varying state data.
    ///
    /// \param resolver  the call name resolver
    /// \param expr      the expression to check
    bool may_use_varying_state(
        ICall_name_resolver const *resolver,
        DAG_node const            *expr) const;

    /// Serialize this lambda function to the given serializer.
    ///
    /// \param is  the serializer
    void serialize(ISerializer *is) const;

    /// Deserialize a lambda function from the given deserializer.
    ///
    /// \param alloc        the allocator
    /// \param mdl          the MDL compiler
    /// \param de           the deserializer
    static Lambda_function *deserialize(
        IAllocator    *alloc,
        MDL           *mdl,
        IDeserializer *de);

    /// Get the resource attribute map of this lambda function.
    Resource_attr_map const &get_resource_attribute_map() const { return m_resource_attr_map; }

    /// Get the execution context of this lambda function.
    ILambda_function::Lambda_execution_context get_execution_context() const {
        return m_context;
    }

    /// Checks if the uniform state was set.
    bool is_uniform_state_set() const;

    /// Enable common subexpression elimination.
    ///
    /// \param flag  If true, CSE will be enabled, else disabled.
    /// \return      The old value of the flag.
    bool enable_cse(bool flag) { return m_node_factory.enable_cse(flag); }

    // for debugging only

    /// Dump a lambda expression to a .gv file.
    ///
    /// \param expr   the lambda root expression
    /// \param name   the name of the file dump
    void dump(DAG_node const *expr, char const *name) const;

    private:
    /// Find the resource tag of a resource.
    ///
    /// \param res_kind        the resource kind
    /// \param res_url         the resource url
    ///
    /// \return 0 if not found, else the assigned tag
    int find_resource_tag(
        Resource_tag_tuple::Kind const res_kind,
        char const                     *res_url) const;

    /// Add tag, version pair for a resource value that might be reachable from this
    /// function.
    ///
    /// \param res_kind        the resource kind
    /// \param res_url         the resource url
    /// \param tag             the tag value
    void add_resource_tag(
        Resource_tag_tuple::Kind const res_kind,
        char const                     *res_url,
        int                            tag);

private:
    typedef ILambda_function::Lambda_execution_context Lambda_execution_context;

    /// Parameter info for every captured lambda parameter.
    struct Parameter_info {
        /// Constructor.
        Parameter_info(
            IType const *type,
            char const  *name)
        : m_type(type)
        , m_name(name)
        {
        }

        IType const *m_type;   ///< The type of the parameter.
        char const  *m_name;   ///< The name of the parameter.
    };

    /// Constructor.
    ///
    /// \param alloc             The allocator.
    /// \param compiler          The core compiler.
    /// \param context           The execution context for this lambda function.
    Lambda_function(
        IAllocator               *alloc,
        MDL                      *compiler,
        Lambda_execution_context context);

    /// Get the internal space from the execution context.
    ///
    /// \param context  the execution context
    static char const *internal_space(
        Lambda_execution_context context);

    /// Create an empty lambda function with the same option as a give other.
    static Lambda_function *clone_empty(Lambda_function const &other);

    /// Return a free root index.
    size_t find_free_root_index();

    /// Returns true if the given semantic belongs to a varying state function.
    ///
    /// \param sema  a MDL intrinsic function semantic
    static bool is_varying_state_semantic(IDefinition::Semantics sema);

    /// Analyze an expression function.
    ///
    /// \param[out] result  the analysis result
    ///
    /// \returns true on success.
    bool analyze(
        DAG_node const            *expr,
        ICall_name_resolver const *resolver,
        Analysis_result           &result) const;

    /// Update the hash value.
    void update_hash() const;

private:
    /// The mdl compiler.
    mi::base::Handle<MDL> m_mdl;

    /// The memory arena that holds all types, symbols and IR nodes of this instance.
    Memory_arena m_arena;

    /// The symbol table;
    Symbol_table m_sym_tab;

    /// The type factory.
    mutable Type_factory m_type_factory;

    /// The value factory.
    mutable Value_factory m_value_factory;

    /// The node factory.
    DAG_node_factory_impl m_node_factory;

    /// The name of this lambda function.
    string m_name;

    typedef ptr_hash_map<DAG_node const, size_t>::Type Root_map;
    typedef vector<DAG_node const *>::Type             Root_vector;

    /// The map of root nodes.
    Root_map m_root_map;

    /// The list of root nodes.
    Root_vector m_roots;

    /// The resource attribute map.
    Resource_attr_map m_resource_attr_map;

    /// True, if the attributes in the resource attribute map are valid.
    /// If resolving resources is disabled, the resource attribute map will only be used
    /// for managing the resource indices.
    bool m_has_resource_attributes;

    /// The execution context of this lambda function.
    Lambda_execution_context m_context;

    /// The hash value of this function.
    mutable DAG_hash m_hash;

    /// The lambda function body expression if this is a simple lambda function.
    DAG_node const *m_body_expr;

    typedef vector<Parameter_info>::Type Param_info_vec;

    /// The captured parameters.
    Param_info_vec m_params;

    typedef map<size_t, size_t>::Type Index_map;

    /// The index mapping.
    Index_map m_index_map;

    /// The serial number of this lambda function.
    mutable unsigned m_serial_number;

    /// The next serial number
    static mi::base::Atom32 g_next_serial;

    /// If true, this function uses the (render) state.
    unsigned m_uses_render_state:1;

    /// If true, garbage collection must run.
    unsigned m_has_dead_code:1;

    /// If true, the switch function was modified.
    unsigned m_is_modified:1;

    /// If true, this function uses a lambda_results array (libbsdf mode).
    unsigned m_uses_lambda_results:1;

    /// If false, serial number requires an update.
    mutable unsigned m_serial_is_valid:1;

    /// If true, the hash is valid.
    mutable unsigned m_hash_is_valid:1;

    /// If true, m_deriv_infos contains valid information.
    bool m_deriv_infos_calculated;

    /// The derivative analysis information, if requested during initialization.
    Derivative_infos m_deriv_infos;

    /// The resource tag map, mapping resource values to (tag, version) pair.
    Resource_tag_map m_resource_tag_map;
};

/// This class holds the DF and non-DF parts of an MDL material surface.
class Distribution_function : public Allocator_interface_implement<IDistribution_function>
{
    typedef Allocator_interface_implement<IDistribution_function> Base;
    friend class Allocator_builder;

public:
    /// Initialize this distribution function object for the given material
    /// with the given requested functions.
    /// Any additionally required expressions from the material will also be handled.
    /// Any material parameters must already be registered in the root lambda at this point.
    /// The DAG nodes must already be owned by the root lambda.
    ///
    /// \param material_constructor       the DAG node of the material constructor
    /// \param requested_functions        the expressions for which functions will be generated
    /// \param num_functions              the number of requested functions
    /// \param include_geometry_normal    if true, the geometry normal will be handled
    /// \param calc_derivative_infos      if true, derivative information will be calculated
    /// \param allow_double_expr_lambdas  if true, expression lambdas may be created for double
    ///                                   values
    /// \param name_resolver              the call name resolver
    ///
    /// \returns EC_NONE, if initialization was successful, an error code otherwise.
    Error_code initialize(
        DAG_node const            *material_constructor,
        Requested_function        *requested_functions,
        size_t                     num_functions,
        bool                       include_geometry_normal,
        bool                       calc_derivative_infos,
        bool                       allow_double_expr_lambdas,
        ICall_name_resolver const *name_resolver) MDL_FINAL;

    /// Get the root lambda function used to build nodes and manage parameters and resources.
    ILambda_function *get_root_lambda() const MDL_FINAL;

    /// Add the given function as main lambda function.
    ///
    /// \param lambda  the function to add
    size_t add_main_function(ILambda_function *lambda);

    /// Get the main lambda function for the given index, representing a requested function.
    ///
    /// \param index  the index of the main lambda
    ///
    /// \returns  the requested main lambda function or NULL, if the index is invalid
    ILambda_function *get_main_function(size_t index) const MDL_FINAL;

    /// Get the number of main lambda functions.
    size_t get_main_function_count() const MDL_FINAL;

    /// Add the given expression lambda function to the distribution function.
    /// The index as a decimal string can be used as name in DAG call nodes with the semantics
    /// DS_INTRINSIC_DAG_CALL_LAMBDA to reference these lambda functions.
    ///
    /// \param lambda  the lambda function responsible for calculating an expression
    ///
    /// \returns  the index of the expression lambda function
    size_t add_expr_lambda_function(ILambda_function *lambda) MDL_FINAL;

    /// Get the expression lambda function for the given index.
    ///
    /// \param index  the index of the expression lambda
    ///
    /// \returns  the requested expression lambda function or NULL, if the index is invalid
    ILambda_function *get_expr_lambda(size_t index) const MDL_FINAL;

    /// Get the number of expression lambda functions.
    size_t get_expr_lambda_count() const MDL_FINAL;

    /// Set a special lambda function for getting certain material properties.
    ///
    /// \param kind    the kind of special lambda function to set
    /// \param lambda  the lambda function to associate with this kind
    void set_special_lambda_function(
        Special_kind kind,
        ILambda_function *lambda) MDL_FINAL;

    /// Get the expression lambda index for the given special lambda function kind.
    ///
    /// \param kind    the kind of special lambda function to get
    ///
    /// \returns  the requested expression lambda index or ~0, if the index is invalid or
    ///           the special lambda function has not been set
    size_t get_special_lambda_function_index(Special_kind kind) const MDL_FINAL;

    /// Returns the number of distribution function handles referenced by this
    /// distribution function.
    size_t get_df_handle_count() const MDL_FINAL;

    /// Returns a distribution function handle referenced by this distribution function.
    ///
    /// \param index  the index of the handle to return
    ///
    /// \return the name of the handle, or \c NULL, if the \p index was out of range.
    char const *get_df_handle(size_t index) const MDL_FINAL;

    /// Register a distribution function handle.
    ///
    /// \param handle_name  the name of the new handle
    ///
    /// \return the index of the handle
    size_t add_df_handle(char const *handle_name)
    {
        m_df_handles.push_back(handle_name);
        return m_df_handles.size() - 1;
    }

    /// Register a distribution function handle for a main function.
    ///
    /// \param main_func_index  the index of the main function
    /// \param handle_name      the name of the new handle
    ///
    /// \return the index of the handle
    size_t add_main_func_df_handle(size_t main_func_index, char const *handle_name)
    {
        MDL_ASSERT(main_func_index < m_main_func_df_handles.size());
        m_main_func_df_handles[main_func_index].push_back(handle_name);
        return m_main_func_df_handles[main_func_index].size() - 1;
    }

    /// Returns the number of distribution function handles referenced by a given main function.
    ///
    /// \param main_func_index  the index of the main function
    ///
    /// \returns  the requested count or ~0, if the index is invalid
    size_t get_main_func_df_handle_count(size_t main_func_index) const MDL_FINAL;

    /// Returns a distribution function handle referenced by a given main function.
    ///
    /// \param main_func_index  the index of the main function
    /// \param index            the index of the handle to return
    ///
    /// \return the name of the handle, or \c NULL, if the \p index was out of range.
    char const *get_main_func_df_handle(size_t main_func_index, size_t index) const MDL_FINAL;

    /// Get the resource attribute map of this distribution function.
    Resource_attr_map const &get_resource_attribute_map() const;

    /// Set a tag, version pair for a resource value that might be reachable from this
    /// function.
    ///
    /// \param res_kind        the resource kind
    /// \param res_url         the resource url
    /// \param tag             the tag value
    void set_resource_tag(
        Resource_tag_tuple::Kind const res_kind,
        char const                     *res_url,
        int                            tag) MDL_FINAL;

    /// Get the derivative information if they were requested during initialization.
    Derivative_infos const *get_derivative_infos() const;

    /// Returns the MDL compiler used to create the distribution function.
    mi::base::Handle<MDL> get_compiler() const { return mi::base::Handle<MDL>(m_mdl); }

    /// Dump the distribution function to a .gv file with the given name.
    void dump(char const *name) const;

    /// Get the derivative information if they were requested during initialization.
    Derivative_infos *get_writable_derivative_infos() { return &m_deriv_infos; }

private:
    /// Find the resource tag of a resource.
    ///
    /// \param res_kind        the resource kind
    /// \param res_url         the resource url
    ///
    /// \return 0 if not found, else the assigned tag
    int find_resource_tag(
        Resource_tag_tuple::Kind const res_kind,
        char const                     *res_url) const;

    /// Add tag, version pair for a resource value that might be reachable from this
    /// function.
    ///
    /// \param res_kind        the resource kind
    /// \param res_url         the resource url
    /// \param tag             the tag value
    void add_resource_tag(
        Resource_tag_tuple::Kind res_kind,
        char const               *res_url,
        int                      tag);

private:
    /// Constructor.
    ///
    /// \param alloc             The allocator.
    /// \param compiler          The core compiler.
    Distribution_function(
        IAllocator                         *alloc,
        MDL                                *compiler);

    /// The MDL compiler.
    mi::base::Handle<MDL> m_mdl;

    /// One lambda function, which owns all nodes and values, and manages parameters and resources.
    mi::base::Handle<ILambda_function> m_root_lambda;

    /// The main lambda functions, which will be exported.
    vector<mi::base::Handle<ILambda_function> >::Type m_main_functions;

    /// Collection of expression lambdas generated from the DAG.
    vector<mi::base::Handle<ILambda_function> >::Type m_expr_lambdas;

    /// Array of indexes into the collection of expression lambdas for special lambda functions
    /// used to get certain material properties.
    /// They are only set to non ~0 values if they are needed by the BSDFs.
    size_t m_special_lambdas[SK_NUM_KINDS];

    /// If true, m_deriv_infos contains valid information.
    bool m_deriv_infos_calculated;

    /// The derivative analysis information, if requested during initialization.
    Derivative_infos m_deriv_infos;

    /// List of DF handle strings owned by the value factory of all main functions.
    vector<char const *>::Type m_df_handles;

    /// List of DF handle strings owned by the value factory per main function.
    vector<vector<char const *>::Type>::Type m_main_func_df_handles;

    typedef vector<Resource_tag_tuple>::Type Resource_tag_map;

    // arena for strings.
    Memory_arena m_arena;

    /// The resource to tag map.
    Resource_tag_map m_resource_tag_map;
};

}  // mdl
}  // mi

#endif // MDL_GENERATOR_DAG_LAMBDA_FUNCTION_H
