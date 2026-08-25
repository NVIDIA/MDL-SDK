/******************************************************************************
 * Copyright (c) 2013-2026, NVIDIA CORPORATION. All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
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

#include "mdl/codegenerators/generator_code/generator_code_resource_manager.h"

#include "generator_dag_derivatives.h"
#include "generator_dag_dumper.h"
#include "generator_dag_ir.h"
#include "generator_dag_unit.h"

namespace mi {
namespace mdl {

class Derivative_infos;
class Function_context;
class IMDL;
class IValue_resource;


/// This class handles the creation and compilation of lambda functions.
///
/// The body (expression) of this lambda function is expression as a DAG to simplify the
/// reuse of DAG components.
/// Once the expression is build, in can be compiled into a function.
class Lambda_function : public Allocator_interface_implement<ILambda_function>
{
    typedef Allocator_interface_implement<ILambda_function> Base;
    friend class Allocator_builder;

    typedef ptr_hash_map<DAG_node const, DAG_node const *>::Type                     Node_cache;
    typedef ptr_hash_map<char const, unsigned, cstring_hash, cstring_equal_to>::Type Fname_tbl;

    /// Helper class for handling node imports.
    class Import_helper {
    public:
        /// Constructor.
        ///
        /// \param dest  the unit in which we will import
        /// \param src   the unit we will import from
        Import_helper(
            DAG_unit       &dest,
            DAG_unit const &src);

        /// Translate a src debug info into a destination debug info.
        DAG_DbgInfo import(DAG_DbgInfo src);

        /// Check if the given expression exists already in the import cache.
        Node_cache::iterator find(DAG_node const *expr) { return m_node_cache.find(expr); }

        /// Return the end iterator of the import cache.
        Node_cache::iterator end() { return m_node_cache.end(); }

        /// Operator [] on the import cache.
        DAG_node const *&operator[](DAG_node const *node) { return m_node_cache[node]; }
    private:
        /// The Node_cache (stores which nodes are already visited and its imported result).
        Node_cache m_node_cache;

        /// The file name table.
        Fname_tbl m_fname_tbl;

        /// The translation table from src file IDs to dest file IDs.
        vector<unsigned>::Type m_translate;

        /// True if debug info translation is enabled.
        bool m_has_dbg_info;
    };

public:
        /// Get the DAG_unit of this lambda function.
    DAG_unit &get_dag_unit() MDL_FINAL;

    /// Get the DAG_unit of this lambda function.
    DAG_unit const &get_dag_unit() const MDL_FINAL;

    /// Get the type factory of this function.
    Type_factory *get_type_factory() MDL_FINAL;

    /// Get the value factory of this function.
    Value_factory *get_value_factory() MDL_FINAL;

    /// Create a constant.
    /// \param  value       The value of the constant.
    /// \param dbg_info     The debug info for this constant if any.
    /// \returns            The created constant.
    ///
    /// \note Use this method to create arguments of the instance.
    DAG_constant const *create_constant(
        IValue const *value,
        DAG_DbgInfo  dbg_info) MDL_FINAL;

    /// Create a call.
    /// \param  name            The absolute name of the called function.
    /// \param  sema            The semantic of the called function.
    /// \param  call_args       The call arguments of the called function.
    /// \param  num_call_args   The number of call arguments.
    /// \param  ret_type        The return type of the called function.
    /// \param dbg_info         The debug info for this call if any.
    ///
    /// \returns                The created call or an equivalent node.
    ///
    /// \note Use this method to create arguments of the instance.
    DAG_node const *create_call(
        char const                    *name,
        IDefinition::Semantics        sema,
        DAG_call::Call_argument const call_args[],
        int                           num_call_args,
        IType const                   *ret_type,
        DAG_DbgInfo                   dbg_info) MDL_FINAL;

    /// Create a parameter reference.
    /// \param  type        The type of the parameter
    /// \param  index       The index of the parameter.
    /// \param dbg_info     The debug info for this parameter if any.
    ///
    /// \returns            The created parameter[index] reference.
    ///
    /// \note If index was mapped using set_parameter_mapping(index, n),
    ///       a parameter[n] will be created.
    DAG_parameter const *create_parameter(
        IType const *type,
        int         index,
        DAG_DbgInfo dbg_info) MDL_FINAL;

    /// Enable common subexpression elimination.
    ///
    /// \param flag  If true, CSE will be enabled, else disabled.
    /// \return      The old value of the flag.
    bool enable_cse(bool flag) MDL_FINAL;

    /// Enable optimization.
    ///
    /// \param flag  If true, optimizations in general will be enabled, else disabled.
    /// \return      The old value of the flag.
    bool enable_opt(bool flag) MDL_FINAL;

    /// Enable unsafe math optimizations.
    ///
    /// \param flag  If true, unsafe math optimizations will be enabled, else disabled.
    /// \return      The old value of the flag.
    bool enable_unsafe_math_opt(bool flag) MDL_FINAL;

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
    /// \param owner  the DAG_unit that owns the expression to import
    /// \param expr   the DAG expression to import
    ///
    /// \return the imported DAG expression
    DAG_node const *import_expr(
        DAG_unit const &owner,
        DAG_node const *expr) MDL_FINAL;

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
    /// \param res_sel         the selector of the texture resource if any
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
        char const                     *res_sel,
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

    /// Analyze one root of a lambda function or the body expression, in case there are no roots.
    ///
    /// \param[in]  proj           the root number, ignored if there are no roots but a body expression.
    /// \param[in]  name_resolver  a call name resolver
    /// \param[out] result         the analysis result
    ///
    /// \return true on success, false if proj is out of bounds or in case of no roots, if there is no body.
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
    /// \param name  the name of the lambda function
    ///
    /// \note: The default name is "lambda".
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
    /// \param i   the material parameter index
    /// \param j   the lambda function parameter index
    ///
    /// \note This mapping will influence the create_parameter() function.
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

    /// Set a tag for a resource value that might be reachable from this lambda function.
    ///
    /// \param res_kind        the resource kind
    /// \param res_url         the resource url
    /// \param res_sel         the resource selector (for textures) or NULL
    /// \param tag             the tag value
    void set_resource_tag(
        Resource_tag_tuple::Kind const res_kind,
        char const                     *res_url,
        char const                     *res_sel,
        int                            tag) MDL_FINAL;

    /// Remap a resource value according to the resource map.
    ///
    /// \param r  the resource
    ///
    /// \return if a resource tag was set for this resource using set_resource_tag() this tag,
    ///         otherwise the tag stored in the value itself
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
    ///
    /// \note switch lambdas return bool here, because their real result is passed by reference
    ///       as the first parameter
    mi::mdl::IType const *get_return_type() const;

    /// Returns true if this lambda function is an entry point.
    bool is_entry_point() const { return true; }

    /// Returns true if this lambda function uses the varying state.
    bool uses_varying_state() const { return m_uses_varying_state; }

    /// Sets whether this lambda function uses the varying state.
    void set_uses_varying_state(bool uses_varying_state) const {
        m_uses_varying_state = uses_varying_state;
    }

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

    // for debugging only

    /// Get the node factory.
    DAG_node_factory_impl const &get_node_factory() const { return m_node_factory; }

    /// Get the node factory (non-const).
    DAG_node_factory_impl &get_node_factory() { return m_node_factory; }

    /// Get the type factory.
    Type_factory const &get_type_factory() const { return m_dag_unit.get_type_factory(); }

    /// Get the value factory.
    Value_factory const &get_value_factory() const { return m_dag_unit.get_value_factory(); }

    /// Returns true if this lambda function owns the given node.
    bool is_owner(DAG_node const *node) const { return m_node_factory.is_owner(node); }

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
    /// \param res_url         the resource selector (for textures) or NULL
    ///
    /// \return 0 if not found, else the assigned tag
    int find_resource_tag(
        Resource_tag_tuple::Kind const res_kind,
        char const                     *res_url,
        char const                     *res_sel) const;

    /// Add a tag for a resource value that might be reachable from this function.
    ///
    /// \param res_kind        the resource kind
    /// \param res_url         the resource url
    /// \param res_sel         the resource selector (for textures) or NULL
    /// \param tag             the tag value
    void add_resource_tag(
        Resource_tag_tuple::Kind const res_kind,
        char const                     *res_url,
        char const                     *res_sel,
        int                            tag);

private:
    typedef ILambda_function::Lambda_execution_context Lambda_execution_context;

    /// Parameter info for every captured lambda parameter.
    struct Parameter_info {
        /// Constructor.
        ///
        /// \param type  the type of the parameter
        /// \param name  the name of the parameter
        ///
        /// \note: it is expected, that the name is stored somewhere else, no copy is made
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

    /// Analyze a DAG expression.
    ///
    /// \param[in]  expr      the DAG expression
    /// \param[in]  resolver  a call name resolver
    /// \param[out] result    the analysis result
    ///
    /// \returns true on success, false if the analysis failed.
    bool analyze(
        DAG_node const            *expr,
        ICall_name_resolver const *resolver,
        Analysis_result           &result) const;

    /// Update the hash value.
    void update_hash() const;

    /// Import (i.e. deep-copy) a DAG expression into this lambda function.
    ///
    /// \param expr           the DAG expression to import
    /// \param import_helper  the import helper
    ///
    /// \return the imported DAG expression
    DAG_node const *do_import_expr(
        DAG_node const *expr,
        Import_helper  &import_helper);

private:
    /// The mdl compiler.
    mi::base::Handle<MDL> m_mdl;

    /// The DAG_unit of this lambda function.
    DAG_unit m_dag_unit;

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

    /// If true, this function uses the varying state.
    mutable unsigned m_uses_varying_state:1;

    /// If true, garbage collection must run.
    unsigned m_has_dead_code:1;

    /// If true, the switch function was modified.
    unsigned m_is_modified:1;

    /// If false, serial number requires an update.
    mutable unsigned m_serial_is_valid:1;

    /// If true, the hash is valid.
    mutable unsigned m_hash_is_valid:1;

    /// If true, m_deriv_infos contains valid information.
    unsigned m_deriv_infos_calculated:1;

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
    /// The possible kinds of special nodes.
    enum Special_kind {
        SK_INVALID = -1,                     ///< Invalid special kind.
        SK_MATERIAL_IOR = 0,                 ///< Special kind for material.ior.
        SK_MATERIAL_THIN_WALLED,             ///< Special kind for material.thin_walled.
        SK_MATERIAL_VOLUME_ABSORPTION,       ///< Special kind for
                                             ///< material.volume.absorption_coefficient.
        SK_MATERIAL_GEOMETRY_DISPLACEMENT,   ///< Special kind for material.geometry.displacement.
        SK_MATERIAL_GEOMETRY_CUTOUT_OPACITY, ///< Special kind for material.geometry.cutout_opacity.
        SK_MATERIAL_GEOMETRY_NORMAL,         ///< Special kind for material.geometry.normal.

        SK_NUM_KINDS                         ///< The number of special kinds.
    };

    /// The possible evaluation states for a requested node.
    enum Eval_state {
        ES_BEGIN_STATE = 0,           ///< The node is evaluated before the end of the evaluation
                                      ///< of geometry.normal.
        ES_AFTER_GEOMETRY_NORMAL = 1, ///< The node is evaluated after geometry.normal has been
                                      ///< evaluated.
        ES_LAST = ES_AFTER_GEOMETRY_NORMAL
    };

    /// Struct representing a DAG node explicitly or implicitly requested by the renderer.
    struct Requested_node {
        /// Constructor.
        ///
        /// \param alloc          the allocator
        /// \param node           the requested node
        /// \param path           the path to the requested node in the material instance.
        ///                       The string must be allocated via the arena.
        /// \param function_name  the function name for the node if it should be exported,
        ///                       nullptr otherwise. The string must be allocated via the arena.
        /// \param eval_state     the state in which the node shall be evaluated
        Requested_node(
            IAllocator     *alloc,
            DAG_node const *node,
            char const     *path,
            char const     *function_name,
            Eval_state      eval_state)
            : node(node)
            , path(path)
            , function_name(function_name)
            , eval_state(eval_state)
            , df_handles(alloc)
        {}

        /// The requested node (owned by the root lambda).
        DAG_node const *node;

        /// The path to the requested node in the material instance (for debugging).
        char const     *path;

        /// The function name if an exported function shall be generated.
        /// nullptr for nodes only required by libbsdf (implicitly requested).
        char const     *function_name;

        /// The evaluation state at which the node must be evaluated.
        Eval_state      eval_state;

        /// List of DF handle strings owned by the value factory referenced transitively by the
        /// requested node.
        vector<char const *>::Type df_handles;
    };

    /// Initialize this distribution function object for the given material instance
    /// with the given requested functions.
    /// Any additionally required expressions from the material will also be handled.
    /// Any material parameters must already be registered in the root lambda at this point.
    ///
    /// \param mat_instance               the material instance
    /// \param requested_functions        the expressions for which functions will be generated
    /// \param num_req_functions          the number of requested functions
    /// \param calc_derivative_infos      if true, derivative information will be calculated
    /// \param enable_spectral_conversions if true, color parameters will be wrapped with spectral conversions
    /// \param name_resolver              the call name resolver
    ///
    /// \returns EC_NONE, if initialization was successful, an error code otherwise.
    Error_code initialize(
        IMaterial_instance const  *mat_instance,
        Requested_function        *requested_functions,
        size_t                     num_req_functions,
        bool                       calc_derivative_infos,
        bool                       enable_spectral_conversions,
        ICall_name_resolver const *name_resolver) MDL_FINAL;

    /// Get the root lambda function used to build nodes and manage parameters and resources.
    /// The body will be set to the constructor from the used material instance in
    /// @ref initialize(). If derivatives are enabled, the body will have been rebuilt with
    /// derivative types.
    Lambda_function *get_root_lambda() const MDL_FINAL;

    /// Add a requested node.
    ///
    /// \param node           the DAG node
    /// \param path           the path to the requested node in the material instance
    /// \param function_name  a function name if the code for the node should be exported
    ///                       or nullptr otherwise
    /// \param eval_state     the state in which the node should be evaluated
    size_t add_requested_node(
        DAG_node const *node,
        char const     *path,
        char const     *function_name,
        Eval_state      eval_state);

    /// Get the requested node for the given index.
    /// The returned pointer becomes invalid, if more requested nodes are added.
    ///
    /// \param index  the index of the requested node
    Requested_node *get_requested_node(size_t index);

    /// Get the requested node for the given index.
    /// The returned pointer becomes invalid, if more requested nodes are added.
    ///
    /// \param index  the index of the requested node
    Requested_node const *get_requested_node(size_t index) const;

    /// Set the number of explicitly requested nodes. These are always the first nodes in the
    /// list of requested nodes.
    void set_explicit_requested_node_count(size_t count)
    {
        m_explicit_requested_node_count = count;
    }

    /// Get the number of explicitly requested nodes. These are always the first nodes in the
    /// list of requested nodes.
    size_t get_explicit_requested_node_count() const MDL_FINAL
    {
        return m_explicit_requested_node_count;
    }

    /// Get the number of explicitly and implicitly requested nodes.
    /// The implicitly requested nodes always follow the explicitly requested nodes.
    size_t get_total_requested_node_count() const MDL_FINAL
    {
        return m_requested_nodes.size();
    }

    /// Get the DAG node for a requested node.
    ///
    /// \param index  the index of the requested node
    DAG_node const *get_requested_dag_node(size_t index) const MDL_FINAL
    {
        if (index >= m_requested_nodes.size()) {
            return nullptr;
        }
        return m_requested_nodes[index].node;
    }

    /// Set a special node for getting certain material properties.
    ///
    /// \param kind  the kind of special lambda function to set
    /// \param node  the DAG node to associate with this kind
    /// \param path  the path to the requested node in the material instance
    void add_special_node(
        Special_kind    kind,
        DAG_node const *node,
        char const     *path);

    /// Set a special node for getting certain material properties.
    ///
    /// \param kind                  the kind of special lambda function to set
    /// \param requested_node_index  the index of the requested node
    void set_special_node(
        Special_kind  kind,
        size_t        requested_node_index);

    /// Get the requested node index for the given special node kind.
    ///
    /// \param kind    the kind of special node to get
    ///
    /// \returns  the requested node index or ~0, if the index is invalid or
    ///           the special node has not been set
    size_t get_special_node_index(Special_kind kind) const;

    /// Returns the number of distribution function handles referenced by this
    /// distribution function.
    size_t get_df_handle_count() const;

    /// Returns a distribution function handle referenced by this distribution function.
    ///
    /// \param index  the index of the handle to return
    ///
    /// \return the name of the handle, or \c NULL, if the \p index was out of range.
    char const *get_df_handle(size_t index) const;

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

    /// Get the resource attribute map of this distribution function.
    Resource_attr_map const &get_resource_attribute_map() const;

    /// Set a tag, version pair for a resource value that might be reachable from this
    /// function.
    ///
    /// \param res_kind        the resource kind
    /// \param res_url         the resource url
    /// \param res_sel         the resource selector (for textures) or NULL
    /// \param tag             the tag value
    void set_resource_tag(
        Resource_tag_tuple::Kind const res_kind,
        char const                     *res_url,
        char const                     *res_sel,
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
    /// \param res_sel         the resource selector (for textures) or NULL
    ///
    /// \return 0 if not found, else the assigned tag
    int find_resource_tag(
        Resource_tag_tuple::Kind const res_kind,
        char const                     *res_url,
        char const                     *res_sel) const;

    /// Add tag, version pair for a resource value that might be reachable from this
    /// function.
    ///
    /// \param res_kind        the resource kind
    /// \param res_url         the resource url
    /// \param res_sel         the resource selector (for textures) or NULL
    /// \param tag             the tag value
    void add_resource_tag(
        Resource_tag_tuple::Kind res_kind,
        char const               *res_url,
        char const               *res_sel,
        int                      tag);

private:
    /// Constructor.
    ///
    /// \param alloc             The allocator.
    /// \param compiler          The core compiler.
    Distribution_function(
        IAllocator *alloc,
        MDL        *compiler);

    /// The MDL compiler.
    mi::base::Handle<MDL> m_mdl;

    /// One lambda function, which owns all nodes and values, and manages parameters and resources.
    mi::base::Handle<Lambda_function> m_root_lambda;

    /// The number of explicitly requested nodes when the @ref initialize function was called.
    size_t m_explicit_requested_node_count;

    /// List of requested nodes from the material instance.
    vector<Requested_node>::Type m_requested_nodes;

    /// Array of indexes into the collection of requested nodes for special nodes
    /// used to get certain material properties.
    /// They are only set to non ~0 values if they are needed by the BSDFs.
    size_t m_special_nodes[SK_NUM_KINDS];

    /// If true, m_deriv_infos contains valid information.
    bool m_deriv_infos_calculated;

    /// The derivative analysis information, if requested during initialization.
    Derivative_infos m_deriv_infos;

    /// List of DF handle strings owned by the value factory of all requested nodes.
    vector<char const *>::Type m_df_handles;

    typedef vector<Resource_tag_tuple>::Type Resource_tag_map;

    // Arena for strings.
    Memory_arena m_arena;

    /// The resource to tag map.
    Resource_tag_map m_resource_tag_map;
};

/// Helper class to dump a distribution function as a dot file.
class Distribution_function_dumper : public DAG_dumper {
    typedef DAG_dumper Base;
public:
    typedef ptr_hash_map<DAG_node const, char const *>::Type Node_color_map;

    /// Constructor.
    ///
    /// \param alloc           the allocator
    /// \param out             an output stream, the dot file is written to
    /// \param dist_func       the distribution function to dump
    /// \param node_color_map  a map from DAG nodes to Graphviz color strings
    Distribution_function_dumper(
        IAllocator                  *alloc,
        IOutput_stream              *out,
        Distribution_function const *dist_func,
        Node_color_map              &node_color_map);

    /// Dump the distribution function to the output stream.
    void dump();

    /// Get the parameter name for the given index if any.
    ///
    /// \param index  the index of the parameter
    char const *get_parameter_name(int index) MDL_FINAL;

    /// Get a color for the given node or nullptr for default
    ///
    /// \param node  the node
    char const *get_node_color(DAG_node const *node) MDL_FINAL;

    /// Get a prefix for the label for the given node or nullptr for no prefix.
    ///
    /// \param node  the node
    char const *get_node_label_prefix(DAG_node const *node) MDL_FINAL;

private:
    /// Buffer used for converting numbers to strings.
    char m_name_buf[40];

    /// Currently processed distribution function.
    Distribution_function const *m_dist_func;

    /// Map from DAG nodes to Graphviz color strings.
    Node_color_map &m_node_color_map;
};

}  // mdl
}  // mi

#endif // MDL_GENERATOR_DAG_LAMBDA_FUNCTION_H
