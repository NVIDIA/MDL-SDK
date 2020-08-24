/******************************************************************************
 * Copyright (c) 2011-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \file mi/mdl/mdl_code_generators.h
/// \brief Interfaces for MDL code generators
#ifndef MDL_CODE_GENERATORS_H
#define MDL_CODE_GENERATORS_H 1

#include <mi/base/iinterface.h>
#include <mi/base/handle.h>
#include <mi/base/interface_declare.h>
#include <mi/mdl/mdl_generated_dag.h>
#include <mi/mdl/mdl_generated_executable.h>

namespace mi {
namespace mdl {

class IDag_builder;
class IModule;
class ISerializer;
class IDeserializer;
class IValue_texture;
class IValue_resource;
class Options;

/// The base interface for all code generators of the MDL core compiler.
class ICode_generator : public
    mi::base::Interface_declare<0x6cb37aae,0x7aff,0x41d1,0xbe,0xff,0x1f,0xf9,0xac,0x20,0x13,0x07,
    mi::base::IInterface>
{
public:

    /// The name of the code generator option to set the internal space.
    #define MDL_CG_OPTION_INTERNAL_SPACE "internal_space"

    /// The name of the code generator option to enable or disable folding of meters_per_scene_unit.
    #define MDL_CG_OPTION_FOLD_METERS_PER_SCENE_UNIT "fold_meters_per_scene_unit"

    /// The name of the code generator option to set meters_per_scene_unit.
    #define MDL_CG_OPTION_METERS_PER_SCENE_UNIT "meters_per_scene_unit"

    /// The name of the code generator option to set wavelength_min.
    #define MDL_CG_OPTION_WAVELENGTH_MIN "wavelength_min"

    /// The name of the code generator option to set wavelength_max.
    #define MDL_CG_OPTION_WAVELENGTH_MAX "wavelength_max"

    /// Get the name of the target language.
    virtual char const *get_target_language() const = 0;

    /// Access options.
    ///
    /// Get access to the code generator options. \see mdl_code_generator_options
    virtual Options &access_options() = 0;
};

/// A simple code cache.
class ICode_cache : public
    mi::base::Interface_declare<0x34ba5c1c,0x4e58,0x42fb,0x8f,0x54,0xd3,0x39,0xb3,0x73,0xf8,0xd6,
    mi::base::IInterface>
{
public:
    /// An entry in the cache.
    struct Entry {
        struct Func_info
        {
            char const *name;
            IGenerated_code_executable::Distribution_kind dist_kind;
            IGenerated_code_executable::Function_kind func_kind;
            char const *prototypes[IGenerated_code_executable::PL_NUM_LANGUAGES];
            size_t arg_block_index;
            size_t num_df_handles;
            char const **df_handles;
            IGenerated_code_executable::State_usage state_usage;
        };

        /// Constructor.
        ///
        /// \note Note that this only transports pointer to the data, the copy must take place
        ///       inside enter() if necessary.
        Entry(
            char const      *code,
            size_t          code_size,
            char const      *const_seg,
            size_t          const_seg_size,
            char const      *arg_layout,
            size_t          arg_layout_size,
            char const      *mapped_strings[],
            size_t          mapped_string_size,
            unsigned        render_state_usage,
            Func_info const *func_infos,
            size_t          func_info_size)
        : code(code)
        , code_size(code_size)
        , const_seg(const_seg)
        , const_seg_size(const_seg_size)
        , arg_layout(arg_layout)
        , arg_layout_size(arg_layout_size)
        , render_state_usage(render_state_usage)
        , mapped_strings(mapped_strings)
        , mapped_string_size(mapped_string_size)
        , func_infos(func_infos)
        , func_info_size(func_info_size)
        {
            size_t sz = 0;
            for (size_t i = 0; i < mapped_string_size; ++i) {
                sz += strlen(mapped_strings[i]) + 1;
            }
            // align
            sz = (sz + size_t(15u)) & ~size_t(15u);
            mapped_string_data_size = sz;

            sz = 0;
            for (size_t i = 0; i < func_info_size; ++i) {
                sz += strlen(func_infos[i].name) + 1;
                for (int j = 0 ; j < int(IGenerated_code_executable::PL_NUM_LANGUAGES); ++j) {
                    sz += strlen(func_infos[i].prototypes[j]) + 1;
                }
                if (func_infos[i].num_df_handles != 0) {
                    // align
                    sz = (sz + sizeof(char *) - 1) & ~(sizeof(char *) - 1);
                    sz += func_infos[i].num_df_handles * sizeof(char *);
                    for (size_t j = 0; j < func_infos[i].num_df_handles; ++j) {
                        sz += strlen(func_infos[i].df_handles[j]) + 1;
                    }
                }
            }
            // align
            sz = (sz + size_t(15u)) & ~size_t(15u);
            func_info_string_data_size = sz;
        }

        /// Return the data size of the cache entry.
        size_t get_cache_data_size() const
        {
            return mapped_string_size * sizeof(char *) + mapped_string_data_size +
                code_size + const_seg_size + arg_layout_size +
                func_info_size * sizeof(Func_info) + func_info_string_data_size;
        }

    public:
        char const         *code;
        size_t             code_size;
        char const         *const_seg;
        size_t             const_seg_size;
        char const         *arg_layout;
        size_t             arg_layout_size;
        unsigned           render_state_usage;
        char const * const *mapped_strings;
        size_t             mapped_string_size;
        size_t             mapped_string_data_size;
        Func_info const    *func_infos;
        size_t             func_info_size;
        size_t             func_info_string_data_size;

    };

    /// Lookup a data blob.
    virtual Entry const *lookup(unsigned char const key[16]) const = 0;

    /// Enter a data blob.
    virtual bool enter(unsigned char const key[16], Entry const &entry) = 0;
};

/// A name resolver interface.
class ICall_name_resolver
{
public:
    /// Find the owner module of a given entity name.
    /// If the entity name does not contain a colon, you should return the builtins module,
    /// which you can identify by IModule::is_builtins().
    ///
    /// \param entity_name    the entity name
    ///
    /// \returns the owning module of this entity if found, NULL otherwise
    virtual IModule const *get_owner_module(char const *entity_name) const = 0;

    /// Find the owner code DAG of a given entity name.
    /// If the entity name does not contain a colon, you should return the builtins DAG,
    /// which you can identify by calling its oner module's IModule::is_builtins().
    ///
    /// \param entity_name    the entity name
    ///
    /// \returns the owning module of this entity if found, NULL otherwise
    virtual IGenerated_code_dag const *get_owner_dag(char const *entity_name) const = 0;
};

/// A resource modifier interface.
class IResource_modifier
{
public:
    /// Replace the given resource value by another of the same type
    /// if necessary.
    ///
    /// \param res    the resource value to replace
    /// \param owner  the owner module of this resource (for resolving path)
    /// \param fact   the value factory to be used to construct a new value if necessary
    ///
    /// \return a new value of the same type or res if no modification is necessary
    virtual IValue_resource const *modify(
        IValue_resource const *res,
        IModule         const *owner,
        IValue_factory        &fact) = 0;
};

/// A helper interface to map user defined functions during Lambda creation.
class ILambda_call_transformer
{
public:
    /// Transform a call into another construct.
    ///
    /// \param call     the call to transform
    /// \param builder  a DAG builder for creating new nodes
    ///
    /// \return NULL if no transformation should be done, else a new DAG call node
    virtual DAG_call const *transform(DAG_call const *call, IDag_builder *builder) = 0;
};

/// A helper interface to enumerate all resources of a lambda function.
class ILambda_resource_enumerator
{
public:
    /// The potential texture usage properties (maybe-analysis).
    enum Texture_usage_property {
        TU_NONE            = 0u,
        TU_HEIGHT          = 1u <<  0,     ///< uses tex::height()
        TU_WIDTH           = 1u <<  1,     ///< uses tex::width()
        TU_DEPTH           = 1u <<  2,     ///< uses tex::depth()
        TU_LOOKUP_FLOAT    = 1u <<  3,     ///< uses tex::lookup_float()
        TU_LOOKUP_FLOAT2   = 1u <<  4,     ///< uses tex::lookup_float2()
        TU_LOOKUP_FLOAT3   = 1u <<  5,     ///< uses tex::lookup_float3()
        TU_LOOKUP_FLOAT4   = 1u <<  6,     ///< uses tex::lookup_float4()
        TU_LOOKUP_COLOR    = 1u <<  7,     ///< uses tex::lookup_color()
        TU_TEXEL_FLOAT     = 1u <<  8,     ///< uses tex::texel_float()
        TU_TEXEL_FLOAT2    = 1u <<  9,     ///< uses tex::texel_float2()
        TU_TEXEL_FLOAT3    = 1u << 10,     ///< uses tex::texel_float3()
        TU_TEXEL_FLOAT4    = 1u << 11,     ///< uses tex::texel_float4()
        TU_TEXEL_COLOR     = 1u << 12,     ///< uses tex::texel_color()
        TU_TEXTURE_ISVALID = 1u << 13,     ///< uses tex::texture_isvalid()
    }; // can be or'ed

    /// The usage of a texture resource.
    typedef unsigned Texture_usage;

    /// Called for a texture resource.
    ///
    /// \param t          the texture resource or an invalid_ref
    /// \param tex_usage  the potential usage of the texture
    virtual void texture(IValue const *t, Texture_usage tex_usage) = 0;

    /// Called for a light profile resource.
    ///
    /// \param t  the light profile resource or an invalid_ref
    virtual void light_profile(IValue const *t) = 0;

    /// Called for a bsdf measurement resource.
    ///
    /// \param t  the bsdf measurement resource or an invalid_ref
    virtual void bsdf_measurement(IValue const *t) = 0;
};

/// A helper interface to retrieve resource attributes.
class ILambda_resource_attribute
{
public:
    /// Retrieve the attributes of a texture.
    ///
    /// \param[in]  t       the texture resource
    /// \param[out] valid   the result of texture_is_valid()
    /// \param[out] width   the width of this resource
    /// \param[out] height  the height of this resource
    /// \param[out] depth   the depth of this resource
    virtual void get_texture_attributes(
        IValue_resource const *t,
        bool                  &valid,
        int                   &width,
        int                   &height,
        int                   &depth) const = 0;

    /// Retrieve the attributes of a light profile.
    ///
    /// \param[in]  t        the light profile resource
    /// \param[out] valid    the result of light_profile_is_valid()
    /// \param[out] power    the result of light_profile_power()
    /// \param[out] maximum  the result of light_profile_maximum()
    virtual void get_light_profile_attributes(
        IValue_resource const *t,
        bool                  &valid,
        float                 &power,
        float                 &maximum) const = 0;

    /// Retrieve the attributes of a bsdf measurement.
    ///
    /// \param[in]  t        the bsdf measurement resource
    /// \param[out] valid    the result of bsdf_measurement_is_valid()
    virtual void get_bsdf_measurement_attributes(
        IValue_resource const *t,
        bool                  &valid) const = 0;
};

/// A lambda function used to express MDL expressions that are part of a material.
///
/// With an #mi::mdl::IGenerated_code_dag::IMaterial_instance at hand, compiling a lambda function
/// usually consists of these steps:
///  - Create the object with #mi::mdl::ICode_generator_dag::create_lambda_function().
///  - Set the function name via #mi::mdl::ILambda_function::set_name().
///  - If class compilation was used, add and map all material parameters of the material
///    instance via #mi::mdl::ILambda_function::add_parameter() and
///    #mi::mdl::ILambda_function::set_parameter_mapping().
///  - Walk the material constructor of the material instance to find the MDL expression.
///  - Import the MDL expression via #mi::mdl::ILambda_function::import_expr().
///  - Set the imported DAG node as body via #mi::mdl::ILambda_function::set_body().
///  - Enumerate the resources used by the expression via
///    #mi::mdl::ILambda_function::enumerate_resources() and map them via the map_*_resource()
///    functions.
///  - Compile the lambda function object or add it to a link unit.
///  - If class compilation was used, use the argument block layout of the result to construct an
///    argument block from the parameter default values of the used material instance.
class ILambda_function : public
    mi::base::Interface_declare<0xd2bd2203,0xc7da,0x4bea,0x8d,0x57,0x46,0xbb,0xd6,0x53,0xab,0xe3,
    IDag_builder>
{
public:
    /// The execution context for lambda functions.
    enum Lambda_execution_context {
        LEC_ENVIRONMENT  = 0,   ///< This lambda function will be executed inside the environment.
        LEC_CORE         = 1,   ///< This lambda function will be executed in the renderer core.
        LEC_DISPLACEMENT = 2,   ///< This lambda function will be executed inside displacement.
    };

    /// The result of analyze().
    struct Analysis_result {
        unsigned tangent_spaces;          ///< Bitmap of maybe used tangent spaces.
        unsigned texture_spaces;          ///< Bitmap of maybe used texture_coordinate() spaces.
        unsigned uses_state_normal:1;     ///< Set, if state::normal() may be used.
        unsigned uses_state_rc_normal:1;  ///< Set, if state::rounded_corner_normal() may be used.
        unsigned uses_texresult_lookup:1; ///< Set, if texture result lookups may be used.
    };

public:
    /// Get the type factory of this function.
    virtual IType_factory *get_type_factory() = 0;

    /// Get the value factory of this function.
    virtual IValue_factory *get_value_factory() = 0;

    /// Create a constant.
    /// \param  value       The value of the constant.
    /// \returns            The created constant.
    virtual DAG_constant const *create_constant(IValue const *value) = 0;

    /// Create a call.
    /// \param  name            The absolute name of the called function.
    /// \param  sema            The semantic of the called function.
    /// \param  call_args       The call arguments of the called function.
    /// \param  num_call_args   The number of call arguments.
    /// \param  ret_type        The return type of the called function.
    /// \returns                The created call.
    virtual DAG_node const *create_call(
        char const                    *name,
        IDefinition::Semantics        sema,
        DAG_call::Call_argument const call_args[],
        int                           num_call_args,
        IType const                   *ret_type) = 0;

    /// Create a parameter reference.
    /// \param  type        The type of the parameter
    /// \param  index       The index of the parameter.
    /// \returns            The created parameter reference.
    virtual DAG_parameter const *create_parameter(
        IType const *type,
        int         index) = 0;

    // ----------------- own methods -----------------

    /// Get the body of this function.
    ///
    /// \return The body expression or NULL if this is a switch function.
    virtual DAG_node const *get_body() const = 0;

    /// Set the body of this function.
    ///
    /// \param expr   the body expression
    virtual void set_body(DAG_node const *expr) = 0;

    /// Import (i.e. deep-copy) a DAG expression into this lambda function.
    ///
    /// Any parameters must have been added and mapped via #mi::mdl::ILambda_function::add_parameter
    /// and #mi::mdl::ILambda_function::set_parameter_mapping before calling this function.
    ///
    /// \param expr  the DAG expression to import
    ///
    /// \return the imported expression
    virtual DAG_node const *import_expr(DAG_node const *expr) = 0;

    /// Store a DAG (root) expression and returns an index for it.
    ///
    /// \param expr  the expression to remember, must be owned by this lambda function
    ///
    /// \return the index of this expression
    ///
    /// \note The same index will be assigned to identical (in the sense of CSE) expressions.
    virtual size_t store_root_expr(DAG_node const *expr) = 0;

    /// Remove a root expression.
    ///
    /// \param idx  the index of the root expression to be removed
    ///
    /// \return true on success, false if idx is invalid
    ///
    /// \note The freed index can be reused.
    virtual bool remove_root_expr(size_t idx) = 0;

    /// Run garbage collection AFTER a root expression was removed.
    ///
    /// \returns a cleaned copy or NULL if all was deleted
    virtual ILambda_function *garbage_collection() = 0;

    /// Get the remembered expression for a given index.
    ///
    /// \param idx  the index of the root expression
    virtual DAG_node const *get_root_expr(size_t idx) const = 0;

    /// Get the number of root expressions.
    virtual size_t get_root_expr_count() const = 0;

    /// Enumerate all used texture resources of this lambda function.
    ///
    /// \param resolver    a call name resolver
    /// \param enumerator  the enumerator interface
    /// \param root        if non-NULL, the root expression to enumerate, else enumerate
    ///                    all roots of a switch function
    virtual void enumerate_resources(
        ICall_name_resolver const   &resolver,
        ILambda_resource_enumerator &enumerator,
        DAG_node const              *root = NULL) const = 0;

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
    virtual void map_tex_resource(
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
        int                            depth) = 0;

    /// Register a light profile resource mapping.
    ///
    /// \param res_kind  the kind of the resource (texture or invalid reference)
    /// \param res_url   the URL of the texture resource if any
    /// \param res_tag   the tag of the texture resource
    /// \param idx       the mapped index value representing the resource in a lookup table
    /// \param valid     true if this is a valid resource, false otherwise
    /// \param power     the power of this light profile
    /// \param maximum   the maximum of this light profile
    virtual void map_lp_resource(
        IValue::Kind res_kind,
        char const   *res_url,
        int          res_tag,
        size_t       idx,
        bool         valid,
        float        power,
        float        maximum) = 0;

    /// Register a bsdf measurement resource mapping.
    ///
    /// \param res_kind  the kind of the resource (texture or invalid reference)
    /// \param res_url   the URL of the texture resource if any
    /// \param res_tag   the tag of the texture resource
    /// \param idx       the mapped index value representing the resource in a lookup table
    /// \param valid     true if this is a valid resource, false otherwise
    virtual void map_bm_resource(
        IValue::Kind res_kind,
        char const   *res_url,
        int          res_tag,
        size_t       idx,
        bool         valid) = 0;

    /// Analyze one root of a lambda function.
    ///
    /// \param[in]  proj           the root number
    /// \param[in]  name_resolver  a call name resolver
    /// \param[out] result         the analysis result
    ///
    /// \return true on success, false if proj is out of bounds
    virtual bool analyze(
        size_t                    proj,
        ICall_name_resolver const *name_resolver,
        Analysis_result           &result) const = 0;

    /// Optimize the lambda function.
    ///
    /// \param[in]  name_resolver   a call name resolver for inlining functions
    /// \param[in]  call_evaluator  a call evaluator for handling some intrinsic functions
    virtual void optimize(
        ICall_name_resolver const *name_resolver,
        ICall_evaluator           *call_evaluator) = 0;

    /// Returns true if a switch function was "modified", by adding a new
    /// root expression.
    ///
    /// \param reset  if true, reset the modify flag
    ///
    /// \note Deleting a root expression does not set the modify flag.
    ///       The idea is, that even with deleted entries an already compiled
    ///       function can be reused (some roots will be never called), but
    ///       adding a new root must trigger recompilation.
    virtual bool is_modified(bool reset = true) = 0;

    /// Returns true if a switch function was "modified" by removing a
    /// root expression.
    virtual bool has_dead_code() const = 0;

    /// Pass the uniform context for a given call node.
    ///
    /// \param name_resolver    the call name resolver
    /// \param expr             the lambda expression
    /// \param world_to_object  the world-to-object transformation matrix for this function
    /// \param object_to_world  the object-to-world transformation matrix for this function
    /// \param object_id        the result of state::object_id() for this function
    ///
    /// \return expr if the uniform state is not used, otherwise a modified call
    virtual DAG_node const *set_uniform_context(
        ICall_name_resolver const *name_resolver,
        DAG_node const            *expr,
        Float4_struct const       world_to_object[4],
        Float4_struct const       object_to_world[4],
        int                       object_id) = 0;

    /// Get a "serial version" number of this lambda function.
    ///
    /// The serial number can be used to distinguish different lambda functions.
    /// In is increased, whenever the modified flag was set or a new lambda function
    /// is created.
    virtual unsigned get_serial_number() const = 0;

    /// Set the name of the lambda function.
    ///
    /// The default name is "lambda".
    virtual void set_name(char const *name) = 0;

    /// Get the name of the lambda function.
    virtual char const *get_name() const = 0;

    /// Get the hash value of this lambda function.
    virtual DAG_hash const *get_hash() const = 0;

    /// Returns the number of parameters of this lambda function.
    virtual size_t get_parameter_count() const = 0;

    /// Return the type of the i'th parameter.
    ///
    /// \param i  the parameter index
    virtual IType const *get_parameter_type(size_t i) const = 0;

    /// Return the name of the i'th parameter.
    ///
    /// \param i  the parameter index
    virtual char const *get_parameter_name(size_t i) const = 0;

    /// Add a new "captured" parameter.
    ///
    /// \param type  the parameter type
    /// \param name  the name of the parameter
    ///
    /// \return  the parameter index
    virtual size_t add_parameter(
        IType const *type,
        char const  *name) = 0;

    /// Map material parameter i to lambda parameter j.
    ///
    /// \param i  the material parameter index
    /// \param j  the lambda parameter index
    virtual void set_parameter_mapping(size_t i, size_t j) = 0;

    /// Initialize the derivative information for this lambda function.
    /// This rewrites the body/sub-expressions with derivative types.
    ///
    /// \param resolver  the call name resolver
    virtual void initialize_derivative_infos(ICall_name_resolver const *resolver) = 0;

    /// Returns true, if the attributes in the resource attribute table are valid.
    /// If false, only the indices are valid.
    virtual bool has_resource_attributes() const = 0;

    /// Sets whether the resource attribute table contains valid attributes.
    virtual void set_has_resource_attributes(bool avail) = 0;

    /// Set a tag, version pair for a resource value that might be reachable from this
    /// function.
    ///
    /// \param res_kind        the resource kind
    /// \param res_url         the resource url
    /// \param tag             the tag value
    virtual void set_resource_tag(
        Resource_tag_tuple::Kind const res_kind,
        char const                     *res_url,
        int                            tag) = 0;

    /// Remap a resource value according to the resource map.
    ///
    /// \param r  the resource
    virtual int get_resource_tag(IValue_resource const *r) const = 0;

    /// Get the number of resource map entries.
    virtual size_t get_resource_entries_count() const = 0;

    /// Get the i'th resource table entry.
    virtual Resource_tag_tuple const *get_resource_entry(size_t index) const = 0;
};

/// An interface used to manage the DF and non-DF parts of an MDL material surface.
///
/// With an #mi::mdl::IGenerated_code_dag::IMaterial_instance at hand, compiling a distribution
/// function usually consists of these steps:
///  - Create the object with #mi::mdl::ICode_generator_dag::create_distribution_function().
///  - Get the main lambda function with #mi::mdl::IDistribution_function::get_main_df().
///  - Set the base function name in the main lambda via #mi::mdl::ILambda_function::set_name().
///  - If class compilation was used, add and map all material parameters of the material
///    instance to the main lambda via #mi::mdl::ILambda_function::add_parameter() and
///    #mi::mdl::ILambda_function::set_parameter_mapping().
///  - Import the material constructor of the material instance into the main lambda
///    via #mi::mdl::ILambda_function::import_expr().
///  - Walk the resulting material constructor DAG node to find the distribution function.
///  - Call #mi::mdl::IDistribution_function::initialize().
///  - Enumerate the resources used by the main lambda function and all expression lambda functions
///    of the distribution function object and map them in the main lambda function.
///  - Compile the distribution function object or add it to a link unit.
///  - If class compilation was used, use the argument block layout of the result to construct an
///    argument block from the parameter default values of the used material instance.
class IDistribution_function : public
    mi::base::Interface_declare<0x280ae146,0x7d9d,0x4cf8,0x97,0xa1,0x8d,0x76,0x26,0xc1,0xaa,0xc2,
    mi::base::IInterface>
{
public:
    /// The possible kinds of special lambda functions.
    enum Special_kind {
        SK_INVALID = -1,                 ///< Invalid special kind.
        SK_MATERIAL_IOR = 0,             ///< Lambda function for material.ior.
        SK_MATERIAL_THIN_WALLED,         ///< Lambda function for material.thin_walled.
        SK_MATERIAL_VOLUME_ABSORPTION,   ///< Lambda function for
                                         ///< material.volume.absorption_coefficient.
        SK_MATERIAL_GEOMETRY_NORMAL,     ///< Lambda function for material.geometry.normal.

        SK_NUM_KINDS                     ///< The number of special lambda function kinds.
    };

    /// The possible kinds of error codes.
    enum Error_code {
        EC_NONE,                            ///< No error.
        EC_INVALID_PARAMETERS,              ///< Invalid parameters were provided.
        EC_INVALID_PATH,                    ///< The path could not be resolved with the given
                                            ///< material constructor.
        EC_UNSUPPORTED_DISTRIBUTION_TYPE,   ///< Currently only BSDFs and EDFs are supported.
        EC_UNSUPPORTED_BSDF,                ///< An unsupported BSDF was provided.
        EC_UNSUPPORTED_EDF,                 ///< An unsupported EDF was provided.
    };

    /// Initialize this distribution function object for the given material
    /// with the given distribution function node. Any additionally required
    /// expressions from the material will also be handled.
    /// Any material parameters must already be registered in the main DF lambda at this point.
    /// The DAG nodes must already be owned by the main DF lambda.
    ///
    /// \param material_constructor       the DAG node of the material constructor
    /// \param path                       the path of the distribution function
    /// \param include_geometry_normal    if true, the geometry normal will be handled
    /// \param calc_derivative_infos      if true, derivative information will be calculated
    /// \param allow_double_expr_lambdas  if true, expression lambdas may be created for double
    ///                                   values
    /// \param name_resolver              the call name resolver
    ///
    /// \returns EC_NONE, if initialization was successful, an error code otherwise.
    virtual Error_code initialize(
        DAG_node const            *material_constructor,
        char const                *path,
        bool                       include_geometry_normal,
        bool                       calc_derivative_infos,
        bool                       allow_double_expr_lambdas,
        ICall_name_resolver const *name_resolver) = 0;

    /// Get the main DF function representing a DF DAG call.
    virtual ILambda_function *get_main_df() const = 0;

    /// Add the given expression lambda function to the distribution function.
    /// The index as a decimal string can be used as name in DAG call nodes with the semantics
    /// DS_INTRINSIC_DAG_CALL_LAMBDA to reference these lambda functions.
    ///
    /// \param lambda  the lambda function responsible for calculating an expression
    ///
    /// \returns  the index of the expression lambda function
    virtual size_t add_expr_lambda_function(ILambda_function *lambda) = 0;

    /// Get the expression lambda function for the given index.
    ///
    /// \param index  the index of the expression lambda
    ///
    /// \returns  the requested expression lambda function or NULL, if the index is invalid
    virtual ILambda_function *get_expr_lambda(size_t index) const = 0;

    /// Get the number of expression lambda functions.
    virtual size_t get_expr_lambda_count() const = 0;

    /// Set a special lambda function for getting certain material properties.
    ///
    /// \param kind    the kind of special lambda function to set
    /// \param lambda  the lambda function to associate with this kind
    virtual void set_special_lambda_function(
        Special_kind kind,
        ILambda_function *lambda) = 0;

    /// Get the expression lambda index for the given special lambda function kind.
    ///
    /// \param kind    the kind of special lambda function to get
    ///
    /// \returns  the requested expression lambda index or ~0, if the index is invalid or
    ///           the special lambda function has not been set
    virtual size_t get_special_lambda_function_index(Special_kind kind) const = 0;

    /// Returns the number of distribution function handles referenced by this
    /// distribution function.
    virtual size_t get_df_handle_count() const = 0;

    /// Returns a distribution function handle referenced by this distribution function.
    ///
    /// \param index  the index of the handle to return
    ///
    /// \return the name of the handle, or \c NULL, if the \p index was out of range.
    virtual const char* get_df_handle(size_t index) const = 0;

    /// Set a tag, version pair for a resource value that might be reachable from this
    /// function.
    ///
    /// \param res_kind        the resource kind
    /// \param res_url         the resource url
    /// \param tag             the tag value
    virtual void set_resource_tag(
        Resource_tag_tuple::Kind const res_kind,
        char const                     *res_url,
        int                            tag) = 0;
};

/// A Link unit used by code generators.
class ILink_unit : public
    mi::base::Interface_declare<0x57d00bae,0x9072,0x4fa9,0xb9,0x40,0x93,0x8f,0xa9,0x6c,0xcd,0x86,
    mi::base::IInterface>
{
public:
    /// Add a lambda function to this link unit.
    ///
    /// All resources used by \p lambda must have been registered in \p lambda via
    /// the map_*_resource functions of #mi::mdl::ILambda_function. It is also recommended
    /// to already map the resources of the default arguments of the used material instance.
    ///
    /// \param lambda               the lambda function to compile
    /// \param name_resolver        the call name resolver
    /// \param kind                 the kind of the lambda function
    /// \param arg_block_index      this variable will receive the index of the target argument
    ///                             block used for this lambda function or ~0 if none is used
    /// \param function_index       the index of the callable function in the created target code.
    ///                             This parameter is optional, provide NULL if not required.
    ///
    /// \return true on success
    virtual bool add(
        ILambda_function const                    *lambda,
        ICall_name_resolver const                 *name_resolver,
        IGenerated_code_executable::Function_kind  kind,
        size_t                                    *arg_block_index,
        size_t                                    *function_index) = 0;

    /// Add a distribution function to this link unit.
    ///
    /// Currently only BSDFs are supported.
    /// For a BSDF, it results in four functions, with their names built from the name of the
    /// main DF function of \p dist_func suffixed with \c "_init", \c "_sample", \c "_evaluate"
    /// and \c "_pdf", respectively.
    /// All resources used by the main DF function of \p dist_func and the expression lambda
    /// functions of \p dist_func must have been registered in the main DF function via
    /// the map_*_resource functions of #mi::mdl::ILambda_function. It is also recommended
    /// to already map the resources of the default arguments of the used material instance.
    ///
    /// \param dist_func            the distribution function to compile
    /// \param name_resolver        the call name resolver
    /// \param arg_block_index      this variable will receive the index of the target argument
    ///                             block used for this distribution function or ~0 if none is used
    /// \param function_index       the index of the callable function in the created target code.
    ///                             This parameter is optional, provide NULL if not required.
    ///
    /// \return true on success
    virtual bool add(
        IDistribution_function const  *dist_func,
        ICall_name_resolver const     *name_resolver,
        size_t                        *arg_block_index,
        size_t                        *function_index) = 0;

    /// Get the number of functions in this link unit.
    virtual size_t get_function_count() const = 0;

    /// Get the name of the i'th function inside this link unit.
    ///
    /// \param i  the index of the function
    ///
    /// \return the name of the i'th function or \c NULL if the index is out of bounds
    virtual char const *get_function_name(size_t i) const = 0;

    /// Returns the distribution kind of the i'th function inside this link unit.
    ///
    /// \param i  the index of the function
    ///
    /// \return The distribution kind of the i'th function or \c FK_INVALID if \p i was invalid.
    virtual IGenerated_code_executable::Distribution_kind get_distribution_kind(size_t i) const = 0;

    /// Returns the function kind of the i'th function inside this link unit.
    ///
    /// \param i  the index of the function
    ///
    /// \return The function kind of the i'th function or \c FK_INVALID if \p i was invalid.
    virtual IGenerated_code_executable::Function_kind get_function_kind(size_t i) const = 0;

    /// Get the index of the target argument block layout for the i'th function inside this link
    /// unit if used.
    ///
    /// \param i  the index of the function
    ///
    /// \return The index of the target argument block layout or ~0 if not used or \p i is invalid.
    virtual size_t get_function_arg_block_layout_index(size_t i) const = 0;

    /// Get the number of target argument block layouts used by this link unit.
    virtual size_t get_arg_block_layout_count() const = 0;

    /// Get the i'th target argument block layout used by this link unit.
    ///
    /// \param i  the index of the target argument block layout
    ///
    /// \return The target argument block layout or \c NULL if \p i is invalid.
    virtual IGenerated_code_value_layout const *get_arg_block_layout(size_t i) const = 0;

    /// Access messages.
    virtual Messages const &access_messages() const = 0;

    /// Returns the prototype of the i'th function inside this link unit.
    ///
    /// \param index   the index of the function.
    /// \param lang    the language to use for the prototype.
    ///
    /// \return The prototype or NULL if \p index is out of bounds or \p lang cannot be used
    ///         for this target code.
    virtual char const *get_function_prototype(
        size_t index,
        IGenerated_code_executable::Prototype_language lang) const = 0;
};

/// The DAG code generator interface.
class ICode_generator_dag : public
    mi::base::Interface_declare<0xfe2ef328,0x2fac,0x4da3,0x81,0x9b,0x34,0x25,0xb5,0x4b,0x54,0x6c,
    ICode_generator>
{
public:

    /// The name of the option to set the context name.
    #define MDL_CG_DAG_OPTION_CONTEXT_NAME "context_name"

    /// The name of the option to dump the material expression DAG for every
    /// compiled module in the DAG backend.
    #define MDL_CG_DAG_OPTION_DUMP_MATERIAL_DAG "dump_material_dag"

    /// The name of the option to include local entities called from materials.
    #define MDL_CG_DAG_OPTION_INCLUDE_LOCAL_ENTITIES "include_local_entities"

    /// The name of the option to mark DAG generated entities.
    #define MDL_CG_DAG_OPTION_MARK_DAG_GENERATED "mark_dag_generated"

    /// The name of the option to forbid local functions inside material bodies.
    #define MDL_CG_DAG_OPTION_NO_LOCAL_FUNC_CALLS "no_local_func_calls"

    /// The name of the option that enables unsafe math optimizations.
    #define MDL_CG_DAG_OPTION_UNSAFE_MATH_OPTIMIZATIONS "unsafe_math_optimizations"

    /// The name of the option that exposes names of let expressions as named temporaries.
    #define MDL_CG_DAG_OPTION_EXPOSE_NAMES_OF_LET_EXPRESSIONS "expose_names_of_let_expressions"

    /// Compile a module.
    /// \param      module  The module to compile.
    /// \returns            The generated code.
    virtual IGenerated_code_dag *compile(IModule const *module) = 0;

    /// Create a new MDL lambda function.
    ///
    /// \param context  the execution context for this lambda function.
    ///
    /// \returns  a new lambda function.
    virtual ILambda_function *create_lambda_function(
        ILambda_function::Lambda_execution_context context) = 0;

    /// Create a new MDL distribution function.
    ///
    /// \returns  a new distribution function.
    virtual IDistribution_function *create_distribution_function() = 0;

    /// Serialize a lambda function to the given serializer.
    ///
    /// \param lambda                the lambda function to serialize
    /// \param is                    the serializer data is written to
    virtual void serialize_lambda(
        ILambda_function const *lambda,
        ISerializer            *is) const = 0;

    /// Deserialize a lambda function from a given deserializer.
    ///
    /// \param ds  the deserializer data is read from
    ///
    /// \return the lambda function
    virtual ILambda_function *deserialize_lambda(IDeserializer *ds) = 0;
};


/// The JIT code generator interface.
class ICode_generator_jit : public
    mi::base::Interface_declare<0x059c7e80,0x696c,0x4684,0xad,0x08,0xb7,0x17,0x72,0x5a,0x55,0x20,
    ICode_generator>
{
    /// The name of the option to disable exception handling in the JIT code generator.
    #define MDL_JIT_OPTION_DISABLE_EXCEPTIONS "jit_disable_exceptions"

    /// The name of the option to let the the JIT code generator create a read-only
    /// constant segment.
    #define MDL_JIT_OPTION_ENABLE_RO_SEGMENT "jit_enable_ro_segment"

    /// The name of the option to set the fast-math optimization of the JIT code generator.
    #define MDL_JIT_OPTION_FAST_MATH "jit_fast_math"

    /// The name of the option to include the uniform state in the MDL state.
    #define MDL_JIT_OPTION_INCLUDE_UNIFORM_STATE "jit_include_uniform_state"

    /// The name of the option to steer linking of libdevice.
    #define MDL_JIT_OPTION_LINK_LIBDEVICE "jit_link_libdevice"

    /// The name of the option to steer linking version of libbsdf to be linked.
    #define MDL_JIT_OPTION_LINK_LIBBSDF_DF_HANDLE_SLOT_MODE "jit_link_libbsdf_df_handle_slot_mode"

    /// The name of the option to map strings to IDs.
    #define MDL_JIT_OPTION_MAP_STRINGS_TO_IDS "jit_map_strings_to_ids"

    /// The name of the option to set the optimization level of the JIT code generator.
    #define MDL_JIT_OPTION_OPT_LEVEL "jit_opt_level"

    /// The name of the option to inline functions aggressively.
    #define MDL_JIT_OPTION_INLINE_AGGRESSIVELY "jit_inline_aggressively"

    /// The name of the option that steers the call mode for the GPU texture lookup.
    #define MDL_JIT_OPTION_TEX_LOOKUP_CALL_MODE "jit_tex_lookup_call_mode"

    /// The name of the option stating whether the texture runtime uses derivatives.
    #define MDL_JIT_OPTION_TEX_RUNTIME_WITH_DERIVATIVES "jit_tex_runtime_with_derivs"

    /// The name of the option to use bitangent instead of tangent_u, tangent_v in the MDL state.
    #define MDL_JIT_OPTION_USE_BITANGENT "jit_use_bitangent"

    /// The name of the option to generate auxiliary methods on distribution functions.
    #define MDL_JIT_OPTION_ENABLE_AUXILIARY "jit_enable_auxiliary"

    /// The name of the option to let the the JIT code generator create a LLVM bitcode
    /// instead of LLVM IR (ascii) code.
    #define MDL_JIT_OPTION_WRITE_BITCODE "jit_write_bitcode"

    /// The name of the option to enable/disable the builtin texture runtime of the native backend
    #define MDL_JIT_USE_BUILTIN_RESOURCE_HANDLER_CPU "jit_use_builtin_resource_handler_cpu"

    /// The name of the option to enable the HLSL resource data struct argument.
    #define MDL_JIT_OPTION_HLSL_USE_RESOURCE_DATA "jit_hlsl_use_resource_data"

    /// The name of the option specifying a comma-separated list of names for which scene data
    /// may be available in the renderer.
    /// For names not in the list, scene::data_isvalid will always return false and
    /// the scene::data_lookup_* functions will always return the provided default value.
    /// Use "*" to specify that scene data for any name may be available.
    #define MDL_JIT_OPTION_SCENE_DATA_NAMES "jit_scene_data_names"

    /// The name of the option specifying a comma-separated list of names of functions which will be
    /// visible in the generated code (empty string means no special restriction).
    /// Can especially be used in combination with \c "jit_llvm_renderer_module" to limit
    /// the number of functions for which target code will be generated.
    #define MDL_JIT_OPTION_VISIBLE_FUNCTIONS "jit_visible_functions"

    /// The name of the option to set a user-specified LLVM implementation for the state module.
    #define MDL_JIT_BINOPTION_LLVM_STATE_MODULE "jit_llvm_state_module"

    /// The name of the option to set a user-specified LLVM renderer module.
    #define MDL_JIT_BINOPTION_LLVM_RENDERER_MODULE "jit_llvm_renderer_module"

public:
    /// The compilation mode for whole module compilation.
    enum Compilation_mode {
        CM_NATIVE  = 0,   ///< Compile into native code.
        CM_PTX     = 1,   ///< Compile into PTX assembler.
        CM_HLSL    = 2,   ///< Compile into HLSL code.
        CM_LLVM_IR = 3    ///< Compile into LLVM IR (LLVM 7.0 compatible).
    };

public:
    /// Compile a whole module.
    ///
    /// \param module  The module to compile.
    /// \param mode    The compilation mode
    ///
    /// \note This method is not used currently for code generation, just
    ///       by the unit tests to test various aspects of the code generator.
    ///
    /// \returns The generated code.
    virtual IGenerated_code_executable *compile(
        IModule const    *module,
        Compilation_mode mode) = 0;

    /// Compile a lambda function using the JIT into an environment (shader) of a scene.
    ///
    /// The generated function will have the signature #mi::mdl::Lambda_environment_function.
    ///
    /// \param lambda         the lambda function to compile
    /// \param name_resolver  the call name resolver
    ///
    /// \return the compiled function or NULL on compilation errors
    virtual IGenerated_code_lambda_function *compile_into_environment(
        ILambda_function const    *lambda,
        ICall_name_resolver const *name_resolver) = 0;

    /// Compile a lambda function using the JIT into a constant function.
    ///
    /// The generated function will have the signature #mi::mdl::Lambda_const_function.
    ///
    /// \param lambda           the lambda function to compile
    /// \param name_resolver    the call name resolver
    /// \param attr             an interface to retrieve resource attributes
    /// \param world_to_object  the world-to-object transformation matrix for this function
    /// \param object_to_world  the object-to-world transformation matrix for this function
    /// \param object_id        the result of state::object_id() for this function
    ///
    /// \return the compiled function or NULL on compilation errors
    virtual IGenerated_code_lambda_function *compile_into_const_function(
        ILambda_function const    *lambda,
        ICall_name_resolver const  *name_resolver,
        ILambda_resource_attribute *attr,
        Float4_struct const        world_to_object[4],
        Float4_struct const        object_to_world[4],
        int                        object_id) = 0;

    /// Compile a lambda switch function having several roots using the JIT into a
    /// function computing one of the root expressions.
    ///
    /// The generated function will have the signature #mi::mdl::Lambda_switch_function.
    ///
    /// \param lambda               the lambda function to compile
    /// \param name_resolver        the call name resolver
    /// \param num_texture_spaces   the number of supported texture spaces
    /// \param num_texture_results  the number of texture result entries
    ///
    /// \return the compiled function or NULL on compilation errors
    virtual IGenerated_code_lambda_function *compile_into_switch_function(
        ILambda_function const    *lambda,
        ICall_name_resolver const *name_resolver,
        unsigned                  num_texture_spaces,
        unsigned                  num_texture_results) = 0;

    /// Compile a lambda switch function having several roots using the JIT into a
    /// function computing one of the root expressions for execution on the GPU.
    ///
    /// The generated function will have the signature #mi::mdl::Lambda_switch_function.
    ///
    /// \param code_cache           If non-NULL, a code cache
    /// \param lambda               the lambda function to compile
    /// \param name_resolver        the call name resolver
    /// \param num_texture_spaces   the number of supported texture spaces
    /// \param num_texture_results  the number of texture result entries
    /// \param sm_version           the target architecture of the GPU
    ///
    /// \return the compiled function or NULL on compilation errors
    virtual IGenerated_code_executable *compile_into_switch_function_for_gpu(
        ICode_cache               *code_cache,
        ILambda_function const    *lambda,
        ICall_name_resolver const *name_resolver,
        unsigned                  num_texture_spaces,
        unsigned                  num_texture_results,
        unsigned                  sm_version) = 0;

    /// Compile a lambda function into a generic function using the JIT.
    ///
    /// The generated function will have the signature #mi::mdl::Lambda_generic_function.
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
    virtual IGenerated_code_lambda_function *compile_into_generic_function(
        ILambda_function const    *lambda,
        ICall_name_resolver const *name_resolver,
        unsigned                  num_texture_spaces,
        unsigned                  num_texture_results,
        ILambda_call_transformer  *transformer) = 0;

    /// Compile a lambda function into a LLVM-IR using the JIT.
    ///
    /// The generated function will have the signature #mi::mdl::Lambda_generic_function or
    /// #mi::mdl::Lambda_switch_function depending on the type of the lambda.
    ///
    /// \param lambda               the lambda function to compile
    /// \param name_resolver        the call name resolver
    /// \param num_texture_spaces   the number of supported texture spaces
    /// \param num_texture_results  the number of texture result entries
    /// \param enable_simd          if true, SIMD instructions will be generated
    ///
    /// \return the compiled function or NULL on compilation errors
    virtual IGenerated_code_executable *compile_into_llvm_ir(
        ILambda_function const    *lambda,
        ICall_name_resolver const *name_resolver,
        unsigned                  num_texture_spaces,
        unsigned                  num_texture_results,
        bool                      enable_simd) = 0;

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
        bool                      llvm_ir_output) = 0;

    /// Compile a distribution function into native code using the JIT.
    ///
    /// The generated functions will have the signatures #mi::mdl::Bsdf_init_function,
    /// #mi::mdl::Bsdf_sample_function, #mi::mdl::Bsdf_evaluate_function and
    /// #mi::mdl::Bsdf_pdf_function.
    ///
    /// Currently only BSDFs are supported.
    /// For a BSDF, it results in four functions, with their names built from the name of the
    /// main DF function of \p dist_func suffixed with \c "_init", \c "_sample", \c "_evaluate"
    /// and \c "_pdf", respectively.
    /// All resources used by the main DF function of \p dist_func and the expression lambda
    /// functions of \p dist_func must have been registered in the main DF function via
    /// the map_*_resource functions of #mi::mdl::ILambda_function. It is also recommended
    /// to already map the resources of the default arguments of the used material instance.
    ///
    /// \param dist_func            the distribution function to compile
    /// \param name_resolver        the call name resolver
    /// \param num_texture_spaces   the number of supported texture spaces
    /// \param num_texture_results  the number of texture result entries
    ///
    /// \return the compiled distribution function or NULL on compilation errors
    virtual IGenerated_code_executable *compile_distribution_function_cpu(
        IDistribution_function const *dist_func,
        ICall_name_resolver const    *name_resolver,
        unsigned                     num_texture_spaces,
        unsigned                     num_texture_results) = 0;

    /// Compile a distribution function into PTX or HLSL using the JIT.
    ///
    /// The generated functions will have the signatures #mi::mdl::Bsdf_init_function,
    /// #mi::mdl::Bsdf_sample_function, #mi::mdl::Bsdf_evaluate_function and
    /// #mi::mdl::Bsdf_pdf_function.
    ///
    /// Currently only BSDFs are supported.
    /// For a BSDF, it results in four functions, with their names built from the name of the
    /// main DF function of \p dist_func suffixed with \c "_init", \c "_sample", \c "_evaluate"
    /// and \c "_pdf", respectively.
    /// All resources used by the main DF function of \p dist_func and the expression lambda
    /// functions of \p dist_func must have been registered in the main DF function via
    /// the map_*_resource functions of #mi::mdl::ILambda_function. It is also recommended
    /// to already map the resources of the default arguments of the used material instance.
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
    virtual IGenerated_code_executable *compile_distribution_function_gpu(
        IDistribution_function const *dist_func,
        ICall_name_resolver const    *name_resolver,
        unsigned                     num_texture_spaces,
        unsigned                     num_texture_results,
        unsigned                     sm_version,
        Compilation_mode             comp_mode,
        bool                         llvm_ir_output) = 0;

    /// Get the device library for PTX compilation.
    ///
    /// \param[out] size        the size of the library
    ///
    /// \return the library as LLVM bitcode representation
    virtual unsigned char const *get_libdevice_for_gpu(
        size_t   &size) = 0;

    /// Get the resolution of the libbsdf multi-scattering lookup table data.
    ///
    /// \param bsdf_data_kind   the kind of the BSDF data, has to be a multiscatter kind
    /// \param[out] theta       will contain the number of IOR values when data is available
    /// \param[out] roughness   will contain the number of roughness values when data is available
    /// \param[out] ior         will contain the number of theta values when data is available
    ///
    /// \returns                true if there is data for this semantic (BSDF)
    virtual bool get_libbsdf_multiscatter_data_resolution(
        IValue_texture::Bsdf_data_kind bsdf_data_kind,
        size_t &theta,
        size_t &roughness,
        size_t &ior) const = 0;

    /// Get access to the libbsdf multi-scattering lookup table data.
    ///
    /// \param bsdf_data_kind  the kind of the BSDF data, has to be a multiscatter kind
    /// \param[out] size       the size of the data
    ///
    /// \returns               the lookup data if available for this semantic (BSDF), NULL otherwise
    virtual unsigned char const *get_libbsdf_multiscatter_data(
        IValue_texture::Bsdf_data_kind bsdf_data_kind,
        size_t                         &size) const = 0;

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
    virtual ILink_unit *create_link_unit(
        Compilation_mode  mode,
        bool              enable_simd,
        unsigned          sm_version,
        unsigned          num_texture_spaces,
        unsigned          num_texture_results) = 0;

    /// Compile a link unit into LLVM-IR, PTX or native code using the JIT.
    ///
    /// \param unit  the link unit to compile
    ///
    /// \return the compiled function or NULL on compilation errors
    virtual IGenerated_code_executable *compile_unit(
        ILink_unit const *unit) = 0;

    /// Create a blank layout used for deserialization of target codes.
    virtual IGenerated_code_value_layout *create_value_layout() const = 0;
};

/*!
\page mdl_code_generator_options Options for the MDL code generators

You can configure the MDL code generators by setting options on the #mi::mdl::Options object
returned by #mi::mdl::ICode_generator::access_options().

These options are supported by all MDL code generators:

- \ref mdl_option_cg_internal_space "internal_space"

These options are specific to the MDL DAG code generator:

- \ref mdl_option_dag_context_name            "context_name"
- \ref mdl_option_dag_dump_material_dag       "dump_material_dag"
- \ref mdl_option_dag_include_local_entities  "include_local_entities"
- \ref mdl_option_dag_mark_dag_generated      "mark_dag_generated"
- \ref mdl_option_dag_no_local_func_calls     "no_local_func_calls"

These options are specific to the MDL JIT code generator:

- \ref mdl_option_jit_disable_exceptions      "jit_disable_exceptions"
- \ref mdl_option_jit_enable_ro_segment       "jit_enable_ro_segment"
- \ref mdl_option_jit_fast_math               "jit_fast_math"
- \ref mdl_option_jit_include_uniform_state   "jit_include_uniform_state"
- \ref mdl_option_jit_inline_aggressively     "jit_inline_aggressively"
- \ref mdl_option_jit_link_libdevice          "jit_link_libdevice"
- \ref mdl_option_jit_llvm_state_module       "jit_llvm_state_module"
- \ref mdl_option_jit_llvm_renderer_module    "jit_llvm_renderer_module"
- \ref mdl_option_jit_map_strings_to_ids      "jit_map_strings_to_ids"
- \ref mdl_option_jit_opt_level               "jit_opt_level"
- \ref mdl_option_jit_tex_lookup_call_mode    "jit_tex_lookup_call_mode"
- \ref mdl_option_jit_tex_runtime_with_derivs "jit_tex_runtime_with_derivs"
- \ref mdl_option_jit_use_bitangent           "jit_use_bitangent"*/
/*!
- \ref mdl_option_jit_use_builtin_res_h       "jit_use_builtin_resource_handler_cpu"
- \ref mdl_option_jit_visible_functions       "jit_visible_functions"

\section mdl_cg_options Generic MDL code generator options

\anchor mdl_option_cg_internal_space
- <b>internal_space</b>: Specifies the internal space used by the code generator for
  constant folding.
  Possible values:
  - \c "coordinate_object": internal space is object space
  - \c "coordinate_world": internal space is world space (default)
  - \c "*": object and world space are treated as the same for constant folding

\section mdl_cg_dag_options Specific MDL DAG code generator options

\anchor mdl_option_dag_context_name
- <b>context_name</b>: Specifies the name of the context for error messages, typically
  the name of the renderer that uses the MDL DAG backend.
  Default: \c "renderer"

\anchor mdl_option_dag_dump_material_dag
- <b>dump_material_dag</b>: If set to \c "true", the DAG code generator will dump
  a material expression DAG for every compiled material.
  Default: \c "false"

\anchor mdl_option_dag_include_local_entities
- <b>include_local_entities</b>: If set to \c "true", definitions for local entities used inside
  materials will be added to the DAG. This option is ignored, if the option
  <b>no_local_func_calls</b> is set to \c "true".
  Default: \c "false"

\anchor mdl_option_dag_mark_dag_generated
- <b>mark_dag_generated</b>: If set to \c "true", all DAG backend generated entities (which are not
  valid MDL functions but operators) will be annotated with the \c "::anno::hidden()" annotation.
  Default: \c "true"

\anchor mdl_option_dag_no_local_func_calls
- <b>no_local_func_calls</b>: If set to \c "true", calling local functions (i.e. non-exported
  functions) inside materials is forbidden. This violates the MDL standard, but simplifies the
  implementation of some renderers.
  Default: \c "false"

\section mdl_cg_jit_options Specific MDL JIT code generator options

\anchor mdl_option_jit_disable_exceptions
- <b>jit_disable_exceptions</b>: If set to \c "true", support for exceptions through special runtime
  function calls is disabled. For PTX JIT compilation, this is always disabled.
  Default: \c "false"

\anchor mdl_option_jit_enable_ro_segment
- <b>jit_enable_ro_segment</b>: If set to \c "true", a read-only constant data segment may be
  created to reduce the amount of constant data in source code output.
  Default: \c "false"

\anchor mdl_option_jit_fast_math
- <b>jit_fast_math</b>: If set to \c "true", the JIT code generator enables unsafe
  math optimizations (this corresponds to the \c -ffast-math option of the GCC compiler).
  Default: \c "true"

\anchor mdl_option_jit_inline_aggressively
- <b>jit_inline_aggressively</b>: If set to \c "true", the JIT code generator will add the LLVM
  AlwaysInline attribute to most generated functions to force aggressive inlining.
  Default: \c "false"

\anchor mdl_option_jit_include_uniform_state
- <b>jit_include_uniform_state</b>: If set to \c "true", the uniform state (the world-to-object and
  object-to-world transforms and the object ID) are expected to be part of the renderer provided
  state. Most compile functions and create_link_unit automatically set this to \c "true" if needed,
  so setting this option does not have any effect in most cases.
  Default: \c "false"

\anchor mdl_option_jit_link_libdevice
- <b>jit_link_libdevice</b>: If set to \c "true", a built-in version of CUDA's libdevice will be
  linked before generating PTX code.
  Default: \c "true"

\anchor mdl_option_jit_llvm_renderer_module
- <b>jit_llvm_renderer_module</b>: A binary option which allows you to set a user-defined LLVM
  renderer module which will be linked and optimized together with the generated code.

\anchor mdl_option_jit_llvm_state_module
- <b>jit_llvm_state_module</b>: A binary option which allows you to set a user-defined LLVM
  implementation for the state module.

\anchor mdl_option_jit_map_strings_to_ids
- <b>jit_map_strings_to_ids</b>: If set to \c "true", strings become mapped to 32-bit IDs.
  Default: \c "false"

\anchor mdl_option_jit_opt_level
- <b>jit_opt_level</b>: The optimization level for the JIT code generator.
  Default: \c "2"

\anchor mdl_option_jit_tex_lookup_call_mode
- <b>jit_tex_lookup_call_mode</b>: Specifies the call mode for texture lookup functions on GPU.
  Possible values:
  - \c "vtable": uses a table of texture function pointers provided via the res_data parameter
    of the generated functions (default)
  - \c "direct_call": the texture functions will be called directly (by name) in the PTX code
  - \c "optix_cp": the generated code will contain OptiX \c rtCallableProgramId variables for
    each texture function which you have to set in your application

\anchor mdl_option_jit_tex_runtime_with_derivs
- <b>jit_tex_runtime_with_derivs</b>: If set to \c "true", the generated code will calculate
  derivatives and provide coordinates with derivatives to some texture runtime functions.
  Default: \c "false"

\anchor mdl_option_jit_use_bitangent
- <b>jit_use_bitangent</b>: If set to \c "true", bitangents will be expected in the renderer
  provided state instead of tangent_u and tangent_v vectors.
  Default: \c "false"
*/
/*!
\anchor mdl_option_jit_write_bitcode
- <b>jit_write_bitcode</b>: If set to \c "true", LLVM bitcode will be generated instead of LLVM IR
  assembly text.
  Default: \c "false"

\anchor mdl_option_jit_use_builtin_res_h
- <b>jit_use_builtin_resource_handler_cpu</b>: If set to \c "false", the built-in texture handler
  is not used when running CPU code. Instead, the user needs to provide a vtable of texture
  functions via the tex_data parameter.
  Default: \c "false"

\anchor mdl_option_jit_visible_functions
- <b>jit_visible_functions</b>: Specifies a comma-separated list of names of functions which will be
  visible in the generated code (empty string means no special restriction).
  Can especially be used in combination with \ref mdl_option_jit_llvm_renderer_module to limit
  the number of functions for which target code will be generated.
  Default: \c ""
*/


}  // mdl
}  // mi

#endif

