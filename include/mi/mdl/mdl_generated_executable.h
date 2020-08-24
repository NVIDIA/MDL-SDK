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
/// \file mi/mdl/mdl_generated_executable.h
/// \brief Interfaces for generated (native) executable code and support to execute it
#ifndef MDL_GENERATED_EXECUTABLE_H
#define MDL_GENERATED_EXECUTABLE_H 1

#include <mi/mdl/mdl_types.h>
#include <mi/mdl/mdl_values.h>
#include <mi/mdl/mdl_stdlib_types.h>
#include <mi/mdl/mdl_generated_code.h>
#include <mi/mdl/mdl_target_types.h>

namespace mi {
namespace mdl {

/// A callback interface to allow the user to handle resources when creating new target argument
/// blocks for class-compiled materials when the arguments contain resources not known
/// during compilation.
class IGenerated_code_value_callback
{
public:
    /// Returns the resource index for the given resource usable by the target code resource
    /// handler for the corresponding resource type.
    ///
    /// \param res  the resource
    ///
    /// \returns a resource index or 0 if no resource index can be returned
    virtual unsigned get_resource_index(
        IValue_resource const *res) = 0;

    /// Returns a string identifier for the given string value usable by the target code.
    ///
    /// The value 0 is always the "not known string".
    ///
    /// \param s  the string value
    virtual unsigned get_string_index(
        IValue_string const *s) = 0;
};

/// Represents the layout of an argument value block with support for nested elements.
class IGenerated_code_value_layout : public
    mi::base::Interface_declare<0x957bf9c9,0x8683,0x4fc8,0x8b,0xc1,0x48,0xb0,0x22,0x5e,0xa1,0xde>
{
public:
    /// Structure representing the state during traversal of the nested layout.
    struct State {
        State(unsigned state_offs = 0, unsigned data_offs = 0)
        : m_state_offs(state_offs)
        , m_data_offs(data_offs)
        {}

        unsigned m_state_offs;
        unsigned m_data_offs;
    };

    /// Get the size of the target argument block.
    virtual size_t get_size() const = 0;

    /// Get the number of arguments / elements at the given layout state.
    ///
    /// \param state  The layout state representing the current nesting within the
    ///               argument value block. The default value is used for the top-level.
    virtual size_t get_num_elements(State state = State()) const = 0;

    /// Get the offset, the size and the kind of the argument / element inside the argument
    /// block at the given layout state.
    ///
    /// \param[out]  kind      Receives the kind of the argument.
    /// \param[out]  arg_size  Receives the size of the argument.
    /// \param[in]   state     The layout state representing the current nesting within the
    ///                        argument value block. The default value is used for the top-level.
    ///
    /// \returns the offset of the requested argument / element or \c size_t  (0) if the state
    ///          is invalid.
    virtual size_t get_layout(
        mi::mdl::IValue::Kind &kind,
        size_t                &arg_size,
        State                 state = State()) const = 0;

    /// Get the layout state for the i'th argument / element inside the argument value block
    /// at the given layout state.
    ///
    /// \param i      The index of the argument / element.
    /// \param state  The layout state representing the current nesting within the argument
    ///               value block. The default value is used for the top-level.
    ///
    /// \returns the layout state for the nested element or a state with m_state_offs set to
    ///          \c mi::Uint32(0) if the element is atomic.
    virtual State get_nested_state(
        size_t i,
        State  state = State()) const = 0;

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
    virtual int set_value(
        char                           *block,
        mi::mdl::IValue const          *value,
        IGenerated_code_value_callback *value_callback,
        State                          state = State()) const = 0;
};

/// The base executable code interface.
class IGenerated_code_executable : public
    mi::base::Interface_declare<0x11c439a5,0x3eaf,0x4e48,0x8b,0xd3,0xb2,0x23,0x3e,0x6a,0x79,0x68,
    IGenerated_code>
{
public:
    /// The potential state usage properties.
    enum State_usage_property {
        SU_POSITION              = 1u <<  0,       ///< uses state::position()
        SU_NORMAL                = 1u <<  1,       ///< uses state::normal()
        SU_GEOMETRY_NORMAL       = 1u <<  2,       ///< uses state::geometry_normal()
        SU_MOTION                = 1u <<  3,       ///< uses state::motion()
        SU_TEXTURE_COORDINATE    = 1u <<  4,       ///< uses state::texture_coordinate()
        SU_TEXTURE_TANGENTS      = 1u <<  5,       ///< uses state::texture_tangent_*()
        SU_TANGENT_SPACE         = 1u <<  6,       ///< uses state::tangent_space()
        SU_GEOMETRY_TANGENTS     = 1u <<  7,       ///< uses state::geometry_tangent_*()
        SU_DIRECTION             = 1u <<  8,       ///< uses state::direction()
        SU_ANIMATION_TIME        = 1u <<  9,       ///< uses state::animation_time()
        SU_ROUNDED_CORNER_NORMAL = 1u << 10,       ///< uses state::rounded_corner_normal()

        SU_ALL_VARYING_MASK      = (1u << 11) - 1, ///< set of varying state

        SU_TRANSFORMS            = 1u << 11,       ///< uses uniform state::transform*()
        SU_OBJECT_ID             = 1u << 12,       ///< uses uniform state::object_id()

        SU_ALL_UNIFORM_MASK      = SU_TRANSFORMS | SU_OBJECT_ID, ///< set of uniform state
    }; // can be or'ed

    /// The state usage bitmap type.
    typedef unsigned State_usage;

    /// Possible kinds of distribution functions.
    enum Distribution_kind {
        DK_NONE,
        DK_BSDF,
        DK_HAIR_BSDF,
        DK_EDF,
        DK_INVALID
    };

    /// Possible kinds of functions.
    enum Function_kind {
        FK_INVALID,
        FK_LAMBDA,
        FK_SWITCH_LAMBDA,
        FK_ENVIRONMENT,
        FK_CONST,
        FK_DF_INIT,
        FK_DF_SAMPLE,
        FK_DF_EVALUATE,
        FK_DF_PDF,
        FK_DF_AUXILIARY
    };

    /// Language to use for the function prototype.
    enum Prototype_language {
        PL_CUDA,
        PL_PTX,
        PL_HLSL,
        PL_GLSL,             // \if MDL_SOURCE_RELEASE Reserved\else GLSL\endif.
        PL_NUM_LANGUAGES
    };

    /// Returns the source code of the module if available.
    ///
    /// \param size  will be assigned to the length of the source code
    /// \returns the source code or NULL if no source is available.
    ///
    /// \note The source code might be generated lazily.
    virtual char const *get_source_code(size_t &size) const = 0;

    /// Get the data for the read-only data segment if available.
    ///
    /// \param size  will be assigned to the length of the RO data segment
    /// \returns the data segment or NULL if no RO data segment is available.
    virtual char const *get_ro_data_segment(size_t &size) const = 0;

    /// Get the used state properties of the generated lambda function code.
    virtual State_usage get_state_usage() const = 0;

    /// Get the number of captured argument block layouts.
    virtual size_t get_captured_argument_layouts_count() const = 0;

    /// Get a captured arguments block layout if available.
    ///
    /// \param i   the index of the block layout
    ///
    /// \returns the layout or NULL if no arguments were captured or the index is invalid.
    virtual IGenerated_code_value_layout const *get_captured_arguments_layout(size_t i) const = 0;

    /// Get the number of mapped string constants used inside the generated code.
    virtual size_t get_string_constant_count() const = 0;

    /// Get the mapped string constant for a given id.
    ///
    /// \param id  the string id (as used in the generated code)
    ///
    /// \return the MDL string constant that corresponds to the given id or NULL
    ///         if id is out of range
    ///
    /// \note that the id 0 is ALWAYS mapped to the empty string ""
    virtual char const *get_string_constant(size_t id) const = 0;

    /// Get the number of functions in this executable code.
    virtual size_t get_function_count() const = 0;

    /// Get the name of the i'th function inside this executable code.
    ///
    /// \param i  the index of the function
    ///
    /// \return the name of the i'th function or \c NULL if the index is out of bounds
    virtual char const *get_function_name(size_t i) const = 0;

    /// Returns the distribution kind of the i'th function inside this executable code.
    ///
    /// \param i  the index of the function
    ///
    /// \return The distribution kind of the i'th function or \c FK_INVALID if \p i was invalid.
    virtual Distribution_kind get_distribution_kind(size_t i) const = 0;

    /// Returns the function kind of the i'th function inside this executable code.
    ///
    /// \param i  the index of the function
    ///
    /// \return The function kind of the i'th function or \c FK_INVALID if \p i was invalid.
    virtual Function_kind get_function_kind(size_t i) const = 0;

    /// Get the index of the target argument block layout for the i'th function inside this
    /// executable code if used.
    ///
    /// \param i  the index of the function
    ///
    /// \return The index of the target argument block layout or ~0 if not used or \p i is invalid.
    virtual size_t get_function_arg_block_layout_index(size_t i) const = 0;

    /// Returns the prototype of the i'th function inside this executable code.
    ///
    /// \param index   the index of the function.
    /// \param lang    the language to use for the prototype.
    ///
    /// \return The prototype or NULL or an empty string if \p index is out of bounds or \p lang
    ///         cannot be used for this target code.
    virtual char const *get_function_prototype(size_t index, Prototype_language lang) const = 0;

    /// Set a function prototype for a function.
    ///
    /// \param index      the index of the function
    /// \param lang       the language of the prototype being set
    /// \param prototype  the function prototype
    virtual void set_function_prototype(
        size_t index,
        Prototype_language lang,
        char const *prototype) = 0;

    /// Add a function to the given target code, also registering the function prototypes
    /// applicable for the used backend.
    ///
    /// \param name             the name of the function to add
    /// \param dist_kind        the kind of distribution to add
    /// \param func_kind        the kind of the function to add
    /// \param arg_block_index  the argument block index for this function or ~0 if not used
    /// \param state_usage      the state usage of the function to add
    ///
    /// \returns the function index of the added function
    virtual size_t add_function_info(
        char const *name,
        Distribution_kind dist_kind,
        Function_kind func_kind,
        size_t arg_block_index,
        IGenerated_code_executable::State_usage state_usage) = 0;

    /// Get the number of distribution function handles referenced by a function.
    ///
    /// \param func_index   the index of the function
    ///
    /// \return The number of distribution function handles referenced or \c 0, if the
    ///         function is not a distribution function.
    virtual size_t get_function_df_handle_count(size_t func_index) const = 0;

    /// Get the name of a distribution function handle referenced by a function.
    ///
    /// \param func_index     The index of the function.
    /// \param handle_index   The index of the handle.
    ///
    /// \return The name of the distribution function handle or \c NULL, if the
    ///         function is not a distribution function or \p index is invalid.
    virtual char const *get_function_df_handle(size_t func_index, size_t handle_index) const = 0;

    /// Add the name of a distribution function handle referenced by a function.
    ///
    /// \param func_index     The index of the function.
    /// \param handle_name    The name of the handle.
    ///
    /// \return The index of the added handle, or ~0, if the \p func_index was invalid.
    virtual size_t add_function_df_handle(
        size_t func_index,
        char const *handle_name) = 0;

    /// Get the state properties used by a function.
    ///
    /// \param func_index     The index of the function.
    ///
    /// \return The state usage or 0, if the \p func_index was invalid.
    virtual State_usage get_function_state_usage(size_t func_index) const = 0;
};

/// A handler for MDL runtime exceptions.
///
/// This interface is used to report occurred exceptions inside JIT compiled
/// MDL code. User application can implement it to catch exception conditions.
class IMDL_exception_handler
{
public:
    /// Report an out-of-bounds access from a compiled MDL module.
    ///
    /// \param index  the (wrong) index value
    /// \param bound  the upper bound for the index operation
    /// \param fname  if non-NULL the filename where the OOB occurred
    /// \param line   if non-zero the line where the OOB occurred
    virtual void out_of_bounds(
        int        index,
        size_t     bound,
        char const *fname,
        unsigned   line) = 0;

    /// Report an integer-division-by-zero from a compiled MDL module.
    ///
    /// \param fname  if non-NULL the filename where the OOB occurred
    /// \param line   if non-zero the line where the OOB occurred
    virtual void div_by_zero(
        char const *fname,
        unsigned   line) = 0;
};

/// A resource lookup handler.
///
/// This interface is used in CPU-mode to delegate various resource
/// related operations to user-code.
class IResource_handler
{
public:
    typedef mi::mdl::stdlib::Tex_wrap_mode Tex_wrap_mode;

    typedef struct {
        float val[2];
        float dx[2];
        float dy[2];
    } Deriv_float2;

    typedef mi::mdl::stdlib::Mbsdf_part Mbsdf_part;

    /// Get the number of bytes that must be allocated for a resource object.
    virtual size_t get_data_size() const = 0;

    /// Initializes a texture data helper object from a given texture tag.
    ///
    /// \param data    a 16byte aligned pointer to allocated data of at least
    ///                get_data_size() bytes
    /// \param shape   the texture type shape to initialize
    /// \param tag     the texture tag
    /// \param gamma   the MDL declared gamma value of the texture
    /// \param ctx     a used defined context parameter
    ///
    /// This function should create all necessary helper data for the given texture tag
    /// and store it into the memory storage provided by \c data.
    /// This data will be passed to all tex lookup functions as parameter \c tex_data.
    virtual void tex_init(
        void                       *data,
        IType_texture::Shape       shape,
        unsigned                   tag,
        IValue_texture::gamma_mode gamma,
        void                       *ctx) = 0;

    /// Terminate a texture data helper object.
    ///
    /// \param data   a 16byte aligned pointer to allocated data of at least get_data_size() bytes
    /// \param shape  the texture type shape of the object to terminate
    ///
    /// Clean up the helper object that was created in tex_init() here.
    /// The parameter \c shape could be used to determine the type.
    virtual void tex_term(
        void                 *data,
        IType_texture::Shape shape) = 0;

    /// Handle tex::width(texture_2d, int2) and tex::height(texture_2d, int2)
    ///
    /// \param result    the result of tex::width and tex::height
    /// \param tex_data  the read-only shared texture data pointer
    /// \param uv_tile   uv_tile parameter of tex::width/height
    virtual void tex_resolution_2d(
        int           result[2],
        void const    *tex_data,
        int const     uv_tile[2]) const = 0;

    /// Handle tex::width(texture_*) (not for udim textures).
    ///
    /// \param tex_data  the read-only shared texture data pointer
    ///
    /// \return The width of the texture represented by tex_data.
    virtual int tex_width(
        void const    *tex_data) const = 0;

    /// Handle tex::height(texture_*) (not for udim textures).
    ///
    /// \param tex_data  the read-only shared texture data pointer
    ///
    /// \return The height of the texture represented by tex_data.
    virtual int tex_height(
        void const    *tex_data) const = 0;

    /// Handle tex::depth(texture_*).
    ///
    /// \param tex_data  the read-only shared texture data pointer
    ///
    /// \return The depth of the texture represented by tex_data.
    virtual int tex_depth(
        void const    *tex_data) const = 0;

    /// Handle tex::lookup_float(texture_2d, ...).
    ///
    /// \param tex_data     the read-only shared texture data pointer
    /// \param thread_data  extra per-thread data that was passed to the lambda function
    /// \param coord        coord parameter of tex::lookup_float(texture_2d, ...)
    /// \param wrap_u       wrap_u parameter of tex::lookup_float(texture_2d, ...)
    /// \param wrap_v       wrap_v parameter of tex::lookup_float(texture_2d, ...)
    /// \param crop_u       crop_u parameter of tex::lookup_float(texture_2d, ...)
    /// \param crop_v       crop_u parameter of tex::lookup_float(texture_2d, ...)
    ///
    /// \return the result of tex::lookup_float(texture_2d, ...)
    virtual float tex_lookup_float_2d(
        void const    *tex_data,
        void          *thread_data,
        float const   coord[2],
        Tex_wrap_mode wrap_u,
        Tex_wrap_mode wrap_v,
        float const   crop_u[2],
        float const   crop_v[2]) const = 0;

    /// Handle tex::lookup_float(texture_2d, ...) with derivatives.
    ///
    /// \param tex_data     the read-only shared texture data pointer
    /// \param thread_data  extra per-thread data that was passed to the lambda function
    /// \param coord        coord parameter of tex::lookup_float(texture_2d, ...)
    /// \param wrap_u       wrap_u parameter of tex::lookup_float(texture_2d, ...)
    /// \param wrap_v       wrap_v parameter of tex::lookup_float(texture_2d, ...)
    /// \param crop_u       crop_u parameter of tex::lookup_float(texture_2d, ...)
    /// \param crop_v       crop_u parameter of tex::lookup_float(texture_2d, ...)
    ///
    /// \return the result of tex::lookup_float(texture_2d, ...)
    virtual float tex_lookup_deriv_float_2d(
        void const         *tex_data,
        void               *thread_data,
        Deriv_float2 const *coord,
        Tex_wrap_mode      wrap_u,
        Tex_wrap_mode      wrap_v,
        float const        crop_u[2],
        float const        crop_v[2]) const = 0;

    /// Handle tex::lookup_float(texture_3d, ...).
    ///
    /// \param tex_data     the read-only shared texture data pointer
    /// \param thread_data  extra per-thread data that was passed to the lambda function
    /// \param coord        coord parameter of tex::lookup_float(texture_3d, ...)
    /// \param wrap_u       wrap_u parameter of tex::lookup_float(texture_3d, ...)
    /// \param wrap_v       wrap_v parameter of tex::lookup_float(texture_3d, ...)
    /// \param wrap_w       wrap_w parameter of tex::lookup_float(texture_3d, ...)
    /// \param crop_u       crop_u parameter of tex::lookup_float(texture_3d, ...)
    /// \param crop_v       crop_u parameter of tex::lookup_float(texture_3d, ...)
    /// \param crop_w       crop_w parameter of tex::lookup_float(texture_3d, ...)
    ///
    /// \return the result of tex::lookup_float(texture_3d, ...)
    virtual float tex_lookup_float_3d(
        void const    *tex_data,
        void          *thread_data,
        float const   coord[3],
        Tex_wrap_mode wrap_u,
        Tex_wrap_mode wrap_v,
        Tex_wrap_mode wrap_w,
        float const   crop_u[2],
        float const   crop_v[2],
        float const   crop_w[2]) const = 0;

    /// Handle tex::lookup_float(texture_cube, ...).
    ///
    /// \param tex_data     the read-only shared texture data pointer
    /// \param thread_data  extra per-thread data that was passed to the lambda function
    /// \param coord        coord parameter of tex::lookup_float(texture_cube, ...)
    ///
    /// \return the result of tex::lookup_float(texture_cube, ...)
    virtual float tex_lookup_float_cube(
        void const    *tex_data,
        void          *thread_data,
        float const   coord[3]) const = 0;

    /// Handle tex::lookup_float(texture_ptex, ...).
    ///
    /// \param tex_data     the read-only shared texture data pointer
    /// \param thread_data  extra per-thread data that was passed to the lambda function
    /// \param channel      channel parameter of tex::lookup_float(texture_ptex, ...)
    ///
    /// \return the result of tex::lookup_float(texture_ptex, ...)
    virtual float tex_lookup_float_ptex(
        void const    *tex_data,
        void          *thread_data,
        int           channel) const = 0;

    /// Handle tex::lookup_float2(texture_2d, ...).
    ///
    /// \param result       the result of tex::lookup_float2(texture_2d, ...)
    /// \param tex_data     the read-only shared texture data pointer
    /// \param thread_data  extra per-thread data that was passed to the lambda function
    /// \param coord        coord parameter of tex::lookup_float2(texture_2d, ...)
    /// \param wrap_u       wrap_u parameter of tex::lookup_float2(texture_2d, ...)
    /// \param wrap_v       wrap_v parameter of tex::lookup_float2(texture_2d, ...)
    /// \param crop_u       crop_u parameter of tex::lookup_float2(texture_2d, ...)
    /// \param crop_v       crop_u parameter of tex::lookup_float2(texture_2d, ...)
    virtual void tex_lookup_float2_2d(
        float         result[2],
        void const    *tex_data,
        void          *thread_data,
        float const   coord[2],
        Tex_wrap_mode wrap_u,
        Tex_wrap_mode wrap_v,
        float const   crop_u[2],
        float const   crop_v[2]) const = 0;

    /// Handle tex::lookup_float2(texture_2d, ...) with derivatives.
    ///
    /// \param result       the result of tex::lookup_float2(texture_2d, ...)
    /// \param tex_data     the read-only shared texture data pointer
    /// \param thread_data  extra per-thread data that was passed to the lambda function
    /// \param coord        coord parameter of tex::lookup_float2(texture_2d, ...)
    /// \param wrap_u       wrap_u parameter of tex::lookup_float2(texture_2d, ...)
    /// \param wrap_v       wrap_v parameter of tex::lookup_float2(texture_2d, ...)
    /// \param crop_u       crop_u parameter of tex::lookup_float2(texture_2d, ...)
    /// \param crop_v       crop_u parameter of tex::lookup_float2(texture_2d, ...)
    virtual void tex_lookup_deriv_float2_2d(
        float              result[2],
        void const         *tex_data,
        void               *thread_data,
        Deriv_float2 const *coord,
        Tex_wrap_mode      wrap_u,
        Tex_wrap_mode      wrap_v,
        float const        crop_u[2],
        float const        crop_v[2]) const = 0;

    /// Handle tex::lookup_float2(texture_3d, ...).
    ///
    /// \param result       the result of tex::lookup_float2(texture_3d, ...)
    /// \param tex_data     the read-only shared texture data pointer
    /// \param thread_data  extra per-thread data that was passed to the lambda function
    /// \param coord        coord parameter of tex::lookup_float2(texture_3d, ...)
    /// \param wrap_u       wrap_u parameter of tex::lookup_float2(texture_3d, ...)
    /// \param wrap_v       wrap_v parameter of tex::lookup_float2(texture_3d, ...)
    /// \param wrap_w       wrap_w parameter of tex::lookup_float2(texture_3d, ...)
    /// \param crop_u       crop_u parameter of tex::lookup_float2(texture_3d, ...)
    /// \param crop_v       crop_u parameter of tex::lookup_float2(texture_3d, ...)
    /// \param crop_w       crop_w parameter of tex::lookup_float2(texture_3d, ...)
    virtual void tex_lookup_float2_3d(
        float         result[2],
        void const    *tex_data,
        void          *thread_data,
        float const   coord[3],
        Tex_wrap_mode wrap_u,
        Tex_wrap_mode wrap_v,
        Tex_wrap_mode wrap_w,
        float const   crop_u[2],
        float const   crop_v[2],
        float const   crop_w[2]) const = 0;

    /// Handle tex::lookup_float2(texture_cube, ...).
    ///
    /// \param result       the result of tex::lookup_float2(texture_cube, ...)
    /// \param tex_data     the read-only shared texture data pointer
    /// \param thread_data  extra per-thread data that was passed to the lambda function
    /// \param coord        coord parameter of tex::lookup_float2(texture_cube, ...)
    virtual void tex_lookup_float2_cube(
        float         result[2],
        void const    *tex_data,
        void          *thread_data,
        float const   coord[3]) const = 0;

    /// Handle tex::lookup_float2(texture_ptex, ...).
    ///
    /// \param result       the result of tex::lookup_float2(texture_ptex, ...)
    /// \param tex_data     the read-only shared texture data pointer
    /// \param thread_data  extra per-thread data that was passed to the lambda function
    /// \param channel      channel parameter of tex::lookup_float2(texture_ptex, ...)
    virtual void tex_lookup_float2_ptex(
        float         result[2],
        void const    *tex_data,
        void          *thread_data,
        int           channel) const = 0;

    /// Handle tex::lookup_float3(texture_2d, ...).
    ///
    /// \param result       the result of tex::lookup_float3(texture_2d, ...)
    /// \param tex_data     the read-only shared texture data pointer
    /// \param thread_data  extra per-thread data that was passed to the lambda function
    /// \param coord        coord parameter of tex::lookup_float3(texture_2d, ...)
    /// \param wrap_u       wrap_u parameter of tex::lookup_float3(texture_2d, ...)
    /// \param wrap_v       wrap_v parameter of tex::lookup_float3(texture_2d, ...)
    /// \param crop_u       crop_u parameter of tex::lookup_float3(texture_2d, ...)
    /// \param crop_v       crop_u parameter of tex::lookup_float3(texture_2d, ...)
    virtual void tex_lookup_float3_2d(
        float         result[3],
        void const    *tex_data,
        void          *thread_data,
        float const   coord[2],
        Tex_wrap_mode wrap_u,
        Tex_wrap_mode wrap_v,
        float const   crop_u[2],
        float const   crop_v[2]) const = 0;

    /// Handle tex::lookup_float3(texture_2d, ...) with derivatives.
    ///
    /// \param result       the result of tex::lookup_float3(texture_2d, ...)
    /// \param tex_data     the read-only shared texture data pointer
    /// \param thread_data  extra per-thread data that was passed to the lambda function
    /// \param coord        coord parameter of tex::lookup_float3(texture_2d, ...)
    /// \param wrap_u       wrap_u parameter of tex::lookup_float3(texture_2d, ...)
    /// \param wrap_v       wrap_v parameter of tex::lookup_float3(texture_2d, ...)
    /// \param crop_u       crop_u parameter of tex::lookup_float3(texture_2d, ...)
    /// \param crop_v       crop_u parameter of tex::lookup_float3(texture_2d, ...)
    virtual void tex_lookup_deriv_float3_2d(
        float              result[3],
        void const         *tex_data,
        void               *thread_data,
        Deriv_float2 const *coord,
        Tex_wrap_mode      wrap_u,
        Tex_wrap_mode      wrap_v,
        float const        crop_u[2],
        float const        crop_v[2]) const = 0;

    /// Handle tex::lookup_float3(texture_3d, ...).
    ///
    /// \param result       the result of tex::lookup_float3(texture_3d, ...)
    /// \param tex_data     the read-only shared texture data pointer
    /// \param thread_data  extra per-thread data that was passed to the lambda function
    /// \param coord        coord parameter of tex::lookup_float3(texture_3d, ...)
    /// \param wrap_u       wrap_u parameter of tex::lookup_float3(texture_3d, ...)
    /// \param wrap_v       wrap_v parameter of tex::lookup_float3(texture_3d, ...)
    /// \param wrap_w       wrap_w parameter of tex::lookup_float3(texture_3d, ...)
    /// \param crop_u       crop_u parameter of tex::lookup_float3(texture_3d, ...)
    /// \param crop_v       crop_u parameter of tex::lookup_float3(texture_3d, ...)
    /// \param crop_w       crop_w parameter of tex::lookup_float3(texture_3d, ...)
    virtual void tex_lookup_float3_3d(
        float         result[3],
        void const    *tex_data,
        void          *thread_data,
        float const   coord[3],
        Tex_wrap_mode wrap_u,
        Tex_wrap_mode wrap_v,
        Tex_wrap_mode wrap_w,
        float const   crop_u[2],
        float const   crop_v[2],
        float const   crop_w[2]) const = 0;

    /// Handle tex::lookup_float3(texture_cube, ...).
    ///
    /// \param result       the result of tex::lookup_float3(texture_cube, ...)
    /// \param tex_data     the read-only shared texture data pointer
    /// \param thread_data  extra per-thread data that was passed to the lambda function
    /// \param coord        coord parameter of tex::lookup_float3(texture_cube, ...)
    virtual void tex_lookup_float3_cube(
        float         result[3],
        void const    *tex_data,
        void          *thread_data,
        float const   coord[3]) const = 0;

    /// Handle tex::lookup_float3(texture_ptex, ...).
    ///
    /// \param result       the result of tex::lookup_float3(texture_ptex, ...)
    /// \param tex_data     the read-only shared texture data pointer
    /// \param thread_data  extra per-thread data that was passed to the lambda function
    /// \param channel      coord parameter of tex::lookup_float3(texture_ptex, ...)
    virtual void tex_lookup_float3_ptex(
        float         result[3],
        void const    *tex_data,
        void          *thread_data,
        int           channel) const = 0;

    /// Handle tex::lookup_float4(texture_2d, ...).
    ///
    /// \param result       the result of tex::lookup_float4(texture_2d, ...)
    /// \param tex_data     the read-only shared texture data pointer
    /// \param thread_data  extra per-thread data that was passed to the lambda function
    /// \param coord        coord parameter of tex::lookup_float4(texture_2d, ...)
    /// \param wrap_u       wrap_u parameter of tex::lookup_float4(texture_2d, ...)
    /// \param wrap_v       wrap_v parameter of tex::lookup_float4(texture_2d, ...)
    /// \param crop_u       crop_u parameter of tex::lookup_float4(texture_2d, ...)
    /// \param crop_v       crop_u parameter of tex::lookup_float4(texture_2d, ...)
    virtual void tex_lookup_float4_2d(
        float         result[4],
        void const    *tex_data,
        void          *thread_data,
        float const   coord[2],
        Tex_wrap_mode wrap_u,
        Tex_wrap_mode wrap_v,
        float const   crop_u[2],
        float const   crop_v[2]) const = 0;

    /// Handle tex::lookup_float4(texture_2d, ...) with derivatives.
    ///
    /// \param result       the result of tex::lookup_float4(texture_2d, ...)
    /// \param tex_data     the read-only shared texture data pointer
    /// \param thread_data  extra per-thread data that was passed to the lambda function
    /// \param coord        coord parameter of tex::lookup_float4(texture_2d, ...)
    /// \param wrap_u       wrap_u parameter of tex::lookup_float4(texture_2d, ...)
    /// \param wrap_v       wrap_v parameter of tex::lookup_float4(texture_2d, ...)
    /// \param crop_u       crop_u parameter of tex::lookup_float4(texture_2d, ...)
    /// \param crop_v       crop_u parameter of tex::lookup_float4(texture_2d, ...)
    virtual void tex_lookup_deriv_float4_2d(
        float              result[4],
        void const         *tex_data,
        void               *thread_data,
        Deriv_float2 const *coord,
        Tex_wrap_mode      wrap_u,
        Tex_wrap_mode      wrap_v,
        float const        crop_u[2],
        float const        crop_v[2]) const = 0;

    /// Handle tex::lookup_float4(texture_3d, ...).
    ///
    /// \param result       the result of tex::lookup_float4(texture_3d, ...)
    /// \param tex_data     the read-only shared texture data pointer
    /// \param thread_data  extra per-thread data that was passed to the lambda function
    /// \param coord        coord parameter of tex::lookup_float4(texture_3d, ...)
    /// \param wrap_u       wrap_u parameter of tex::lookup_float4(texture_3d, ...)
    /// \param wrap_v       wrap_v parameter of tex::lookup_float4(texture_3d, ...)
    /// \param wrap_w       wrap_w parameter of tex::lookup_float4(texture_3d, ...)
    /// \param crop_u       crop_u parameter of tex::lookup_float4(texture_3d, ...)
    /// \param crop_v       crop_u parameter of tex::lookup_float4(texture_3d, ...)
    /// \param crop_w       crop_w parameter of tex::lookup_float4(texture_3d, ...)
    virtual void tex_lookup_float4_3d(
        float         result[4],
        void const    *tex_data,
        void          *thread_data,
        float const   coord[3],
        Tex_wrap_mode wrap_u,
        Tex_wrap_mode wrap_v,
        Tex_wrap_mode wrap_w,
        float const   crop_u[2],
        float const   crop_v[2],
        float const   crop_w[2]) const = 0;

    /// Handle tex::lookup_float4(texture_cube, ...).
    ///
    /// \param result       the result of tex::lookup_float4(texture_cube, ...)
    /// \param tex_data     the read-only shared texture data pointer
    /// \param thread_data  extra per-thread data that was passed to the lambda function
    /// \param coord        coord parameter of tex::lookup_float3(texture_cube, ...)
    virtual void tex_lookup_float4_cube(
        float         result[4],
        void const    *tex_data,
        void          *thread_data,
        float const   coord[3]) const = 0;

    /// Handle tex::lookup_float4(texture_ptex, ...).
    ///
    /// \param result       the result of tex::lookup_float4(texture_ptex, ...)
    /// \param tex_data     the read-only shared texture data pointer
    /// \param thread_data  extra per-thread data that was passed to the lambda function
    /// \param channel      channel parameter of tex::lookup_float(texture_ptex, ...)
    virtual void tex_lookup_float4_ptex(
        float         result[4],
        void const    *tex_data,
        void          *thread_data,
        int           channel) const = 0;

    /// Handle tex::lookup_color(texture_2d, ...).
    ///
    /// \param rgb          the result of tex::lookup_color(texture_2d, ...) as RGB
    /// \param tex_data     the read-only shared texture data pointer
    /// \param thread_data  extra per-thread data that was passed to the lambda function
    /// \param coord        coord parameter of tex::lookup_color(texture_2d, ...)
    /// \param wrap_u       wrap_u parameter of tex::lookup_color(texture_2d, ...)
    /// \param wrap_v       wrap_v parameter of tex::lookup_color(texture_2d, ...)
    /// \param crop_u       crop_u parameter of tex::lookup_color(texture_2d, ...)
    /// \param crop_v       crop_u parameter of tex::lookup_color(texture_2d, ...)
    virtual void tex_lookup_color_2d(
        float         rgb[3],
        void const    *tex_data,
        void          *thread_data,
        float const   coord[2],
        Tex_wrap_mode wrap_u,
        Tex_wrap_mode wrap_v,
        float const   crop_u[2],
        float const   crop_v[2]) const = 0;

    /// Handle tex::lookup_color(texture_2d, ...) with derivatives.
    ///
    /// \param rgb          the result of tex::lookup_color(texture_2d, ...) as RGB
    /// \param tex_data     the read-only shared texture data pointer
    /// \param thread_data  extra per-thread data that was passed to the lambda function
    /// \param coord        coord parameter of tex::lookup_color(texture_2d, ...)
    /// \param wrap_u       wrap_u parameter of tex::lookup_color(texture_2d, ...)
    /// \param wrap_v       wrap_v parameter of tex::lookup_color(texture_2d, ...)
    /// \param crop_u       crop_u parameter of tex::lookup_color(texture_2d, ...)
    /// \param crop_v       crop_u parameter of tex::lookup_color(texture_2d, ...)
    virtual void tex_lookup_deriv_color_2d(
        float              rgb[3],
        void const         *tex_data,
        void               *thread_data,
        Deriv_float2 const *coord,
        Tex_wrap_mode      wrap_u,
        Tex_wrap_mode      wrap_v,
        float const        crop_u[2],
        float const        crop_v[2]) const = 0;

    /// Handle tex::lookup_color(texture_3d, ...).
    ///
    /// \param rgb          the result of tex::lookup_color(texture_3d, ...) as RGB
    /// \param tex_data     the read-only shared texture data pointer
    /// \param thread_data  extra per-thread data that was passed to the lambda function
    /// \param coord        coord parameter of tex::lookup_color(texture_3d, ...)
    /// \param wrap_u       wrap_u parameter of tex::lookup_color(texture_3d, ...)
    /// \param wrap_v       wrap_v parameter of tex::lookup_color(texture_3d, ...)
    /// \param wrap_w       wrap_w parameter of tex::lookup_color(texture_3d, ...)
    /// \param crop_u       crop_u parameter of tex::lookup_color(texture_3d, ...)
    /// \param crop_v       crop_u parameter of tex::lookup_color(texture_3d, ...)
    /// \param crop_w       crop_w parameter of tex::lookup_color(texture_3d, ...)
    virtual void tex_lookup_color_3d(
        float         rgb[3],
        void const    *tex_data,
        void          *thread_data,
        float const   coord[3],
        Tex_wrap_mode wrap_u,
        Tex_wrap_mode wrap_v,
        Tex_wrap_mode wrap_w,
        float const   crop_u[2],
        float const   crop_v[2],
        float const   crop_w[2]) const = 0;

    /// Handle tex::lookup_color(texture_cube, ...).
    ///
    /// \param rgb          the result of tex::lookup_color(texture_cube, ...) as RGB
    /// \param tex_data     the read-only shared texture data pointer
    /// \param thread_data  extra per-thread data that was passed to the lambda function
    /// \param coord        coord parameter of tex::lookup_color(texture_cube, ...)
    virtual void tex_lookup_color_cube(
        float         rgb[3],
        void const    *tex_data,
        void          *thread_data,
        float const   coord[3]) const = 0;

    /// Handle tex::lookup_color(texture_ptex, ...).
    ///
    /// \param rgb          the result of tex::lookup_color(texture_ptex, ...) as RGB
    /// \param tex_data     the read-only shared texture data pointer
    /// \param thread_data  extra per-thread data that was passed to the lambda function
    /// \param channel      channel parameter of tex::lookup_color(texture_ptex, ...)
    virtual void tex_lookup_color_ptex(
        float         rgb[3],
        void const    *tex_data,
        void          *thread_data,
        int           channel) const = 0;

    /// Handle tex::texel_float(texture_2d, ...).
    ///
    /// \param tex_data     the read-only shared texture data pointer
    /// \param thread_data  extra per-thread data that was passed to the lambda function
    /// \param coord        coord parameter of tex::texel_float(texture_2d, ...)
    /// \param uv_tile      uv_tile parameter of tex::texel_float(texture_2d, ...)
    ///
    /// \return the result of tex::texel_float(texture_2d, ...)
    virtual float tex_texel_float_2d(
        void const    *tex_data,
        void          *thread_data,
        int const     coord[2],
        int const     uv_tile[2]) const = 0;

    /// Handle tex::texel_float2(texture_2d, ...).
    ///
    /// \param result       the result of tex::texel_float2(texture_2d, ...)
    /// \param tex_data     the read-only shared texture data pointer
    /// \param thread_data  extra per-thread data that was passed to the lambda function
    /// \param coord        coord parameter of tex::texel_float2(texture_2d, ...)
    /// \param uv_tile      uv_tile parameter of tex::texel_float2(texture_2d, ...)
    virtual void tex_texel_float2_2d(
        float         result[2],
        void const    *tex_data,
        void          *thread_data,
        int const     coord[2],
        int const     uv_tile[2]) const = 0;

    /// Handle tex::texel_float3(texture_2d, ...).
    ///
    /// \param result       the result of tex::texel_float3(texture_2d, ...)
    /// \param tex_data     the read-only shared texture data pointer
    /// \param thread_data  extra per-thread data that was passed to the lambda function
    /// \param coord        coord parameter of tex::texel_float3(texture_2d, ...)
    /// \param uv_tile      uv_tile parameter of tex::texel_float3(texture_2d, ...)
    virtual void tex_texel_float3_2d(
        float         result[3],
        void const    *tex_data,
        void          *thread_data,
        int const     coord[2],
        int const     uv_tile[2]) const = 0;

    /// Handle tex::texel_float4(texture_2d, ...).
    ///
    /// \param result       the result of tex::texel_float4(texture_2d, ...)
    /// \param tex_data     the read-only shared texture data pointer
    /// \param thread_data  extra per-thread data that was passed to the lambda function
    /// \param coord        coord parameter of tex::texel_float4(texture_2d, ...)
    /// \param uv_tile      uv_tile parameter of tex::texel_float4(texture_2d, ...)
    virtual void tex_texel_float4_2d(
        float         result[4],
        void const    *tex_data,
        void          *thread_data,
        int const     coord[2],
        int const     uv_tile[2]) const = 0;

    /// Handle tex::texel_color(texture_2d, ...).
    ///
    /// \param rgb          the result of tex::texel_color(texture_2d, ...) as RGB
    /// \param tex_data     the read-only shared texture data pointer
    /// \param thread_data  extra per-thread data that was passed to the lambda function
    /// \param coord        coord parameter of tex::texel_color(texture_2d, ...)
    /// \param uv_tile      uv_tile parameter of tex::texel_color(texture_2d, ...)
    virtual void tex_texel_color_2d(
        float         rgb[3],
        void const    *tex_data,
        void          *thread_data,
        int const     coord[2],
        int const     uv_tile[2]) const = 0;

    /// Handle tex::texel_float(texture_3d, ...).
    ///
    /// \param tex_data     the read-only shared texture data pointer
    /// \param thread_data  extra per-thread data that was passed to the lambda function
    /// \param coord        coord parameter of tex::texel_float(texture_3d, ...)
    ///
    /// \return the result of tex::texel_float(texture_3d, ...)
    virtual float tex_texel_float_3d(
        void const    *tex_data,
        void          *thread_data,
        int const     coord[3]) const = 0;

    /// Handle tex::texel_float2(texture_3d, ...).
    ///
    /// \param result       the result of tex::texel_float2(texture_3d, ...)
    /// \param tex_data     the read-only shared texture data pointer
    /// \param thread_data  extra per-thread data that was passed to the lambda function
    /// \param coord        coord parameter of tex::texel_float2(texture_3d, ...)
    virtual void tex_texel_float2_3d(
        float         result[2],
        void const    *tex_data,
        void          *thread_data,
        int const     coord[3]) const = 0;

    /// Handle tex::texel_float3(texture_3d, ...).
    ///
    /// \param result       the result of tex::texel_float3(texture_3d, ...)
    /// \param tex_data     the read-only shared texture data pointer
    /// \param thread_data  extra per-thread data that was passed to the lambda function
    /// \param coord        coord parameter of tex::texel_float3(texture_3d, ...)
    virtual void tex_texel_float3_3d(
        float         result[3],
        void const    *tex_data,
        void          *thread_data,
        int const     coord[3]) const = 0;

    /// Handle tex::texel_float4(texture_3d, ...).
    ///
    /// \param result       the result of tex::texel_float4(texture_3d, ...)
    /// \param tex_data     the read-only shared texture data pointer
    /// \param thread_data  extra per-thread data that was passed to the lambda function
    /// \param coord        coord parameter of tex::texel_float4(texture_3d, ...)
    virtual void tex_texel_float4_3d(
        float         result[4],
        void const    *tex_data,
        void          *thread_data,
        int const     coord[3]) const = 0;

    /// Handle tex::texel_color(texture_3d, ...).
    ///
    /// \param rgb          the result of tex::texel_color(texture_3d, ...) as RGB
    /// \param tex_data     the read-only shared texture data pointer
    /// \param thread_data  extra per-thread data that was passed to the lambda function
    /// \param coord        coord parameter of tex::texel_color(texture_3d, ...)
    virtual void tex_texel_color_3d(
        float         rgb[3],
        void const    *tex_data,
        void          *thread_data,
        int const     coord[3]) const = 0;

    /// Handle tex::texture_isvalid().
    ///
    /// \param tex_data  the read-only shared texture data pointer
    virtual bool tex_isvalid(
        void const *tex_data) const = 0;

    /// Initializes a light profile data helper object from a given light profile tag.
    ///
    /// \param data    a 16byte aligned pointer to allocated data of at least
    ///                get_data_size() bytes
    /// \param tag     the light profile tag
    /// \param ctx     a used defined context parameter
    ///
    /// This function should create all necessary helper data for the given light profile tag
    /// and store it into the memory storage provided by \c data.
    /// This data will be passed to all light profile attribute functions as parameter \c lp_data.
    virtual void lp_init(
        void     *data,
        unsigned tag,
        void     *ctx) = 0;

    /// Terminate a light profile data helper object.
    ///
    /// \param data   a 16byte aligned pointer to allocated data of at least get_data_size() bytes
    ///
    /// Clean up the helper object that was created in lp_init() here.
    virtual void lp_term(
        void *data) = 0;

    /// Get the light profile power value.
    ///
    /// \param lp_data      the read-only shared light profile data pointer
    /// \param thread_data  extra per-thread data that was passed to the lambda function
    ///
    /// \return The power value of the light profile represented by lp_data.
    virtual float lp_power(
        void const *lp_data,
        void       *thread_data) const = 0;

    /// Get the light profile maximum value.
    ///
    /// \param lp_data      the read-only shared light profile data pointer
    /// \param thread_data  extra per-thread data that was passed to the lambda function
    ///
    /// \return The maximum value of the light profile represented by lp_data.
    virtual float lp_maximum(
        void const *lp_data,
        void       *thread_data) const = 0;

    /// Handle df::light_profile_isvalid().
    ///
    /// \param lp_data  the read-only shared light profile data pointer
    ///
    /// \return True if lp_data represents a valid light profile, false otherwise.
    virtual bool lp_isvalid(
        void const *lp_data) const = 0;

    /// Handle df::light_profile_evaluate(...).
    ///
    /// \param lp_data          the read-only shared resource data pointer
    /// \param thread_data      extra per-thread data that was passed to the lambda function
    /// \param theta_phi        spherical coordinates of the requested direction
    /// \return                 the intensity of the light source at the given direction
    virtual float lp_evaluate(
        void const      *lp_data,
        void            *thread_data,
        const float     theta_phi[2]) const = 0;

    /// Handle df::light_profile_sample(...).
    ///
    /// \param result           sampled theta and phi as well as the pdf
    /// \param lp_data          the read-only shared resource data pointer
    /// \param thread_data      extra per-thread data that was passed to the lambda function
    /// \param xi               set of independent uniformly distributed random value
    virtual void lp_sample(
        float           result[3],
        void const      *lp_data,
        void            *thread_data,
        const float     xi[3]) const = 0;

    /// Handle df::light_profile_pdf(...).
    ///
    /// \param lp_data          the read-only shared resource data pointer
    /// \param thread_data      extra per-thread data that was passed to the lambda function
    /// \param theta_phi        spherical coordinates of the requested direction
    /// \return                 the resulting pdf for a corresponding lookup
    virtual float lp_pdf(
        void const      *lp_data,
        void            *thread_data,
        const float     theta_phi[2]) const = 0;

    /// Initializes a bsdf measurement data helper object from a given bsdf measurement tag.
    ///
    /// \param data    a 16byte aligned pointer to allocated data of at least
    ///                get_data_size() bytes
    /// \param tag     the bsdf measurement tag
    /// \param ctx     a used defined context parameter
    ///
    /// This function should create all necessary helper data for the given bsdf measurement tag
    /// and store it into the memory storage provided by \c data.
    /// This data will be passed to all bsdf measurement attribute functions as
    /// parameter \c lp_data.
    virtual void bm_init(
        void     *data,
        unsigned tag,
        void     *ctx) = 0;

    /// Terminate a bsdf measurement data helper object.
    ///
    /// \param data   a 16byte aligned pointer to allocated data of at least get_data_size() bytes
    ///
    /// Clean up the helper object that was created in mp_init() here.
    virtual void bm_term(
        void *data) = 0;

    /// Handle df::bsdf_measurement_isvalid().
    ///
    /// \param bm_data  the read-only shared bsdf measurement data pointer
    ///
    /// \return True if bm_data represents a valid bsdf measurement, false otherwise.
    virtual bool bm_isvalid(
        void const *bm_data) const = 0;

    /// Handle df::bsdf_measurement_resolution(...).
    ///
    /// \param result    the result of bm::theta_res, bm::phi_res, bm::channels
    /// \param bm_data   the read-only shared resource data pointer
    /// \param part      part of the BSDF that is requested
    virtual void bm_resolution(
        unsigned        result[3],
        void const      *bm_data,
        Mbsdf_part      part) const = 0;

    /// Handle df::bsdf_measurement_evaluate(...).
    ///
    /// \param result           the result of lookup
    /// \param bm_data          the read-only shared resource data pointer
    /// \param thread_data      extra per-thread data that was passed to the lambda function
    /// \param theta_phi_in     spherical coordinates of the incoming direction
    /// \param theta_phi_out    spherical coordinates of the outgoing direction
    /// \param part             part of the BSDF that is requested
    virtual void bm_evaluate(
        float           result[3],
        void const      *bm_data,
        void            *thread_data,
        const float     theta_phi_in[2],
        const float     theta_phi_out[2],
        Mbsdf_part      part) const = 0;

    /// Handle df::bsdf_measurement_sample(...).
    ///
    /// \param result           sampled theta and phi as well as the pdf
    /// \param bm_data          the read-only shared resource data pointer
    /// \param thread_data      extra per-thread data that was passed to the lambda function
    /// \param theta_phi_out    spherical coordinates of the outgoing direction
    /// \param xi               set of independent uniformly distributed random value
    /// \param part             part of the BSDF that is requested
    virtual void bm_sample(
        float           result[3],
        void const      *bm_data,
        void            *thread_data,
        const float     theta_phi_out[2],
        const float     xi[3],
        Mbsdf_part      part) const = 0;

    /// Handle df::bsdf_measurement_pdf(...).
    ///
    /// \param bm_data          the read-only shared resource data pointer
    /// \param thread_data      extra per-thread data that was passed to the lambda function
    /// \param theta_phi_in     spherical coordinates of the incoming direction
    /// \param theta_phi_out    spherical coordinates of the outgoing direction
    /// \param part             part of the BSDF that is requested
    /// \return                 the resulting pdf for a corresponding lookup
    virtual float bm_pdf(
        void const      *bm_data,
        void            *thread_data,
        const float     theta_phi_in[2],
        const float     theta_phi_out[2],
        Mbsdf_part      part) const = 0;

    /// Handle df::bsdf_measurement_albedos(...).
    ///
    /// \param result           maximum (in case of color) albedos for reflection and transmission
    ///                         for the selected direction and globally
    /// \param bm_data          the read-only shared resource data pointer
    /// \param thread_data      extra per-thread data that was passed to the lambda function
    /// \param theta_phi        spherical coordinates of the requested direction
    virtual void bm_albedos(
        float           result[4],
        void const      *bm_data,
        void            *thread_data,
        const float     theta_phi[2]) const = 0;
};

/// Executable code of a compiled lambda function.
///
/// This interface represents a JIT compiled lambda function.
/// lambda function are either compiled for execution on CPU or GPU.
/// If compiled for CPU execution, this interface contains methods to run the compiled
/// code.
/// If compiled for GPU execution only PTX code is provided.
class IGenerated_code_lambda_function : public
    mi::base::Interface_declare<0x7e100527,0x0ae6,0x46e3,0x83,0x7a,0x45,0x84,0x99,0x4f,0xe4,0x22,
    IGenerated_code_executable>
{
public:
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
    virtual void init(
        void                   *ctx,
        IMDL_exception_handler *exc_handler,
        IResource_handler      *res_handler) = 0;

    /// Terminates the resource handling.
    virtual void term() = 0;

    /// Run this compiled lambda functions as an environment function on the CPU.
    ///
    /// \param[in]  index     the index of the function to execute
    /// \param[out] result    out: the result will be written to
    /// \param[in]  state     the state of the shader
    /// \param[in]  tex_data  extra thread data for the texture handler
    ///
    /// \returns false if execution was aborted by runtime error, true otherwise
    ///
    /// Executes a compiled environment function.
    virtual bool run_environment(
        size_t                          index,
        RGB_color                       *result,
        Shading_state_environment const *state,
        void                            *tex_data = NULL) = 0;

    /// Run this compiled lambda functions as a uniform function returning bool on the CPU.
    ///
    /// \param[out] result  the result will be written to
    ///
    /// \returns false if execution was aborted by runtime error, true otherwise
    ///
    /// Executes a compiled uniform function returning bool.
    virtual bool run(bool &result) = 0;

    /// Run this compiled lambda functions as a uniform function returning int on the CPU.
    ///
    /// \param[out] result  the result will be written to
    ///
    /// \returns false if execution was aborted by runtime error, true otherwise
    ///
    /// Executes a compiled uniform function returning int.
    virtual bool run(int &result) = 0;

    /// Run this compiled lambda functions as a uniform function returning unsigned on the CPU.
    ///
    /// \param[out] result  the result will be written to
    ///
    /// \returns false if execution was aborted by runtime error, true otherwise
    ///
    /// Executes a compiled uniform function returning unsigned.
    virtual bool run(unsigned &result) = 0;

    /// Run this compiled lambda functions as a uniform function returning float on the CPU.
    ///
    /// \param[out] result  the result will be written to
    ///
    /// \returns false if execution was aborted by runtime error, true otherwise
    ///
    /// Executes a compiled uniform function returning float.
    virtual bool run(float &result) = 0;

    /// Run this compiled lambda functions as a uniform function returning float2 on the CPU.
    ///
    /// \param[out] result  the result will be written to
    ///
    /// \returns false if execution was aborted by runtime error, true otherwise
    ///
    /// Executes a compiled uniform function returning float2.
    virtual bool run(Float2_struct &result) = 0;

    /// Run this compiled lambda functions as a uniform function returning float3 on the CPU.
    ///
    /// \param[out] result  the result will be written to
    ///
    /// \returns false if execution was aborted by runtime error, true otherwise
    ///
    /// Executes a compiled uniform function returning float3.
    virtual bool run(Float3_struct &result) = 0;

    /// Run this compiled lambda functions as a uniform function returning float4 on the CPU.
    ///
    /// \param[out] result  the result will be written to
    ///
    /// \returns false if execution was aborted by runtime error, true otherwise
    ///
    /// Executes a compiled uniform function returning float4.
    virtual bool run(Float4_struct &result) = 0;

    /// Run this compiled lambda functions as a uniform function returning float3x3 on the CPU.
    ///
    /// \param[out] result  the result will be written to
    ///
    /// \returns false if execution was aborted by runtime error, true otherwise
    ///
    /// Executes a compiled uniform function returning float3x3.
    virtual bool run(Matrix3x3_struct &result) = 0;

    /// Run this compiled lambda functions as a uniform function returning float4x4 on the CPU.
    ///
    /// \param[out] result  the result will be written to
    ///
    /// \returns false if execution was aborted by runtime error, true otherwise
    ///
    /// Executes a compiled uniform function returning float4x4.
    virtual bool run(Matrix4x4_struct &result) = 0;

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
    virtual bool run(char const *&result) = 0;

    /// Run this compiled lambda switch function on the CPU.
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
    virtual bool run_core(
        unsigned                     proj,
        Float3_struct                &result,
        Shading_state_material const *state,
        void                         *tex_data,
        void const                   *cap_args) = 0;

    /// Run a compiled lambda switch function on the CPU.
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
    ///       It can be used as a replacement to run_core() if this funciton is NOT a
    ///       switch function.
    virtual bool run_generic(
        size_t                       index,
        void                         *result,
        Shading_state_material const *state,
        void                         *tex_data,
        void const                   *cap_args) = 0;

    /// Run a compiled init function on the CPU. This may modify the texture results buffer
    /// of the given state and the normal field.
    ///
    /// \param[in]  index     the index of the function to execute
    /// \param[in]  state     the core state
    /// \param[in]  tex_data  extra thread data for the texture handler
    /// \param[in]  cap_args  the captured arguments block, if arguments were captured
    ///
    /// \returns false if execution was aborted by runtime error, true otherwise
    virtual bool run_init(
        size_t                 index,
        Shading_state_material *state,
        void                   *tex_data,
        void const             *cap_args) = 0;

    /// Returns the index of the given resource for use as an parameter to a resource-related
    /// function in the generated CPU code.
    ///
    /// \param tag  the resource tag
    ///
    /// \returns the resource index or 0 if the resource is unknown.
    virtual unsigned get_known_resource_index(unsigned tag) const = 0;
};

} // mdl
} // mi

#endif

