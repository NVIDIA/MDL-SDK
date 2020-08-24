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
/// \file mi/mdl/mdl_generated_dag.h
/// \brief Interfaces for the MDL DAG Intermediate Representation.
#ifndef MDL_GENERATED_DAG_H
#define MDL_GENERATED_DAG_H 1

#include <cstring>
#include <mi/mdl/mdl_iowned.h>
#include <mi/mdl/mdl_generated_code.h>
#include <mi/mdl/mdl_values.h>

namespace mi {
namespace mdl {

class ICall_name_resolver;
class IResource_modifier;
class IValue;
class IValue_float;
class IValue_resource;
class IValue_factory;
class Messages;
class IType_factory;

/// A hash value.
///
/// Hash values are used inside the MDL Core library to identify changes
/// inside (re-)compiled material instances.
class DAG_hash {
public:
    /// Default constructor.
    DAG_hash() {
        memset(hash, 0, sizeof(hash));
    }

    /// Constructor from an array of 16 bytes.
    ///
    /// \param arr  an array of 16 bytes
    explicit DAG_hash(unsigned char const (&arr)[16])
    {
        memcpy(hash, arr, sizeof(hash));
    }

    /// Compare two hashes for equality.
    ///
    /// \param other  the other hash
    bool operator==(DAG_hash const &other) const {
        return memcmp(hash, other.hash, sizeof(hash)) == 0;
    }

    /// Compare two hashes for non-equality.
    ///
    /// \param other  the other hash
    bool operator!=(DAG_hash const &other) const {
        return memcmp(hash, other.hash, sizeof(hash)) != 0;
    }

    /// Compare two hashes for less-than.
    ///
    /// \param other  the other hash
    bool operator<(DAG_hash const &other) const {
        return memcmp(hash, other.hash, sizeof(hash)) < 0;
    }

    /// Compare two hashes for less-than or equal.
    ///
    /// \param other  the other hash
    bool operator<=(DAG_hash const &other) const {
        return memcmp(hash, other.hash, sizeof(hash)) <= 0;
    }

    /// Compare two hashes for greater-than.
    ///
    /// \param other  the other hash
    bool operator>(DAG_hash const &other) const {
        return memcmp(hash, other.hash, sizeof(hash)) > 0;
    }

    /// Compare two hashes for greater-than or equal.
    ///
    /// \param other  the other hash
    bool operator>=(DAG_hash const &other) const {
        return memcmp(hash, other.hash, sizeof(hash)) >= 0;
    }

    /// Read hash bytes.
    ///
    /// \param n  index of the byte to access
    unsigned char operator[](size_t n) const {
        if (n < 16)
            return hash[n];
        return 0;
    }

    /// Access raw data.
    ///
    /// \return a pointer to the raw 16 hash bytes
    unsigned char *data() {
        return hash;
    }

    /// Read-only access raw data.
    ///
    /// \return a read-only pointer to the raw 16 hash bytes
    unsigned char const *data() const {
        return hash;
    }

    /// Get the size of the raw data.
    static size_t size() { return 16; }

private:
    unsigned char hash[16];
};

/// A node inside the DAG Intermediate Representation.
class DAG_node : public Interface_owned {
public:
    /// The possible kinds of DAG IR nodes.
    enum Kind {
        EK_CONSTANT,  ///< A constant.
        EK_TEMPORARY, ///< A temporary.
        EK_CALL,      ///< A call.
        EK_PARAMETER  ///< A parameter. 
    };

    /// Get the kind of this DAG IR node.
    virtual Kind get_kind() const = 0;

    /// Get the type of this DAG IR node.
    virtual IType const *get_type() const = 0;

    /// Get the unique ID of this DAG IR node.
    ///
    /// \note Only nodes created by the same factory have unique IDs, do not
    ///       compare nodes from different factories.
    virtual size_t get_id() const = 0;
};

/// A DAG IR constant.
///
/// This node represents constants inside the DAG Intermediate Representation.
class DAG_constant : public DAG_node {
public:
    /// The kind of this subclass.
    static Kind const s_kind = EK_CONSTANT;

    /// Get the value of the constant.
    virtual IValue const *get_value() const = 0;
};

/// A DAG IR temporary reference.
///
/// This node represents temporaries (aka nodes with a "name") inside the
/// DAG Intermediate Representation.
class DAG_temporary : public DAG_node {
public:
    /// The kind of this subclass.
    static Kind const s_kind = EK_TEMPORARY;

    /// Get the index (the "name") of the referenced temporary.
    virtual int get_index() const = 0;

    /// Get the node of the temporary.
    virtual DAG_node const *get_expr() const = 0;
};

/// A DAG IR call.
///
/// This node represents calls to callable entities inside the
/// DAG Intermediate Representation.
class DAG_call : public DAG_node {
public:
    /// A simple value helper class, a pair of an argument expression and a parameter name.
    struct Call_argument {
        /// Default constructor.
        Call_argument() {
            arg = NULL;
            param_name = NULL;
        }

        /// Constructor.
        ///
        /// \param arg         The call argument expression.
        /// \param param_name  The name of the parameter.
        Call_argument(
            DAG_node const *arg,
            char const     *param_name)
        : arg(arg)
        , param_name(param_name)
        {}

        DAG_node const *arg;        ///< The call argument expression.
        char const     *param_name; ///< The name of the parameter.
    };

public:
    /// The kind of this subclass.
    static Kind const s_kind = EK_CALL;

    /// Get the signature of the called function.
    ///
    /// \returns            The signature of the function.
    virtual char const *get_name() const = 0;

    /// Get the number of arguments.
    ///
    /// \returns            The number of arguments.
    virtual int get_argument_count() const = 0;

    /// Get the name of the parameter corresponding to the argument at position index.
    ///
    /// \param index        The index of the argument.
    /// \returns            The name of the parameter.
    virtual char const *get_parameter_name(int index) const = 0;

    /// Get the argument at position index.
    ///
    /// \param index        The index of the argument.
    /// \returns            The argument expression.
    virtual DAG_node const *get_argument(int index) const = 0;

    /// Get the argument for parameter name.
    ///
    /// \param name         The name of the parameter corresponding to the argument.
    /// \returns            The argument expression or NULL if no argument with given
    ///                     name exists.
    virtual DAG_node const *get_argument(char const *name) const = 0;

    /// Get the semantic of the called function if known.
    virtual IDefinition::Semantics get_semantic() const = 0;

    /// Set the argument expression of a call.
    ///
    /// \param index  the index of the argument expression to set
    /// \param arg    the new argument
    virtual void set_argument(
        int            index,
        DAG_node const *arg) = 0;

    /// Get the name hash.
    virtual size_t get_name_hash() const = 0;
};

/// A DAG IR parameter reference.
///
/// This node represents a reference to a parameter of the entity that is represented
/// by the DAG IR.
class DAG_parameter : public DAG_node {
public:
    /// The kind of this subclass.
    static Kind const s_kind = EK_PARAMETER;

    /// Get the index of the referenced parameter.
    virtual int get_index() const = 0;
};

/// An interface to interrogate tag values for resource values.
class IResource_tagger : public Interface_owned {
public:
    /// Get a tag,for a resource constant that might be reachable from this DAG.
    ///
    /// \param res             a resource
    virtual int get_resource_tag(
        IValue_resource const *res) const = 0;
};

/// A Builder for DAG graphs.
///
/// This interface is implemented by various entities that can build DAG IR
/// representations.
///
/// Note that there is no way to create temporaries here. Temporaries are
/// not meant to be created by a user, but are automatically created by the
/// DAG IR optimizer if requested.
class IDag_builder : public
    mi::base::Interface_declare
    <0x01dbe5fb,0xa13d,0x42eb,0xbe,0x3e,0x0a,0x1d,0x2c,0x53,0x11,0xdf,
    mi::base::IInterface>
{
public:
    /// Get the type factory of this instance.
    virtual IType_factory *get_type_factory() = 0;

    /// Get the value factory of this instance.
    virtual IValue_factory *get_value_factory() = 0;

    /// Create a constant.
    ///
    /// \param value       The value of the constant.
    /// \returns           The created constant.
    virtual DAG_constant const *create_constant(IValue const *value) = 0;

    /// Create a call.
    ///
    /// \param signature       The signature of the called function.
    /// \param sema            The semantic of the called function.
    /// \param call_args       The call arguments of the called function.
    /// \param num_call_args   The number of call arguments.
    /// \param ret_type        The return type of the called function.
    /// \returns               The created call.
    virtual DAG_node const *create_call(
        char const                    *signature,
        IDefinition::Semantics        sema,
        DAG_call::Call_argument const call_args[],
        int                           num_call_args,
        IType const                   *ret_type) = 0;

    /// Create a parameter reference.
    ///
    /// \param type        The type of the parameter
    /// \param index       The index of the parameter.
    /// \returns           The created parameter reference.
    virtual DAG_parameter const *create_parameter(
        IType const *type,
        int         index) = 0;
};

/// A Helper interface to do renderer specific constant folding.
///
/// This interface is called by the DAG IR optimizer on functions that
/// have a known semantics but cannot be folded in general. A renderer might have
/// additional knowledge that allows folding.
///
/// One example is tex::width(). A renderer might already know the width of a
/// texture, so it could replace the runtime call by a constant.
class ICall_evaluator {
public:
    /// Check whether evaluate_intrinsic_function() should be called for an unhandled
    /// intrinsic functions with the given semantic.
    ///
    /// \param sema  the semantic to check for
    virtual bool is_evaluate_intrinsic_function_enabled(
        IDefinition::Semantics sema) const = 0;

    /// Evaluate an intrinsic function.
    ///
    /// \param value_factory  The value factory to create values.
    /// \param sema           The semantic of the intrinsic function to evaluate.
    /// \param arguments      The arguments of the function.
    /// \param n_arguments    The number of arguments.
    ///
    /// \returns the result of the function call or IValue_bad.
    virtual IValue const *evaluate_intrinsic_function(
        IValue_factory         *value_factory,
        IDefinition::Semantics sema,
        IValue const * const   arguments[],
        size_t                 n_arguments) const = 0;
};

/// A container of DAG representations of a module containing materials, functions, constants,
/// types and module annotations.
///
/// This object can be generated via ICode_generator_dag::compile() from a module
/// loaded via IMDL::load_module().
class IGenerated_code_dag : public
    mi::base::Interface_declare<0x9dbfd12e,0x8207,0x4a47,0x8c,0x1f,0x5f,0x9a,0xc4,0x9b,0x00,0xf6,
    IGenerated_code>
{
public:
    /// Properties of DAG functions.
    enum Function_property {
        FP_ALLOW_INLINE      = 0, ///< True, if it is legal to inline this function.
        FP_IS_EXPORTED       = 1, ///< True, if this function is exported.
        FP_USES_TEXTURES     = 2, ///< True, if this function uses the texture functions (either
                                  ///  directly or by calling another function that uses textures).
        FP_CAN_THROW_BOUNDS  = 3, ///< True, if this function can throw a out-of-bounds exception.
        FP_CAN_THROW_DIVZERO = 4, ///< True, if this function can throw a div-by-zero exception.
        FP_IS_UNIFORM        = 5, ///< True, if this function is uniform.
        FP_IS_NATIVE         = 6, ///< True, if this function was declared native.
    };

    /// Properties of DAG annotations.
    enum Annotation_property {
        AP_IS_EXPORTED       = 1, ///< True, if this annotation is exported.
    };

    /// The node factory for DAG IR nodes.
    class DAG_node_factory
    {
    public:
        /// Create a constant.
        ///
        /// \param  value       The value of the constant.
        /// \returns            The created constant.
        virtual DAG_constant const *create_constant(IValue const *value) = 0;

        /// Create a temporary reference.
        ///
        /// \param node         The DAG node that is "named" by this temporary.
        /// \param index        The index of the temporary.
        /// \returns            The created temporary reference.
        virtual DAG_temporary const *create_temporary(DAG_node const *node, int index)  = 0;

        /// Create a call.
        ///
        /// \param signature       The signature of the called function.
        /// \param sema            The semantics of the called function.
        /// \param call_args       The call arguments of the called function.
        /// \param num_call_args   The number of call arguments.
        /// \param ret_type        The return type of the function.
        /// \returns               The created call or an equivalent expression.
        virtual DAG_node const *create_call(
            char const                    *signature,
            IDefinition::Semantics        sema,
            DAG_call::Call_argument const call_args[],
            int                           num_call_args,
            IType const                   *ret_type) = 0;

        /// Create a parameter reference.
        ///
        /// \param type        The type of the parameter
        /// \param index       The index of the parameter.
        /// \returns           The created parameter reference.
        virtual DAG_parameter const *create_parameter(IType const *type, int index) = 0;

        /// Get the type factory associated with this expression factory.
        ///
        /// \returns            The type factory.
        virtual IType_factory *get_type_factory() = 0;

        /// Get the value factory associated with this expression factory.
        ///
        /// \returns            The value factory.
        virtual IValue_factory *get_value_factory() = 0;
    };

    static char const MESSAGE_CLASS = 'C';

    /// The possible error codes.
    enum Error_code {
        EC_NONE,                   ///< No error has occurred.
        EC_INVALID_INDEX,          ///< The given index is invalid.
        EC_MATERIAL_HAS_ERROR,     ///< The material cannot be instantiated because it has errors.
        EC_TOO_FEW_ARGUMENTS,      ///< Not enough arguments were supplied for the material.
        EC_TOO_MANY_ARGUMENTS,     ///< Too many arguments were supplied for the material.
        EC_INSTANTIATION_ERROR,    ///< An error occurred during instantiation.
        EC_ARGUMENT_TYPE_MISMATCH, ///< An instance argument is of wrong type.
        EC_WRONG_TRANSMISSION_ON_THIN_WALLED,  ///< Different transmission on thin_walled material.
    };

    /// An instantiated material.
    ///
    /// With an IGenerated_code_dag at hand, creating an instantiated material usually
    /// consists of these steps:
    ///  - Create the object with IGenerated_code_dag::create_material_instance().
    ///  - Build the argument list using the IValue_factory retrieved from
    ///    IGenerated_code_dag::IMaterial_instance::get_value_factory().
    ///    Material parameter defaults retrieved from
    ///    IGenerated_code_dag::get_material_parameter_default() can be used directly.
    ///  - Call IGenerated_code_dag::IMaterial_instance::initialize().
    class IMaterial_instance : public
        mi::base::Interface_declare
        <0x29c36255,0x7558,0x4865,0xa7,0x7e,0xaa,0x3a,0x50,0x4f,0x70,0xbc,
        IDag_builder>
    {
    public:
        /// Instantiation flags.
        enum Flags {
            INSTANCE_COMPILATION = 0 << 0,  ///< Do an instance compilation, default.
            CLASS_COMPILATION    = 1 << 0,  ///< Do a class compilation.
            NO_ARGUMENT_INLINE   = 1 << 1,  ///< CLASS_COMPILATION: Do not inline any arguments.
            NO_RESOURCE_SHARING  = 1 << 2,  ///< CLASS_COMPILATION: Do not share resource arguments.
            NO_STRING_PARAMS     = 1 << 3,  ///< CLASS_COMPILATION: Do not create string parameters.
            NO_TERNARY_ON_DF     = 1 << 4,  ///< CLASS_COMPILATION: Do not allow ?: on df.
            NO_BOOL_PARAMS       = 1 << 5,  ///< CLASS_COMPILATION: Do not create bool parameters.
            NO_ENUM_PARAMS       = 1 << 6,  ///< CLASS_COMPILATION: Do not create enum parameters.
            /// CLASS_COMPILATION: Do not create a parameter for geometry.cutout_opacity if its
            /// value is constant 0.0f or 1.0f.
            NO_TRIVIAL_CUTOUT_OPACITY = 1 << 7,
            /// CLASS_COMPILATION: Do not create layering calls for transparent layers, i.e., with
            /// weight 0.0f.
            NO_TRANSPARENT_LAYERS     = 1 << 8,

            DEFAULT_CLASS_COMPILATION =  ///< Do class compilation with default flags.
                CLASS_COMPILATION |
                NO_ARGUMENT_INLINE |
                NO_RESOURCE_SHARING
        };

        /// Material slots on which hashes are calculated.
        enum Slot {
            MS_THIN_WALLED,                     ///< .thin_walled

            MS_SURFACE_BSDF_SCATTERING,         ///< .surface.scattering
            MS_SURFACE_EMISSION_EDF_EMISSION,   ///< .surface.emission.emission
            MS_SURFACE_EMISSION_INTENSITY,      ///< .surface.emission.intensity

            MS_BACKFACE_BSDF_SCATTERING,        ///< .backface.scattering
            MS_BACKFACE_EMISSION_EDF_EMISSION,  ///< .backface.emission.emission
            MS_BACKFACE_EMISSION_INTENSITY,     ///< .backface.emission.intensity

            MS_IOR,                             ///< .ior

            MS_VOLUME_VDF_SCATTERING,           ///< .volume.scattering
            MS_VOLUME_ABSORPTION_COEFFICIENT,   ///< .volume.absorption_coefficient
            MS_VOLUME_SCATTERING_COEFFICIENT,   ///< .volume.scattering_coefficient

            MS_GEOMETRY_DISPLACEMENT,           ///< .geometry.displacement
            MS_GEOMETRY_CUTOUT_OPACITY,         ///< .geometry.cutout_opacity
            MS_GEOMETRY_NORMAL,                 ///< .geometry.normal

            MS_HAIR,                            ///< .hair

            MS_LAST = MS_HAIR
        };

        /// Property flags of an instance.
        enum Property {
            IP_DEPENDS_ON_TRANSFORM           = 0x01,   ///< depends on object transforms
            IP_DEPENDS_ON_OBJECT_ID           = 0x02,   ///< depends of the object id
            IP_DEPENDS_ON_GLOBAL_DISTRIBUTION = 0x04,   ///< depends on global distribution (edf)
            IP_USES_TERNARY_OPERATOR          = 0x08,   ///< uses the ternary operator '?:'
            IP_USES_TERNARY_OPERATOR_ON_DF    = 0x10,   ///< uses the ternary operator '?:' on *df
            IP_CLASS_COMPILED                 = 0x20,   ///< was class compiled
            IP_DISTILLED                      = 0x40,   ///< was created by the distiller
            IP_DEPENDS_ON_UNIFORM_SCENE_DATA  = 0x80,   ///< depends on uniform scene data
        };

        /// Opacity of an instance.
        enum Opacity {
            OPACITY_OPAQUE,             ///< opaque for sure
            OPACITY_TRANSPARENT,        ///< transparent for sure
            OPACITY_UNKNOWN             ///< opacity unknown (depends on parameter
                                        ///  or complex user expression)
        };

        typedef unsigned Properties;

    public:
        // ----------------- from IDAG_builder -----------------

        /// Get the type factory of this instance.
        ///
        /// Use this factory to create or import types owned by this instance.
        virtual IType_factory *get_type_factory() = 0;

        /// Get the value factory of this instance.
        ///
        /// Use this factory to create or import values owned by this instance.
        virtual IValue_factory *get_value_factory() = 0;

        /// Create a constant node.
        ///
        /// \param value       The value of the constant.
        /// \returns           The created constant.
        ///
        /// \note Use this method to create arguments of the instance.
        virtual DAG_constant const *create_constant(IValue const *value) = 0;

        /// Create a call node.
        ///
        /// \param signature       The signature of the called function.
        /// \param sema            The semantic of the called function.
        /// \param call_args       The call arguments of the called function.
        /// \param num_call_args   The number of call arguments.
        /// \param ret_type        The return type of the called function.
        /// \returns               The created call.
        ///
        /// \note Use this method to create arguments of the instance.
        virtual DAG_node const *create_call(
            char const                    *signature,
            IDefinition::Semantics        sema,
            DAG_call::Call_argument const call_args[],
            int                           num_call_args,
            IType const                   *ret_type) = 0;

        /// Create a parameter reference node.
        ///
        /// \param type        The type of the parameter
        /// \param index       The index of the parameter.
        /// \returns           The created parameter reference.
        virtual DAG_parameter const *create_parameter(
            IType const *type,
            int         index) = 0;

        // ----------------- own methods -----------------

        /// Initialize this material instance.
        ///
        /// \param resolver                   The call name resolver.
        /// \param resource_modifier          The resource modifier or NULL.
        /// \param code_dag                   The generated code DAG.
        /// \param argc                       The number of arguments.
        /// \param argv                       An array of pointers to argument DAG nodes.
        ///                                   The nodes will be imported into the material instance.
        /// \param use_temporaries            If true, hide multiple used subexpressions behind
        ///                                   temporaries, if false, generate a true DAG.
        /// \param flags                      Instantiation flags.
        /// \param evaluator                  If non-NULL, use this evaluator additionally to fold
        ///                                   intrinsic functions first.
        /// \param fold_meters_per_scene_unit
        ///                                   If true, occurrences of the functions
        ///                                   state::meters_per_scene_unit() and
        ///                                   state::scene_units_per_meter() will be folded
        ///                                   using the \c mdl_meters_per_scene_unit parameter.
        /// \param mdl_meters_per_scene_unit  The value for the meter/scene unit conversion
        ///                                   only used when folding is enabled.
        /// \param wavelength_min             The value for the state::wavelength_min() function.
        /// \param wavelength_max             The value for the state::wavelength_max() function.
        /// \param fold_params                Names of parameters to be folded in class-compilation
        ///                                   mode (in addition to flags).
        /// \param num_fold_params            The number of parameter names to be folded.
        ///
        /// \returns                The error code of the initialization.
        ///
        /// Arguments are always given by position.
        /// If a NULL argument is given an EC_INSTANTIATION_ERROR is returned in error_code.
        virtual Error_code initialize(
            ICall_name_resolver       *resolver,
            IResource_modifier        *resource_modifier,
            IGenerated_code_dag const *code_dag,
            size_t                    argc,
            DAG_node const            *argv[],
            bool                      use_temporaries,
            unsigned                  flags,
            ICall_evaluator           *evaluator,
            bool                      fold_meters_per_scene_unit,
            float                     mdl_meters_per_scene_unit,
            float                     wavelength_min,
            float                     wavelength_max,
            char const * const        fold_params[],
            size_t                    num_fold_params) = 0;

        /// Return the material constructor of this instance.
        ///
        /// This method returns the body expression of a material instance. This is always
        /// a call to the MDL material constructor.
        virtual DAG_call const *get_constructor() const = 0;

        /// Return the number of temporaries of this instance.
        virtual size_t get_temporary_count() const = 0;

        /// Return the value of the temporary at index.
        ///
        /// \param index  the index of the temporary
        virtual DAG_node const *get_temporary_value(size_t index) const = 0;

        /// Return the number of parameters of this instance.
        ///
        /// \note: Returns always 0 in instance compilation mode.
        virtual size_t get_parameter_count() const = 0;

        /// Return the default value of a parameter of this instance.
        ///
        /// \param index  the index of the parameter
        virtual IValue const *get_parameter_default(size_t index) const = 0;

        /// Return the hash value of this material instance.
        ///
        /// This returns the hash-value of the body expression of this material instance.
        virtual DAG_hash const *get_hash() const = 0;

        /// Return the hash value of one material slot of this material instance.
        ///
        /// \param slot  the material slot
        ///
        /// This returns the hash value of a sub expression of the material instance.
        virtual DAG_hash const *get_slot_hash(Slot slot) const = 0;

        /// Return the canonical parameter name of the given parameter.
        ///
        /// \param index  the index of the parameter
        virtual char const *get_parameter_name(size_t index) const = 0;

        /// Returns true if this instance depends on object transforms.
        ///
        /// If this returns \c true, the material body expression of this material instance
        /// might depend on the MDL uniform \c state::transform_*() state functions.
        virtual bool depends_on_transform() const = 0;

        /// Returns true if this instance depends on the object id.
        ///
        /// If this returns \c true, the material body expression of this material instance
        /// might depend on the MDL uniform \c state::object_id() state function.
        virtual bool depends_on_object_id() const = 0;

        /// Returns true if this instance depends on the global distribution (edf).
        ///
        /// If this returns \c true, the material body expression of this material instance
        /// might depend on the MDL edf with global distribution.
        virtual bool depends_on_global_distribution() const = 0;

        /// Returns true if this instance depends on uniform scene data.
        virtual bool depends_on_uniform_scene_data() const = 0;

        /// Returns the number of scene data attributes referenced by this instance.
        virtual size_t get_referenced_scene_data_count() const = 0;

        /// Return the name of a scene data attribute referenced by this instance.
        ///
        /// \param index  the index of the scene data attribute
        virtual char const *get_referenced_scene_data_name(size_t index) const = 0;

        /// Returns the opacity of this instance.
        virtual Opacity get_opacity() const = 0;

        /// Returns the surface opacity of this instance.
        virtual Opacity get_surface_opacity() const = 0;

        /// Returns the cutout opacity of this instance if it is constant.
        ///
        /// \return if the cutout opacity is a constant (and was read),
        ///         NULL if it depends on parameters / complex user expressions
        virtual IValue_float const *get_cutout_opacity() const = 0;

        /// Access messages.
        virtual Messages const &access_messages() const = 0;

        /// Get the instance properties.
        virtual Properties get_properties() const = 0;

        /// Get the internal space.
        virtual char const *get_internal_space() const = 0;

        /// Set a tag, version pair for a resource constant that might be reachable from this
        /// instance.
        ///
        /// \param res             a resource
        /// \param tag             the tag value
        virtual void set_resource_tag(
            IValue_resource const *res,
            int                   tag) = 0;

        /// Get the number of resource tag map entries.
        virtual size_t get_resource_tag_map_entries_count() const = 0;

        /// Get the i'th resource tag map entry or NULL if the index is out of bounds;
        ///
        /// \param index  the index of the resource map entry.
        virtual Resource_tag_tuple const *get_resource_tag_map_entry(size_t index) const = 0;

        /// Get the resource tagger for this material instance.
        virtual IResource_tagger *get_resource_tagger() const = 0;
    };

    // -------------------------- methods --------------------------

public:
    /// Get the absolute module name of the module from which this code was generated.
    virtual char const *get_module_name() const = 0;

    /// Get the module file name of the module from which this code was generated.
    virtual char const *get_module_file_name() const = 0;

    /// Get the number of modules directly imported by the module
    /// from which this code was generated.
    virtual size_t get_import_count() const = 0;

    /// Get the name of module at index imported from the module
    /// from which this code was generated.
    ///
    /// \param index  the index of the requested imported module
    virtual char const *get_import(size_t index) const = 0;

    /// Get the number of functions in the generated code.
    ///
    /// \returns    The number of functions in this generated code.
    virtual size_t get_function_count() const = 0;

    /// Get the return type of the function at function_index.
    ///
    /// \param function_index  The index of the function.
    /// \returns               The return type of the function.
    virtual IType const *get_function_return_type(size_t function_index) const = 0;

    /// Get the semantics of the function at function_index.
    ///
    /// \param function_index  The index of the function.
    /// \returns               The semantics of the function.
    virtual IDefinition::Semantics get_function_semantics(size_t function_index) const = 0;

    /// Get the name of the function at function_index.
    ///
    /// \param function_index  The index of the function.
    /// \returns               The name of the function.
    virtual char const *get_function_name(size_t function_index) const = 0;

    /// Get the simple name of the function at function_index.
    ///
    /// \param function_index  The index of the function.
    /// \returns               The simple name of the function.
    virtual char const *get_simple_function_name(size_t function_index) const = 0;

    /// Get the original name of the function at function_index if the function name is an alias,
    /// i.e. re-exported from a module.
    ///
    /// \param function_index  The index of the function.
    /// \returns               The original name of the function or NULL.
    virtual char const *get_original_function_name(size_t function_index) const = 0;

    /// Get the parameter count of the function at function_index.
    ///
    /// \param function_index  The index of the function.
    /// \returns               The number of parameters of the function.
    virtual size_t get_function_parameter_count(size_t function_index) const = 0;

    /// Get the parameter type of the parameter at parameter_index
    /// of the function at function_index.
    ///
    /// \param function_index  The index of the function.
    /// \param parameter_index The index of the parameter.
    /// \returns               The type of the parameter.
    virtual IType const *get_function_parameter_type(
        size_t function_index,
        size_t parameter_index) const = 0;

    /// Get the parameter type name of the parameter at parameter_index
    /// of the function at function_index.
    /// \param      function_index  The index of the function.
    /// \param      parameter_index The index of the parameter.
    /// \returns                    The type name of the parameter.
    virtual char const *get_function_parameter_type_name(
        size_t function_index,
        size_t parameter_index) const = 0;

    /// Get the parameter name of the parameter at parameter_index
    /// of the function at function_index.
    ///
    /// \param function_index  The index of the function.
    /// \param parameter_index The index of the parameter.
    /// \returns               The name of the parameter.
    virtual char const *get_function_parameter_name(
        size_t function_index,
        size_t parameter_index) const = 0;

    /// Get the index of the parameter parameter_name.
    ///
    /// \param function_index  The index of the function.
    /// \param parameter_name  The name of the parameter.
    /// \returns               The index of the parameter, or -1 if it does not exist.
    virtual size_t get_function_parameter_index(
        size_t     function_index,
        char const *parameter_name) const = 0;

    /// Get the enable_if condition for the given function parameter if one was specified.
    ///
    /// \param function_index  The index of the function.
    /// \param parameter_index The index of the parameter.
    /// \returns               The enable_if condition for this parameter or NULL.
    virtual DAG_node const *get_function_parameter_enable_if_condition(
        size_t function_index,
        size_t parameter_index) const = 0;

    /// Get the number of parameters whose enable_if condition depends on this parameter.
    ///
    /// \param function_index  The index of the function.
    /// \param parameter_index The index of the parameter.
    /// \returns               Number of depended parameter conditions.
    virtual size_t get_function_parameter_enable_if_condition_users(
        size_t function_index,
        size_t parameter_index) const = 0;

    /// Get a parameter index whose enable_if condition depends on this parameter.
    ///
    /// \param function_index  The index of the function.
    /// \param parameter_index The index of the parameter.
    /// \param user_index      The index of the user.
    /// \returns               The index of the depended parameter.
    virtual size_t get_function_parameter_enable_if_condition_user(
        size_t function_index,
        size_t parameter_index,
        size_t user_index) const = 0;

    /// Get the function hash value for the given function index if available.
    ///
    /// \param function_index  The index of the function.
    /// \returns               The function hash of the function or NULL if no hash
    ///                        value is available or the index is out of bounds.
    virtual DAG_hash const *get_function_hash(
        size_t function_index) const = 0;

    /// Get the number of materials in the generated code.
    ///
    /// \returns    The number of materials in this generated code.
    virtual size_t get_material_count() const = 0;

    /// Get the name of the material at material_index.
    ///
    /// \param material_index  The index of the material.
    /// \returns               The name of the material.
    virtual char const *get_material_name(size_t material_index) const = 0;

    /// Get the simple name of the material at material_index.
    ///
    /// \param material_index  The index of the material.
    /// \returns               The simple name of the material.
    virtual char const *get_simple_material_name(size_t material_index) const = 0;

    /// Get the original name of the material at material_index if the material name is an alias.
    ///
    /// \param material_index  The index of the material.
    /// \returns               The name of the material or NULL.
    virtual char const *get_original_material_name(size_t material_index) const = 0;

    /// Get the parameter count of the material at material_index.
    ///
    /// \param material_index  The index of the material.
    /// \returns               The number of parameters of the material.
    virtual size_t get_material_parameter_count(size_t material_index) const = 0;

    /// Get the parameter type of the parameter at parameter_index
    /// of the material at material_index.
    ///
    /// \param material_index   The index of the material.
    /// \param parameter_index  The index of the parameter.
    /// \returns                The type of the parameter.
    virtual IType const *get_material_parameter_type(
        size_t material_index,
        size_t parameter_index) const = 0;

    /// Get the parameter name of the parameter at parameter_index
    /// of the material at material_index.
    ///
    /// \param material_index   The index of the material.
    /// \param parameter_index  The index of the parameter.
    /// \returns                The name of the parameter.
    virtual char const *get_material_parameter_name(
        size_t material_index,
        size_t parameter_index) const = 0;

    /// Get the index of the parameter parameter_name.
    ///
    /// \param material_index  The index of the material.
    /// \param parameter_name  The name of the parameter.
    /// \returns               The index of the parameter, or -1 if it does not exist.
    virtual size_t get_material_parameter_index(
        size_t     material_index,
        char const *parameter_name) const = 0;

    /// Get the enable_if condition for the given material parameter if one was specified.
    ///
    /// \param material_index   The index of the material.
    /// \param parameter_index  The index of the parameter.
    /// \returns                The enable_if condition for this parameter or NULL.
    virtual DAG_node const *get_material_parameter_enable_if_condition(
        size_t material_index,
        size_t parameter_index) const = 0;

    /// Get the number of parameters whose enable_if condition depends on this parameter.
    ///
    /// \param material_index   The index of the material.
    /// \param parameter_index  The index of the parameter.
    /// \returns                Number of depended parameter conditions.
    virtual size_t get_material_parameter_enable_if_condition_users(
        size_t material_index,
        size_t parameter_index) const = 0;

    /// Get a parameter index whose enable_if condition depends on this parameter.
    ///
    /// \param material_index   The index of the material.
    /// \param parameter_index  The index of the parameter.
    /// \param user_index       The index of the user.
    /// \returns                The index of the depended parameter.
    virtual size_t get_material_parameter_enable_if_condition_user(
        size_t material_index,
        size_t parameter_index,
        size_t user_index) const = 0;

    /// Get the node factory.
    virtual DAG_node_factory *get_node_factory() = 0;

    /// Get the number of annotations of the function at function_index.
    ///
    /// \param function_index  The index of the function.
    /// \returns               The number of annotations.
    virtual size_t get_function_annotation_count(
        size_t function_index) const = 0;

    /// Get the annotation at annotation_index of the function at function_index.
    ///
    /// \param function_index    The index of the function.
    /// \param annotation_index  The index of the annotation.
    /// \returns                 The annotation.
    virtual DAG_node const *get_function_annotation(
        size_t function_index,
        size_t annotation_index) const = 0;

    /// Get the number of annotations of the function return type at function_index.
    ///
    /// \param function_index The index of the function.
    /// \returns              The number of annotations.
    virtual size_t get_function_return_annotation_count(
        size_t function_index) const = 0;

    /// Get the annotation at annotation_index of the function return type at function_index.
    ///
    /// \param function_index    The index of the function.
    /// \param annotation_index  The index of the annotation.
    /// \returns                 The annotation.
    virtual DAG_node const *get_function_return_annotation(
        size_t function_index,
        size_t annotation_index) const = 0;

    /// Get the default initializer of the parameter at parameter_index
    /// of the function at function_index.
    ///
    /// \param function_index   The index of the function.
    /// \param parameter_index  The index of the parameter.
    /// \returns                The default initializer or NULL if not available.
    virtual DAG_node const *get_function_parameter_default(
        size_t function_index,
        size_t parameter_index) const = 0;

    /// Get the number of annotations of the parameter at parameter_index
    /// of the function at function_index.
    ///
    /// \param function_index   The index of the function.
    /// \param parameter_index  The index of the parameter.
    /// \returns                The number of annotations.
    virtual size_t get_function_parameter_annotation_count(
        size_t function_index,
        size_t parameter_index) const = 0;

    /// Get the annotation at annotation_index of the parameter at parameter_index
    /// of the function at function_index.
    ///
    /// \param function_index    The index of the function.
    /// \param parameter_index   The index of the parameter.
    /// \param annotation_index  The index of the annotation.
    /// \returns                 The annotation.
    virtual DAG_node const *get_function_parameter_annotation(
        size_t function_index,
        size_t parameter_index,
        size_t annotation_index) const = 0;

    /// Get the number of temporaries used by the function at function_index.
    ///
    /// \param function_index      The index of the function.
    /// \returns                   The number of temporaries used by material.
    virtual size_t get_function_temporary_count(
        size_t function_index) const = 0;

    /// Get the temporary at temporary_index used by the function at function_index.
    ///
    /// \param function_index      The index of the function.
    /// \param temporary_index     The index of the temporary variable.
    /// \returns                   The value of the temporary variable.
    virtual DAG_node const *get_function_temporary(
        size_t function_index,
        size_t temporary_index) const = 0;

    /// Get the temporary name at temporary_index used by the function at function_index.
    ///
    /// \param function_index      The index of the function.
    /// \param temporary_index     The index of the temporary variable.
    /// \returns                   The name of the temporary variable.
    virtual char const *get_function_temporary_name(
        size_t function_index,
        size_t temporary_index) const = 0;

    /// Get the body of the function at function_index.
    ///
    /// \param function_index      The index of the function.
    /// \returns                   The body of the function.
    virtual DAG_node const *get_function_body(
        size_t function_index) const = 0;

    /// Get the property flag of the function at function_index.
    ///
    /// \param function_index  The index of the function.
    /// \param fp              The requested property.
    /// \returns               True if this function has the property, false if not.
    virtual bool get_function_property(
        size_t            function_index,
        Function_property fp) const = 0;

    /// Get the number of entities referenced by a function.
    ///
    /// \param function_index  The index of the function.
    /// \returns               Number of function that might be called by this function
    virtual size_t get_function_references_count(size_t function_index) const = 0;

    /// Get the signature of the i'th reference of a function.
    ///
    /// \param function_index  The index of the function.
    /// \param callee_index    The index of the callee.
    /// \returns               Number of function that might be called by this function
    virtual char const *get_function_reference(
        size_t function_index,
        size_t callee_index) const = 0;

    /// Return the original function name of a cloned function or "" if the function
    /// is not a clone.
    ///
    /// \param function_index   The index of the function.
    /// \returns                The absolute name of the original function or "".
    virtual char const *get_cloned_function_name(
        size_t function_index) const = 0;

    /// Get the number of annotations of the material at material_index.
    ///
    /// \param material_index  The index of the material.
    /// \returns               The number of annotations.
    virtual size_t get_material_annotation_count(
        size_t material_index) const = 0;

    /// Get the annotation at annotation_index of the material at material_index.
    ///
    /// \param material_index      The index of the material.
    /// \param annotation_index    The index of the annotation.
    /// \returns                   The annotation.
    virtual DAG_node const *get_material_annotation(
        size_t material_index,
        size_t annotation_index) const = 0;

    /// Get the default initializer of the parameter at parameter_index
    /// of the material at material_index.
    ///
    /// \param material_index      The index of the material.
    /// \param parameter_index     The index of the parameter.
    /// \returns                   The type of the parameter.
    virtual DAG_node const *get_material_parameter_default(
        size_t material_index,
        size_t parameter_index) const = 0;

    /// Get the number of annotations of the parameter at parameter_index
    /// of the material at material_index.
    ///
    /// \param material_index      The index of the material.
    /// \param parameter_index     The index of the parameter.
    /// \returns                   The number of annotations.
    virtual size_t get_material_parameter_annotation_count(
        size_t material_index,
        size_t parameter_index) const = 0;

    /// Get the annotation at annotation_index of the parameter at parameter_index
    /// of the material at material_index.
    ///
    /// \param material_index      The index of the material.
    /// \param parameter_index     The index of the parameter.
    /// \param annotation_index    The index of the annotation.
    /// \returns                   The annotation.
    virtual DAG_node const *get_material_parameter_annotation(
        size_t material_index,
        size_t parameter_index,
        size_t annotation_index) const = 0;

    /// Get the number of temporaries used by the material at material_index.
    ///
    /// \param material_index      The index of the material.
    /// \returns                   The number of temporaries used by material.
    virtual size_t get_material_temporary_count(
        size_t material_index) const = 0;

    /// Get the temporary at temporary_index used by the material at material_index.
    ///
    /// \param material_index      The index of the material.
    /// \param temporary_index     The index of the temporary variable.
    /// \returns                   The value of the temporary variable.
    virtual DAG_node const *get_material_temporary(
        size_t material_index,
        size_t temporary_index) const = 0;

    /// Get the temporary name at temporary_index used by the material at material_index.
    ///
    /// \param material_index      The index of the material.
    /// \param temporary_index     The index of the temporary variable.
    /// \returns                   The name of the temporary variable.
    virtual char const *get_material_temporary_name(
        size_t material_index,
        size_t temporary_index) const = 0;

    /// Get the value of the material at material_index.
    ///
    /// \param material_index      The index of the material.
    /// \returns                   The value of the material.
    virtual DAG_node const *get_material_value(
        size_t material_index) const = 0;

    /// Get the export flags of the material at material_index.
    ///
    /// \param material_index  The index of the material.
    /// \returns               True if this is an exported material, false if it is local.
    virtual bool get_material_exported(size_t material_index) const = 0;

    /// Return the original material name of a cloned material or "" if the material
    /// is not a clone.
    ///
    /// \param material_index   The index of the material.
    /// \returns                The absolute name of the original material or "".
    virtual char const *get_cloned_material_name(
        size_t material_index) const = 0;

    /// Create a new material instance.
    ///
    /// \param[in]  index       The index of the material to instantiate.
    /// \param[out] error_code  The error code of the instance creation.
    /// \returns                The material instance.
    ///
    virtual IMaterial_instance *create_material_instance(
        size_t     index,
        Error_code *error_code = NULL) const = 0;

    /// Get the number of exported user types.
    virtual size_t get_type_count() const = 0;

    /// Get the name of the type at index.
    ///
    /// \param index  The index of the type.
    /// \returns      The name of the type.
    virtual char const *get_type_name(
        size_t index) const = 0;

    /// Get the original name of the type at index if the type is an alias.
    ///
    /// \param index  The index of the type.
    /// \returns      The original name of the type or NULL.
    virtual char const *get_original_type_name(
        size_t index) const = 0;

    /// Get the user type at index.
    ///
    /// \param index  The index of the type.
    /// \returns      The type.
    virtual IType const *get_type(
        size_t index) const = 0;

    /// Returns true if the type at index is exported.
    ///
    /// \param index  The index of the type.
    /// \returns      true for exported types.
    virtual bool is_type_exported(
        size_t index) const = 0;

    /// Get the number of annotations of the type at index.
    ///
    /// \param index  The index of the type.
    /// \return       The number of annotations of the type.
    virtual size_t get_type_annotation_count(
        size_t index) const = 0;

    /// Get the annotation at annotation_index of the type at type_index.
    ///
    /// \param type_index          The index of the type.
    /// \param annotation_index    The index of the annotation.
    /// \returns                   The annotation.
    virtual DAG_node const *get_type_annotation(
        size_t type_index,
        size_t annotation_index) const = 0;

    /// Get the number of type sub-entities (fields or enum constants).
    ///
    /// \param type_index          The index of the type.
    /// \returns                   The number of sub entities.
    virtual size_t get_type_sub_entity_count(
        size_t type_index) const = 0;

    /// Get the name of a type sub-entity (field or enum constant).
    ///
    /// \param type_index          The index of the type.
    /// \param entity_index        The index of the sub entity.
    /// \returns                   The name of a sub-entity.
    virtual char const *get_type_sub_entity_name(
        size_t type_index,
        size_t entity_index) const = 0;

    /// Get the type of a type sub-entity (field or enum constant).
    ///
    /// \param type_index          The index of the type.
    /// \param entity_index        The index of the sub entity.
    /// \returns                   The type of sub-entity.
    virtual IType const *get_type_sub_entity_type(
        size_t type_index,
        size_t entity_index) const = 0;

    /// Get the number of annotations of a type sub-entity (field or enum constant) at index.
    ///
    /// \param type_index          The index of the type.
    /// \param entity_index        The index of the sub entity.
    /// \returns                   The number of annotations of the type sub-entity.
    virtual size_t get_type_sub_entity_annotation_count(
        size_t type_index,
        size_t entity_index) const = 0;

    /// Get the annotation at annotation_index of the type sub-entity at (type_index, entity_index).
    ///
    /// \param type_index          The index of the type.
    /// \param entity_index        The index of the sub entity.
    /// \param annotation_index    The index of the annotation.
    /// \returns                   The annotation.
    virtual DAG_node const *get_type_sub_entity_annotation(
        size_t type_index,
        size_t entity_index,
        size_t annotation_index) const = 0;

    /// Get the number of exported constants.
    virtual size_t get_constant_count() const = 0;

    /// Get the name of the constant at index.
    ///
    /// \param index  The index of the constant.
    /// \returns      The name of the constant.
    virtual char const *get_constant_name(
        size_t index) const = 0;

    /// Get the value of the constant at index.
    ///
    /// \param index  The index of the constant.
    /// \returns      The value of the constant.
    virtual DAG_constant const *get_constant_value(
        size_t index) const = 0;

    /// Get the number of annotations of the constant at index.
    ///
    /// \param index  The index of the constant.
    virtual size_t get_constant_annotation_count(
        size_t index) const = 0;

    /// Get the annotation at annotation_index of the constant at constant_index.
    ///
    /// \param constant_index      The index of the constant.
    /// \param annotation_index    The index of the annotation.
    /// \returns                   The annotation.
    virtual DAG_node const *get_constant_annotation(
        size_t constant_index,
        size_t annotation_index) const = 0;

    /// Access messages.
    virtual Messages const &access_messages() const = 0;

    /// Access messages (writable).
    virtual Messages &access_messages() = 0;

    /// Returns the amount of used memory by this code DAG.
    virtual size_t get_memory_size() const = 0;

    /// Get the number of annotations of the module.
    ///
    /// \returns                    The number of annotations.
    virtual size_t get_module_annotation_count() const = 0;

    /// Get the annotation at annotation_index of the module.
    ///
    /// \param annotation_index    The index of the annotation.
    /// \returns                   The annotation.
    virtual DAG_node const *get_module_annotation(
        size_t annotation_index) const = 0;

    /// Get the internal space.
    virtual char const *get_internal_space() const = 0;

    /// Get the number of annotations in the generated code.
    ///
    /// \returns    The number of annotations in this generated code.
    virtual size_t get_annotation_count() const = 0;

    /// Get the semantics of the annotation at annotation_index.
    ///
    /// \param annotation_index  The index of the annotation.
    /// \returns                 The semantics of the annotation.
    virtual IDefinition::Semantics get_annotation_semantics(
        size_t annotation_index) const = 0;

    /// Get the name of the annotation at annotation_index.
    ///
    /// \param annotation_index  The index of the annotation.
    /// \returns                 The name of the annotation.
    virtual char const *get_annotation_name(
        size_t annotation_index) const = 0;

    /// Get the simple name of the annotation at annotation_index.
    ///
    /// \param annotation_index  The index of the annotation.
    /// \returns                 The simple name of the annotation.
    virtual char const *get_simple_annotation_name(
        size_t annotation_index) const = 0;

    /// Get the original name of the annotation at annotation_index if the annotation name is
    /// an alias, i.e. re-exported from a module.
    ///
    /// \param annotation_index  The index of the annotation.
    /// \returns                 The original name of the annotation or NULL.
    virtual char const *get_original_annotation_name(
        size_t annotation_index) const = 0;

    /// Get the parameter count of the annotation at annotation_index.
    ///
    /// \param annotation_index  The index of the annotation.
    /// \returns                 The number of parameters of the annotation.
    virtual size_t get_annotation_parameter_count(
        size_t annotation_index) const = 0;

    /// Get the parameter type of the parameter at parameter_index
    /// of the annotation at annotation_index.
    ///
    /// \param annotation_index  The index of the annotation.
    /// \param parameter_index   The index of the parameter.
    /// \returns                 The type of the parameter.
    virtual IType const *get_annotation_parameter_type(
        size_t annotation_index,
        size_t parameter_index) const = 0;

    /// Get the parameter type name of the parameter at parameter_index
    /// of the annotation at annotation_index.
    /// \param annotation_index  The index of the annotation.
    /// \param parameter_index   The index of the parameter.
    /// \returns                 The type of the parameter.
    virtual char const *get_annotation_parameter_type_name(
        size_t annotation_index,
        size_t parameter_index) const = 0;

    /// Get the parameter name of the parameter at parameter_index
    /// of the annotation at annotation_index.
    ///
    /// \param annotation_index  The index of the annotation.
    /// \param parameter_index   The index of the parameter.
    /// \returns                 The name of the parameter.
    virtual char const *get_annotation_parameter_name(
        size_t annotation_index,
        size_t parameter_index) const = 0;

    /// Get the index of the parameter parameter_name.
    ///
    /// \param annotation_index  The index of the annotation.
    /// \param parameter_name    The name of the parameter.
    /// \returns                 The index of the parameter, or -1 if it does not exist.
    virtual size_t get_annotation_parameter_index(
        size_t     annotation_index,
        char const *parameter_name) const = 0;

    /// Get the default initializer of the parameter at parameter_index
    /// of the annotation at annotation_index.
    ///
    /// \param annotation_index   The index of the annotation.
    /// \param parameter_index    The index of the parameter.
    /// \returns                  The default initializer or NULL if not available.
    virtual DAG_node const *get_annotation_parameter_default(
        size_t annotation_index,
        size_t parameter_index) const = 0;

    /// Get the property flag of the annotation at annotation_index.
    ///
    /// \param annotation_index  The index of the annotation.
    /// \param ap                The requested annotation property.
    /// \returns                 True if this annotation has the property, false if not.
    virtual bool get_annotation_property(
        size_t              annotation_index,
        Annotation_property ap) const = 0;

    /// Get the number of annotations of the annotation at annotation_index.
    ///
    /// \param annotation_index  The index of the annotation.
    /// \returns               The number of annotations.
    virtual size_t get_annotation_annotation_count(
        size_t annotation_index) const = 0;

    /// Get the annotation at annotation_index of the annotation (declaration) at anno_decl_index.
    ///
    /// \param anno_decl_index    The index of the annotation (declaration).
    /// \param annotation_index   The index of the annotation.
    /// \returns                  The annotation.
    virtual DAG_node const *get_annotation_annotation(
        size_t anno_decl_index,
        size_t annotation_index) const = 0;

    /// Get a tag,for a resource constant that might be reachable from this DAG.
    ///
    /// \param res             a resource
    virtual int get_resource_tag(
        IValue_resource const *res) const = 0;

    /// Set a tag, version pair for a resource constant that might be reachable from this DAG.
    ///
    /// \param res             a resource
    /// \param tag             the tag value
    virtual void set_resource_tag(
        IValue_resource const *res,
        int                   tag) = 0;

    /// Get the number of resource tag map entries.
    virtual size_t get_resource_tag_map_entries_count() const = 0;

    /// Get the i'th resource tag map entry or NULL if the index is out of bounds;
    ///
    /// \param index  the index of the resource map entry.
    virtual Resource_tag_tuple const *get_resource_tag_map_entry(size_t index) const = 0;

    /// Get the resource tagger for this code DAG.
    virtual IResource_tagger *get_resource_tagger() const = 0;
};

/// Check if a DAG node is of a certain type.
template<typename T>
bool is(DAG_node const *node)
{
    return node->get_kind() == T::s_kind;
}

/// Cast to subtype or return NULL if types do not match.
template<typename T>
T *as(DAG_node *type) {
    return (type->get_kind() == T::s_kind) ? static_cast<T *>(type) : NULL;
}
 
/// Cast to subtype or return NULL if types do not match.
template<typename T>
T const *as(DAG_node const *type) {
    return (type->get_kind() == T::s_kind) ? static_cast<const T *>(type) : NULL;
}

/*!
\page mdl_dag_ir MDL DAG IR

The MDL DAG IR is an intermediate representation currently used for MDL materials,
MDL material instances, and lambda functions (part of materials).
The DAG IR is a directed acyclic dependence graph with exactly one root, representing one
MDL expression.

A node \c A has an edge to a node \c B, if and only if the computation of \c A depends on
the computation of \c B, in particular the expression \c B is an argument of the call \c A.
Hence, edges in the DAG IR are in reverse order compared to data-flow graphs, where
the edges would be from the definition to the user.
In DAG IR, all edges go from the user to the definition (the source of a value).

Currently, MDL statements cannot be represented in the DAG IR, hence the bodies
of MDL functions cannot be represented.
This will be changed in the future.

Currently, the DAG IR consists of only four different DAG IR nodes:

 - A \c DAG_constant represents a single literal value inside the IR.
   \c DAG_constants are always leafs inside the DAG.

 - A \c DAG_call represents the call to a function, operator, or material
   inside the IR. Its children are the arguments of the call.
   The edges to the children are attributed by the parameter name and an index.
   They must always correspond to the parameter order of the called entity.

 - A \c DAG_parameter represents a parameter of the owning entity.
   Parameters only have one index attribute that corresponds to the index
   of the parameter in the DAG IR owner.

 - A \c DAG_temporary is an optional node in the IR.
   It can be used to split the DAG into a set of expression trees.
   If a node has several users, say A<-B and A<-C, the definition node can be
   replaced by instances of a temporary T, i.e. T'<-B and T''<-C.
   The nodes T' and T'' use the same temporary index \c i.
   The owner of the DAG IR then stores under index \c i the expression \c A.

Note that temporary nodes have an expression edge, that points directly to
the expression. Hence even with temporaries, an application can either stop
at a temporary, getting a tree view, or skip temporaries (by following the \c expr
edge) to see a DAG view.

Every node in a DAG IR is typed (by using an IType).

Note that this representation can be seen as a single static assignment (SSA)
representation of an expression.
Due to this, powerful optimization can be done on the DAG IR and are automatically
enabled during construction, like

 - constant folding
 - common sub-expression evaluation
 - simplification of arithmetic expressions
 - reassociation of sub-expressions

If not explicitly requested via the \c use_temporaries parameter of
IGenerated_code_dag::IMaterial_instance::initialize(), the DAG backend will not create
\c DAG_temporary nodes.
An application should operate on the DAG IR directly to get the most out of this representation.
*/

} // mdl
} // mi

#endif
