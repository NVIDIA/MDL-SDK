/***************************************************************************************************
 * Copyright (c) 2006-2020, NVIDIA CORPORATION. All rights reserved.
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
 **************************************************************************************************/
/// \file
/// \brief Base class for database elements with attributes

#ifndef MI_NEURAYLIB_IATTRIBUTE_SET_H
#define MI_NEURAYLIB_IATTRIBUTE_SET_H

#include <mi/base/interface_declare.h>
#include <mi/neuraylib/idata.h>
#include <mi/neuraylib/type_traits.h>

namespace mi {
namespace neuraylib {

/** \addtogroup mi_neuray_scene_element
@{
*/

/// Propagation types for attribute inheritance.
///
/// Inheritance of attributes allows you to easily make changes to an entire subtree of the scene
/// graph. \if IRAY_API For example, by setting the \c visible attribute of an
/// #mi::neuraylib::IGroup element to \c false, the entire subtree will become invisible (unless one
/// of its elements has a \c visible attribute, too, and its value is \c true). Similarly, materials
/// can be inherited down the scene graph to elements that do not have their own material.\endif
///
///
/// Assume the scene graph contains an inner node P (the parent) \if IRAY_API like
/// #mi::neuraylib::IGroup or #mi::neuraylib::IInstance \endif with a (possibly inherited)
/// attribute A and a child node C. If the child node C does not have the attribute A, the child
/// will inherit P's value of A (without actually storing an explicit copy). If the child node C has
/// the attribute A, it will use its own value of the attribute without taking P's value into
/// consideration. This is the standard propagation rule which is represented by the propagation
/// type #mi::neuraylib::PROPAGATION_STANDARD.
///
/// The outcome of the last case can be reverted by setting the \c override flag of the attribute A
/// on the parent node (or at the node from where it was inherited to P): in this case P's value of
/// A is inherited to the child node C, no matter whether the child has the attribute or not. This
/// propagation mode is represented by #mi::neuraylib::PROPAGATION_OVERRIDE.
///
/// \note There is one exception to these rules: If the child node C is a geometry object, then P's
///       value of A is used, no matter whether the propagation mode was set to
///       #mi::neuraylib::PROPAGATION_OVERRIDE or #mi::neuraylib::PROPAGATION_STANDARD. The
///       rationale is that for geometry objects the attribute on the object itself should be a
///       default that is only used if no value for that attribute is inherited.
///
/// \note Inherited attributes are made available to the render modes, but are not available via the
///       \neurayApiName.
///
/// \see #mi::neuraylib::IAttribute_set::set_attribute_propagation()
/// \see #mi::neuraylib::IAttribute_set::get_attribute_propagation()
enum Propagation_type {
    PROPAGATION_STANDARD, ///< Standard inheritance of attributes without any special flags.
    PROPAGATION_OVERRIDE, ///< The \c override flag for attribute inheritance.
    PROPAGATION_FORCE_32_BIT = 0xffffffffU
};

mi_static_assert( sizeof( Propagation_type) == sizeof( Uint32));

/// The attribute set comprises all attributes attached to a database element.
///
/// Attributes are pieces of information that can be attached to any database element. Basically,
/// an attribute set is a map from attribute names (strings) to attribute values (instances of
/// #mi::IData).
///
/// Attributes can be inherited in the scene graph. For details, see
/// #mi::neuraylib::Propagation_type.
///
/// \note Setting an attribute value is done by value (or deep copy) and not by reference. This
/// is relevant for types that usually follow reference semantics, for example, arrays or
/// structures. Note that references to other DB elements (see #mi::IRef) are still stored as
/// reference.
///
/// \if IRAY_API
///
/// \see
/// The #mi::neuraylib::IOptions class has many attributes controlling global settings.
///
/// \par Attributes
/// - bool \b disable \n
///   If set to \c true, the element is ignored. If the element references sub-elements in the scene
///   graph, like instances or groups do, these references will be ignored as well. (Of course,
///   there may be other references of non-disabled elements that can make shared sub-elements in
///   their context visible.) This attribute is not supported for decals. \n
///   \n
///   This attribute, once set to \c true, cannot be reset to \c false by elements lower in the
///   scene graph. This can be used to efficiently turn off objects, lights and groups by disabling
///   their referencing instance element; it would be much more expensive to detach or attach them
///   to the scene graph because that requires preprocessing the scene again.
/// - bool \b visible \n
///   The object or light is visible to primary rays. This attribute is not supported for decals.
/// - bool \b matte \n
///   The object or light is treated as a matte object.
/// - #mi::Float32 \b matte_shadow_intensity \n
///   Scaling factor to tune the artificial shadow cast on matte objects. The default is 1.0.
/// - bool \b matte_connect_to_environment \n
///   Only effective if the backplate function or backplate color is set. Matte
///   objects will then use the environment instead of the backplate for secondary interactions.
/// - bool \b matte_connect_from_camera \n
///   Only effective if no backplate function and no backplate color is set, or if 
///   a backplate function or backplate color is set and in addition the
///   matte_connect_to_environment flag. Matte objects will then use a local lookup into the real
///   world photograph instead of a global one.
/// - bool \b backplate_mesh \n
///   Only effective if the backplate function or backplate color is set. The object is then
///   treated as a 3D-environment/backplate.
/// - bool \b movable \n
///   The object may be subject to frequent transformation changes. Render modes might take that
///   hint into account and use special data structures to speed up such transformation changes.
///   This also controls instancing in user mode. See the [:ipmlink rendering Instancing] section for details.
/// - bool \b reflection_cast \n
///   The object is visible as a reflection in reflective objects.
/// - bool \b reflection_recv \n
///   The object is reflective.
/// - bool \b refraction_cast \n
///   The object is visible through refractive objects.
/// - bool \b refraction_recv \n
///   The object is refractive.
/// - bool \b shadow_cast \n
///   The object casts shadows.
/// - bool \b shadow_recv \n
///   The object can have shadows cast onto it.
/// - #mi::Sint32 \b label \n
///   An object ID that is useful in conjunction with render target canvases of type
///   #mi::neuraylib::TYPE_OBJECT_ID.
///   See #mi::neuraylib::IRender_target_base for details.
/// - #mi::Sint32 \b material_id \n
///   A material ID that is useful in conjunction with render target canvases of type
///   #mi::neuraylib::TYPE_MATERIAL_ID.
///   See #mi::neuraylib::IRender_target_base for details.
/// - const char* \b handle \n
///   An object or light ID that is useful in conjunction with Light Path Expressions.
///   See Section [:ipmlink reference Light path expressions] of the Iray Programmer's Manual
///   for details.
/// - bool \b shadow_terminator_offset \n
///   Controls the automatic shadow terminator handling.
///   See the [:ipmlink physically_plausible_scene_setup Tessellating curved surfaces] section for details.
/// \par
/// The following attribute is only meaningful for instances of #mi::neuraylib::ITriangle_mesh,
/// #mi::neuraylib::IPolygon_mesh, #mi::neuraylib::ISubdivision_surface,
/// #mi::neuraylib::IFreeform_surface, #mi::neuraylib::IOn_demand_mesh, #mi::neuraylib::ILight,
/// #mi::neuraylib::IDecal, and via inheritance for instances of #mi::neuraylib::IGroup and
/// #mi::neuraylib::IInstance.
/// - #mi::IRef \b material or #mi::IArray \b material \n
///   A reference to a material instance, or an array of such references.
/// \par
/// For decals, the array is limited to length 1.
/// \par
/// The following attribute is only meaningful for instances of #mi::neuraylib::ILight, and via
/// inheritance for instances of #mi::neuraylib::IGroup and #mi::neuraylib::IInstance.
/// - bool \b important \n
///   A light flagged with this attribute will be given preference before other lights in the case
///   that a render mode does not handle all lights.
/// - bool \b light_portal \n
///   A light flagged with this attribute does not emit any light by itself but acts as hint that
///   light is coming from its direction.
/// \par
/// The following attribute is only meaningful for instances of #mi::neuraylib::ITriangle_mesh,
/// #mi::neuraylib::IPolygon_mesh, #mi::neuraylib::ISubdivision_surface, and
/// #mi::neuraylib::IFreeform_surface, and via inheritance
/// for instances of #mi::neuraylib::IGroup and #mi::neuraylib::IInstance.
/// - struct Approx \b approx \n
///   This attribute controls the refinement of triangle and polygon meshes with displacement, and
///   the subdivision level of subdivision surfaces and freeform surfaces. The attribute is a struct
///   and has the members described below. A corresponding structure declaration is registered under
///   the type name \c Approx.
///   - #mi::Float32 \b const_u \n
///     Stores the constant \c c or \c c_u for parametric or distance ratio approximation, or the
///     length bound for length approximation.
///   - #mi::Float32 \b const_v \n
///     Stores the constant \c c_v for parametric approximation.
///   - #mi::Sint8 \b method \n
///     Three methods are available, parametric approximation (0), length approximation (1), and
///     distance ratio approximation (3).
///     \n
///     Parametric approximation is available for triangle and polygon meshes with displacement,
///     subdivision and freeform surfaces. For the first three object types it subdivides each
///     primitive (triangle or quadrangle) into 4^\c c primitives for some parameter \c c (polygons
///     are tessellated first). For freeform surfaces, assume that each surface patch has degrees \c
///     deg_u and \c deg_v. Each surface patch is subdivided into \c deg_u * \c c_u times \c deg_v *
///     \c c_v triangle pairs.
///     \n
///     Length approximation is available for triangle and polygon meshes with displacement, and for
///     freeform surfaces, but not for subdivision surfaces. It subdivides the primitives until all
///     edges have a length (in object space) below a specified bound.
///     \n
///     Distance ratio approximation is available only for freeform surfaces. The parameter \c c
///     is an upper bound for the ratio of the distance of the approximation to the original
///     curve/surface and the length of the corresponding edge in the approximation. For example,
///     a value of 0.01 means that the error is at most 1/100 of the edge length in the
///     approximation. Typical values of the approximation constant for the distance ratio method
///     are in the range [0.01,0.1].
///     \n
///     The length bound as well as the parameter \c c are stored in the field \c const_u. The
///     The parameters \c c_u and \c c_v are stored in the fields \c const_u and \c const_v,
///     respectively.
///   - #mi::Sint8 \b sharp \n
///     Unused.
/// \par
/// The following attribute is only meaningful for instances of #mi::neuraylib::IFreeform_surface,
/// and via inheritance for instances of #mi::neuraylib::IGroup and #mi::neuraylib::IInstance.
/// - struct Approx \b approx_curve \n
///   This attribute controls the subdivision level of curve segments of freeform surfaces. The
///   attribute is a struct and has the same structure as the "approx" attribute (see above). Note
///   that there is only one attribute for all curve segments together.
/// \par
/// The following attributes are only meaningful for instances of #mi::neuraylib::ITriangle_mesh,
/// #mi::neuraylib::IPolygon_mesh, #mi::neuraylib::ISubdivision_surface,
/// #mi::neuraylib::IFreeform_surface, #mi::neuraylib::IOn_demand_mesh,
/// and via inheritance for instances of #mi::neuraylib::IGroup and #mi::neuraylib::IInstance.
/// - #mi::IArray \b decals \n
///   An array of references (#mi::IRef) that attaches the given decals or instances of decals
///   to the scene element. This is similar to the \c material attribute, however, note that
///   in contrast to the \c material attribute the scene element with the \c decals attribute
///   influences the world-to-object transformation of the decal. This attribute is inherited as
///   usual with the exception that when a parent node P and a child node C both have the \c decals
///   attribute the rules detailed in #mi::neuraylib::Propagation_type do not apply. Instead, the
///   array elements of the \c decals attribute in P are appended to the array elements from C.
/// - #mi::IArray \b enabled_decals \n
///   An array of references (#mi::IRef) that acts as filter to determine the active decals.
///   Only decals in that array can be active. Defaults to the (inherited) value of the \c decals
///   attribute.
/// - #mi::IArray \b disabled_decals \n
///   An array of references (#mi::IRef) that acts as filter to determine the active decals.
///   Decals in that array are never active. Defaults to the empty array.
/// \par
/// The list of active decals at a geometry node is given by the intersection of \c decals and
/// \c enabled_decals minus \c disabled_decals (taking attribute inheritance into account).
/// \par
/// The element order in the \c decals attribute is used to break ties between decals of equal
/// priority: if decal D1 is in front of decal D2 and both have equal priorities, decal D1 will
/// appear on top of decal D2 (assuming both are overlapping, otherwise it is not relevant anyway).
/// \par
/// The following attributes are only meaningful for instances of #mi::neuraylib::ITriangle_mesh,
/// #mi::neuraylib::IPolygon_mesh, #mi::neuraylib::ISubdivision_surface,
/// #mi::neuraylib::IFreeform_surface, #mi::neuraylib::IOn_demand_mesh,
/// and via inheritance for instances of #mi::neuraylib::IGroup and #mi::neuraylib::IInstance.
/// - #mi::IArray \b projectors \n
///   An array of references (#mi::IRef) that attaches the given projectors or instances of
///   projectors to the scene element. This is similar to the \c material attribute, however, note
///   that in contrast to the \c material attribute the scene element with the \c projectors
///   attribute influences the world-to-object transformation of the projector. This attribute is
///   inherited as usual with the exception that the propagation type
///   #mi::neuraylib::PROPAGATION_OVERRIDE is not supported.
/// - #mi::IRef \b active_projector \n
///   Specifies the active projector. Defaults to the (inherited) first element of the \c projector
///   attribute. This attribute allows to specify a projector different from the inherited default,
///   or to disable the inherited default. Only projectors that appear in \c projectors attributes
///   on the path from this node to the root are feasible.
///
/// \see
/// The free functions #mi::set_value() and #mi::get_value() including the various specializations
/// may help to write/read values to/from attributes. Note that there are variants operating on
/// \link mi_neuray_scene_element attributes sets \endlink as well as directly on instances of
/// \link mi_neuray_types mi::IData \endlink.
///
/// \par Example
/// The following code snippet is an example that shows how to copy all attributes from one
/// attribute set to a different one. Three boolean flags allow to customize its behavior. Note
/// that the function stops as soon as the assignment fails for one attribute (which might
/// happen if \c adjust_attribute_types is \c false and the types are incompatible).
///
/// \code
/// mi::Sint32 copy_attribute(
///     mi::neuraylib::IFactory* factory,
///     const mi::neuraylib::IAttribute_set* source,
///     mi::neuraylib::IAttribute_set* target,
///     bool create_missing_attributes,
///     bool remove_excess_attributes,
///     bool adjust_attribute_types)
/// {
///     if( !factory || !source || !target)
///         return -1;
///
///      mi::Sint32 index = 0;
///      const char* name;
///      while( name = source->enumerate_attributes( index++)) {
///
///          mi::base::Handle<const mi::IData> source_attr(
///              source->access_attribute<mi::IData>( name));
///          std::string source_type = source_attr->get_type_name();
///
///          mi::base::Handle<mi::IData> target_attr;
///          if( !target->is_attribute( name)) {
///             if( !create_missing_attributes)
///                 continue; // skip attribute
///             else
///                 target_attr = target->create_attribute<mi::IData>( name, source_type.c_str());
///         } else {
///             if( adjust_attribute_types
///                 && source_type != target->get_attribute_type_name( name)) {
///                 target->destroy_attribute( name);
///                 target_attr = target->create_attribute<mi::IData>( name, source_type.c_str());
///             } else
///                 target_attr = target->edit_attribute<mi::IData>( name);
///         }
///
///         mi::Sint32 result = factory->assign_from_to( source_attr.get(), target_attr.get());
///         if( result != 0)
///             return -2;
///      }
///
///      if( !remove_excess_attributes)
///          return 0;
///
///      index = 0;
///      std::vector<std::string> to_be_removed;
///      while( name = target->enumerate_attributes( index++))
///          if( !source->is_attribute( name))
///              to_be_removed.push_back( name);
///      for( mi::Size i = 0; i < to_be_removed.size(); ++i)
///          target->destroy_attribute( to_be_removed[i].c_str());
///
///      return 0;
/// }
/// \endcode
///
/// \endif
class IAttribute_set :
    public base::Interface_declare<0x1bcb8d48,0x10c1,0x4b3e,0x9b,0xfa,0x06,0x23,0x61,0x81,0xd3,0xe1>
{
public:
    /// Creates a new attribute \p name of the type \p type.
    ///
    /// \param name         The name of the attribute. The name must not contain \c "[", \c "]", or
    ///                     \c "."
    /// \param type         The type of the attribute. See \ref mi_neuray_types for a list of
    ///                     supported attribute types.
    /// \return             A pointer to the created attribute, or \c NULL in case of failure.
    ///                     Reasons for failure are:
    ///                     - \p name or \p type is invalid,
    ///                     - there is already an attribute with the name \p name, or
    ///                     - \p name is the name of a reserved attribute and \p type does not match
    ///                       the required type(s) of such an attribute.
    virtual IData* create_attribute( const char* name, const char* type) = 0;

    /// Creates a new attribute \p name of the type \p type.
    ///
    /// See \ref mi_neuray_types for a list of supported attribute types.
    ///
    /// Note that there are two versions of this templated member function, one that takes only one
    /// argument (the attribute name), and another one that takes two arguments (the attribute name
    /// and the type name). The version with one argument can only be used to create a subset of
    /// supported attribute types: it supports only those types where the type name can be deduced
    /// from the template parameter, i.e., it does not support arrays and structures. The version
    /// with two arguments can be used to create attributes of any supported type (but requires the
    /// type name as parameter, which for redundant for many types). Attempts to use the version
    /// with one argument with a template parameter where the type name can not be deduced results
    /// in compiler errors.
    ///
    /// This templated member function is a wrapper of the non-template variant for the user's
    /// convenience. It eliminates the need to call
    /// #mi::base::IInterface::get_interface(const Uuid &)
    /// on the returned pointer, since the return type already is a pointer to the type \p T
    /// specified as template parameter.
    ///
    /// \tparam T           The interface type of the attribute.
    /// \param name         The name of the attribute. The name must not contain \c "[", \c "]", or
    ///                     \c "."
    /// \param type         The type of the attribute. See \ref mi_neuray_types for a list of
    ///                     supported attribute types.
    /// \return             A pointer to the created attribute, or \c NULL in case of failure.
    ///                     Reasons for failure are:
    ///                     - \p name or \p type is invalid,
    ///                     - there is already an attribute with the name \p name,
    ///                     - \p name is the name of a reserved attribute and \p type does not match
    ///                       the required type(s) of such an attribute, or
    ///                     - \p T does not match \p type.
    template<class T>
    T* create_attribute( const char* name, const char* type)
    {
        IData* ptr_iinterface = create_attribute( name, type);
        if ( !ptr_iinterface)
            return 0;
        T* ptr_T = static_cast<T*>( ptr_iinterface->get_interface( typename T::IID()));
        ptr_iinterface->release();
        return ptr_T;
    }

    /// Creates a new attribute \p name of the type \p T.
    ///
    /// See \ref mi_neuray_types for a list of supported attribute types.
    ///
    /// Note that there are two versions of this templated member function, one that takes only one
    /// argument (the attribute name), and another one that takes two arguments (the attribute name
    /// and the type name). The version with one argument can only be used to create a subset of
    /// supported attribute types: it supports only those types where the type name can be deduced
    /// from the template parameter, i.e., it does not support arrays and structures. The version
    /// with two arguments can be used to create attributes of any supported type (but requires the
    /// type name as parameter, which is redundant for many types). Attempts to use the version
    /// with one argument with a template parameter where the type name can not be deduced results
    /// in compiler errors.
    ///
    /// This templated member function is a wrapper of the non-template variant for the user's
    /// convenience. It eliminates the need to call
    /// #mi::base::IInterface::get_interface(const Uuid &)
    /// on the returned pointer, since the return type already is a pointer to the type \p T
    /// specified as template parameter.
    ///
    /// \tparam T           The interface type of the attribute.
    /// \param name         The name of the attribute. The name must not contain \c "[", \c "]", or
    ///                     \c "."
    /// \return             A pointer to the created attribute, or \c NULL in case of failure.
    ///                     Reasons for failure are:
    ///                     - \p name or \p type is invalid,
    ///                     - there is already an attribute with the name \p name, or
    ///                     - \p name is the name of a reserved attribute and \p T does not match
    ///                       the required type(s) of such an attribute.
    template<class T>
    T* create_attribute( const char* name)
    {
        return create_attribute<T>( name, Type_traits<T>::get_type_name());
    }

    /// Destroys the attribute \p name.
    ///
    /// \param name         The name of the attribute to destroy.
    /// \return             Returns \c true if the attribute has been successfully destroyed, and
    ///                     \c false otherwise (there is no attribute with the name \p name).
    virtual bool destroy_attribute( const char* name) = 0;

    /// Returns a const pointer to the attribute \p name.
    ///
    /// \param name         The name of the attribute. In addition, you can also access parts of
    ///                     array or structure attributes directly. For an array element add
    ///                     the index in square brackets to the attribute name. For a structure
    ///                     member add a dot and the name of the structure member to the attribute
    ///                     name.
    /// \return             A pointer to the attribute, or \c NULL if there is no attribute with
    ///                     the name \p name.
    virtual const IData* access_attribute( const char* name) const = 0;

    /// Returns a const pointer to the attribute \p name.
    ///
    /// This templated member function is a wrapper of the non-template variant for the user's
    /// convenience. It eliminates the need to call
    /// #mi::base::IInterface::get_interface(const Uuid &)
    /// on the returned pointer, since the return type already is a pointer to the type \p T
    /// specified as template parameter.
    ///
    /// \tparam T           The interface type of the attribute.
    /// \param name         The name of the attribute. In addition, you can also access parts of
    ///                     array or structure attributes directly. For an array element add
    ///                     the index in square brackets to the attribute name. For a structure
    ///                     member add a dot and the name of the structure member to the attribute
    ///                     name.
    /// \return             A pointer to the attribute, or \c NULL if there is no attribute with
    ///                     the name \p name or if \p T does not match the attribute's type.
    template<class T>
    const T* access_attribute( const char* name) const
    {
        const IData* ptr_iinterface = access_attribute( name);
        if ( !ptr_iinterface)
            return 0;
        const T* ptr_T = static_cast<const T*>( ptr_iinterface->get_interface( typename T::IID()));
        ptr_iinterface->release();
        return ptr_T;
    }

    /// Returns a mutable pointer to the attribute \p name.
    ///
    /// \param name         The name of the attribute. In addition, you can also access parts of
    ///                     array or structure attributes directly. For an array element add
    ///                     the index in square brackets to the attribute name. For a structure
    ///                     member add a dot and the name of the structure member to the attribute
    ///                     name.
    /// \return             A pointer to the attribute, or \c NULL if there is no attribute with
    ///                     the name \p name.
    virtual IData* edit_attribute( const char* name) = 0;

    /// Returns a mutable pointer to the attribute \p name.
    ///
    /// This templated member function is a wrapper of the non-template variant for the user's
    /// convenience. It eliminates the need to call
    /// #mi::base::IInterface::get_interface(const Uuid &)
    /// on the returned pointer, since the return type already is a pointer to the type \p T
    /// specified as template parameter.
    ///
    /// \tparam T           The interface type of the attribute.
    /// \param name         The name of the attribute. In addition, you can also access parts of
    ///                     array or structure attributes directly. For an array element add
    ///                     the index in square brackets to the attribute name. For a structure
    ///                     member add a dot and the name of the structure member to the attribute
    ///                     name.
    /// \return             A pointer to the attribute, or \c NULL if there is no attribute with
    ///                     the name \p name or if \p T does not match the attribute's type.
    template<class T>
    T* edit_attribute( const char* name)
    {
        IData* ptr_iinterface = edit_attribute( name);
        if ( !ptr_iinterface)
            return 0;
        T* ptr_T = static_cast<T*>( ptr_iinterface->get_interface( typename T::IID()));
        ptr_iinterface->release();
        return ptr_T;
    }

    /// Indicates existence of an attribute.
    ///
    /// \param name         The name of the attribute. In addition, you can also checks for parts of
    ///                     array or structure attributes directly. For an array element add
    ///                     the index in square brackets to the attribute name. For a structure
    ///                     member add a dot and the name of the structure member to the attribute
    ///                     name.
    /// \return             \c true if the attribute set contains this attribute (and the attribute
    ///                     contains the requested array element or struct member),
    ///                     \c false otherwise
    virtual bool is_attribute( const char* name) const = 0;

    /// Returns the type of an attribute.
    ///
    /// See \ref mi_neuray_types for a list of supported attribute types.
    ///
    /// \param name         The name of the attribute. In addition, you can also query parts of
    ///                     array or structure attributes directly. For an array element add
    ///                     the index in square brackets to the attribute name. For a structure
    ///                     member add a dot and the name of the structure member to the attribute
    ///                     name.
    /// \return             The type name of the attribute (or part thereof), or \c NULL if there
    ///                     is no attribute with the name \p name.
    ///
    /// \note The return value of this method is only valid until the next call of this method
    ///       or any non-const methods on this instance.
    virtual const char* get_attribute_type_name( const char* name) const = 0;

    /// Sets the propagation type of the attribute \p name.
    ///
    /// \return
    ///                     -  0: Success.
    ///                     - -1: Invalid parameters (\c NULL pointer or invalid enum value).
    ///                     - -2: There is no attribute with name \p name.
    virtual Sint32 set_attribute_propagation( const char* name, Propagation_type value) = 0;

    /// Returns the propagation type of the attribute \p name.
    ///
    /// \note This method always returns #PROPAGATION_STANDARD in case of errors.
    virtual Propagation_type get_attribute_propagation( const char* name) const = 0;

    /// Returns the name of the attribute indicated by \p index.
    ///
    /// \param index        The index of the attribute.
    /// \return             The name of the attribute indicated by \p index, or \c NULL if \p index
    ///                     is out of bounds.
    virtual const char* enumerate_attributes( Sint32 index) const = 0;
};

/*@}*/ // end group mi_neuray_scene_element

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_IATTRIBUTE_SET_H
