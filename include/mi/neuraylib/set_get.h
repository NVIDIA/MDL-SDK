/***************************************************************************************************
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
 **************************************************************************************************/
/// \file
/// \brief Helper functions to set/get values of #mi::IData.

#ifndef MI_NEURAYLIB_SET_GET_H
#define MI_NEURAYLIB_SET_GET_H

#include <mi/base/config.h>
#include <mi/base/handle.h>
#include <mi/neuraylib/iattribute_set.h>
#include <mi/neuraylib/ibbox.h>
#include <mi/neuraylib/icolor.h>
#include <mi/neuraylib/ienum.h>
#include <mi/neuraylib/imatrix.h>
#include <mi/neuraylib/inumber.h>
#include <mi/neuraylib/iref.h>
#include <mi/neuraylib/ispectrum.h>
#include <mi/neuraylib/istring.h>
#include <mi/neuraylib/iuuid.h>
#include <mi/neuraylib/ivector.h>
#include <mi/neuraylib/type_traits.h>

namespace mi {

/** \addtogroup mi_neuray_types
@{
*/

/// Simplifies setting the value of #mi::IData from the corresponding classes from the %base and
/// %math API.
///
/// \param data            The instance of #mi::IData to modify.
/// \param value           The new value to be set.
/// \return
///                        -  0: Success.
///                        - -1: The dynamic type of \p data does not match the static type of
///                              \p value.
///                        - -2: The value of \p value is not valid.
///
/// This general template handles #mi::INumber and expects one of the following types as second
/// argument:
/// - \c bool,
/// - #mi::Sint8, #mi::Sint16, #mi::Sint32, #mi::Sint64,
/// - #mi::Uint8, #mi::Uint16, #mi::Uint32, #mi::Uint64,
/// - #mi::Size, #mi::Difference
/// - #mi::Float32, or #mi::Float64.
///
/// It also handles #mi::IEnum and expects an #mi::Sint32 argument in that case.
template<class T>
mi::Sint32 set_value( mi::IData* data, const T& value)
{
    mi::base::Handle<mi::INumber> v( data->get_interface<mi::INumber>());
    if( v.is_valid_interface()) {
        v->set_value( value);
        return 0;
    }
    mi::base::Handle<mi::IEnum> e( data->get_interface<mi::IEnum>());
    if( e.is_valid_interface()) {
        mi::Sint32 result = e->set_value( static_cast<mi::Sint32>( value));
        return result == 0 ? 0 : -2;
    }
    return -1;
}

/// This specialization handles #mi::IString and #mi::IRef.
///
/// It expects a \c const \c char* argument. See #mi::set_value() for details.
inline mi::Sint32 set_value( mi::IData* data, const char* value)
{
    mi::base::Handle<mi::IString> s( data->get_interface<mi::IString>());
    if( s.is_valid_interface()) {
        s->set_c_str( value);
        return 0;
    }
    mi::base::Handle<mi::IRef> r( data->get_interface<mi::IRef>());
    if( r.is_valid_interface()) {
        mi::Sint32 result = r->set_reference( value);
        return result == 0 ? 0 : -2;
    }
    mi::base::Handle<mi::IEnum> e( data->get_interface<mi::IEnum>());
    if( e.is_valid_interface()) {
        mi::Sint32 result = e->set_value_by_name( value);
        return result == 0 ? 0 : -2;
    }
    return -1;
}

/// This specialization handles #mi::IString and #mi::IRef.
///
/// It expects a \c const std::string& argument. See #mi::set_value() for details.
inline mi::Sint32 set_value( mi::IData* data, const std::string& value)
{
    return set_value(data,value.c_str());
}

/// This specialization handles #mi::IUuid.
///
/// It expects an #mi::base::Uuid argument. See #mi::set_value() for details.
inline mi::Sint32 set_value( mi::IData* data, const mi::base::Uuid& value)
{
    mi::base::Handle<mi::IUuid> u( data->get_interface<mi::IUuid>());
    if( u.is_valid_interface()) {
        u->set_uuid( value);
        return 0;
    }
    return -1;
}

/// This specialization handles the vector specializations of #mi::ICompound.
///
/// It expects one of the following specializations of #mi::math::Vector as its second argument:
/// - #mi::Boolean_2, #mi::Boolean_3, #mi::Boolean_4
/// - #mi::Sint32_2, #mi::Sint32_3, #mi::Sint32_4
/// - #mi::Uint32_2, #mi::Uint32_3, #mi::Uint32_4
/// - #mi::Float32_2, #mi::Float32_3, #mi::Float32_4
/// - #mi::Float64_2, #mi::Float64_3, #mi::Float64_4
///
/// See #mi::set_value() for details.
template <class T, Size DIM>
inline mi::Sint32 set_value( mi::IData* data, const mi::math::Vector<T,DIM>& value)
{
    typedef typename mi::Vector_type_traits<T,DIM>::Interface_type Vector_interface_type;
    mi::base::Handle<Vector_interface_type> v( data->get_interface<Vector_interface_type>());
    if( v.is_valid_interface()) {
        v->set_value( value);
        return 0;
    }
    return -1;
}

/// This specialization handles the matrix specializations of #mi::ICompound.
///
/// It expects one of the following specializations of #mi::math::Matrix as its second argument:
/// - #mi::Boolean_2_2, #mi::Boolean_2_3, ..., #mi::Boolean_4_4
/// - #mi::Sint32_2_2, #mi::Sint32_2_3, ..., #mi::Sint32_4_4
/// - #mi::Uint32_2_2, #mi::Uint32_2_3, ..., #mi::Uint32_4_4
/// - #mi::Float32_2_2, #mi::Float32_2_3,...,  #mi::Float32_4_4
/// - #mi::Float64_2_2, #mi::Float64_2_3,...,  #mi::Float64_4_4
///
/// See #mi::set_value() for details.
template <class T, Size ROW, Size COL>
mi::Sint32 set_value( mi::IData* data, const mi::math::Matrix<T,ROW,COL>& value)
{
    typedef typename mi::Matrix_type_traits<T,ROW,COL>::Interface_type Matrix_interface_type;
    mi::base::Handle<Matrix_interface_type> m( data->get_interface<Matrix_interface_type>());
    if( m.is_valid_interface()) {
        m->set_value( value);
        return 0;
    }
    return -1;
}

/// This specialization handles #mi::IColor and #mi::IColor3.
///
/// It expects an #mi::Color argument. See #mi::set_value() for details.
inline mi::Sint32 set_value( mi::IData* data, const mi::Color& value)
{
    mi::base::Handle<mi::IColor> c( data->get_interface<mi::IColor>());
    if( c.is_valid_interface()) {
        c->set_value( value);
        return 0;
    }
    mi::base::Handle<mi::IColor3> c3( data->get_interface<mi::IColor3>());
    if( c3.is_valid_interface()) {
        c3->set_value( value);
        return 0;
    }
    return -1;
}

/// This specialization handles #mi::ISpectrum.
///
/// It expects an #mi::Spectrum argument. See #mi::set_value() for details.
inline mi::Sint32 set_value( mi::IData* data, const mi::Spectrum& value)
{
    mi::base::Handle<mi::ISpectrum> s( data->get_interface<mi::ISpectrum>());
    if( s.is_valid_interface()) {
        s->set_value( value);
        return 0;
    }
    return -1;
}

/// This specialization handles #mi::IBbox3.
///
/// It expects an #mi::Bbox3 argument. See #mi::set_value() for details.
inline mi::Sint32 set_value( mi::IData* data, const mi::Bbox3& value)
{
    mi::base::Handle<mi::IBbox3> b( data->get_interface<mi::IBbox3>());
    if( b.is_valid_interface()) {
        b->set_value( value);
        return 0;
    }
    return -1;
}

/// This variant handles elements of collections identified via an additional index.
///
/// \param data            The instance of #mi::IData to modify.
/// \param index           The index of the affected collection element.
/// \param value           The new value to be set.
/// \return
///                        -  0: Success.
///                        - -1: The dynamic type of \p data does not match the static type of
///                              \p value.
///                        - -2: The value of \p value is not valid.
///                        - -3: The index or key is not valid for the collection.
template<class T>
mi::Sint32 set_value( mi::IData* data, mi::Size index, const T& value)
{
    mi::base::Handle<mi::IData_collection> c( data->get_interface<mi::IData_collection>());
    if( c.is_valid_interface()) {
        mi::base::Handle<mi::IData> d( c->get_value<mi::IData>( index));
        if( !d.is_valid_interface())
            return -3;
        return set_value( d.get(), value);
    }
    return -1;
}

/// This variant handles elements of collections identified via an additional key.
///
/// \param data            The instance of #mi::IData to modify.
/// \param key             The key of the affected collection element.
/// \param value           The new value to be set.
/// \return
///                        -  0: Success.
///                        - -1: The dynamic type of \p data does not match the static type of
///                              \p value.
///                        - -2: The value of \p value is not valid.
///                        - -3: The key is not valid.
template<class T>
mi::Sint32 set_value( mi::IData* data, const char* key, const T& value)
{
    mi::base::Handle<mi::IData_collection> c( data->get_interface<mi::IData_collection>());
    if( c.is_valid_interface()) {
        mi::base::Handle<mi::IData> d( c->get_value<mi::IData>( key));
        if( !d.is_valid_interface())
            return -3;
        return set_value( d.get(), value);
    }
    return -1;
}

/// Simplifies reading the value of #mi::IData into the corresponding classes from the %base and
/// %math API.
///
/// \param data            The instance of #mi::IData to read.
/// \param[out] value      The new value will be stored here.
/// \return
///                        -  0: Success.
///                        - -1: The dynamic type of \p data does not match the static type of
///                              \p value.
///
/// This general template handles #mi::INumber, and expects one of the following types as second
/// argument:
/// - \c bool,
/// - #mi::Sint8, #mi::Sint16, #mi::Sint32, #mi::Sint64,
/// - #mi::Uint8, #mi::Uint16, #mi::Uint32, #mi::Uint64,
/// - #mi::Size, #mi::Difference
/// - #mi::Float32, or #mi::Float64.
///
/// It also handles #mi::IEnum and expects an #mi::Sint32 argument in that case.
template<class T>
mi::Sint32 get_value( const mi::IData* data, T& value)
{
    mi::base::Handle<const mi::INumber> v( data->get_interface<mi::INumber>());
    if( v.is_valid_interface()) {
        v->get_value( value);
        return 0;
    }
// disable C4800: 'mi::Sint32' : forcing value to bool 'true' or 'false' (performance warning)
// disable C4800: 'mi::Uint32' : forcing value to bool 'true' or 'false' (performance warning)
#ifdef MI_COMPILER_MSC
#pragma warning( push )
#pragma warning( disable : 4800 )
#endif
    mi::base::Handle<const mi::IEnum> e( data->get_interface<mi::IEnum>());
    if( e.is_valid_interface()) {
        value = static_cast<T>( e->get_value());
        return 0;
    }
#ifdef MI_COMPILER_MSC
#pragma warning( pop )
#endif
    return -1;
}

/// This specialization handles #mi::IString and #mi::IRef.
///
/// It expects a \c const \c char* argument. See #mi::get_value() for details.
inline mi::Sint32 get_value( const mi::IData* data, const char*& value)
{
    mi::base::Handle<const mi::IString> i( data->get_interface<mi::IString>());
    if( i.is_valid_interface()) {
        value = i->get_c_str();
        return 0;
    }
    mi::base::Handle<const mi::IRef> r( data->get_interface<mi::IRef>());
    if( r.is_valid_interface()) {
        value = r->get_reference_name();
        return 0;
    }
    mi::base::Handle<const mi::IEnum> e( data->get_interface<mi::IEnum>());
    if( e.is_valid_interface()) {
        value = e->get_value_by_name();
        return 0;
    }
    return -1;
}

/// This specialization handles #mi::IUuid.
///
/// It expects an #mi::base::Uuid argument. See #mi::get_value() for details.
inline mi::Sint32 get_value( const mi::IData* data, mi::base::Uuid& value)
{
    mi::base::Handle<const mi::IUuid> u( data->get_interface<mi::IUuid>());
    if( u.is_valid_interface()) {
        value = u->get_uuid();
        return 0;
    }
    return -1;
}

/// This specialization handles the vector specializations of #mi::ICompound.
///
/// It expects one of the following specializations of #mi::math::Vector as its second argument:
/// - #mi::Boolean_2, #mi::Boolean_3, #mi::Boolean_4
/// - #mi::Sint32_2, #mi::Sint32_3, #mi::Sint32_4
/// - #mi::Uint32_2, #mi::Uint32_3, #mi::Uint32_4
/// - #mi::Float32_2, #mi::Float32_3, #mi::Float32_4
/// - #mi::Float64_2, #mi::Float64_3, #mi::Float64_4
///
/// See #mi::get_value() for details.
template <class T, Size DIM>
inline mi::Sint32 get_value( const mi::IData* data, mi::math::Vector<T,DIM>& value)
{
    typedef typename mi::Vector_type_traits<T,DIM>::Interface_type Vector_interface_type;
    mi::base::Handle<const Vector_interface_type> v( data->get_interface<Vector_interface_type>());
    if( v.is_valid_interface()) {
        value = v->get_value();
        return 0;
    }
    return -1;
}

/// This specialization handles the matrix specializations of #mi::ICompound.
///
/// It expects one of the following specializations of #mi::math::Matrix as its second argument:
/// - #mi::Boolean_2_2, #mi::Boolean_2_3, ..., #mi::Boolean_4_4
/// - #mi::Sint32_2_2, #mi::Sint32_2_3, ..., #mi::Sint32_4_4
/// - #mi::Uint32_2_2, #mi::Uint32_2_3, ..., #mi::Uint32_4_4
/// - #mi::Float32_2_2, #mi::Float32_2_3,...,  #mi::Float32_4_4
/// - #mi::Float64_2_2, #mi::Float64_2_3,...,  #mi::Float64_4_4
///
/// See #mi::get_value() for details.
template <class T, Size ROW, Size COL>
mi::Sint32 get_value( const mi::IData* data, mi::math::Matrix<T,ROW,COL>& value)
{
    typedef typename mi::Matrix_type_traits<T,ROW,COL>::Interface_type Matrix_interface_type;
    mi::base::Handle<const Matrix_interface_type> m( data->get_interface<Matrix_interface_type>());
    if( m.is_valid_interface()) {
        value = m->get_value();
        return 0;
    }
    return -1;
}

/// This specialization handles #mi::IColor and #mi::IColor3.
///
/// It expects an #mi::Color argument. See #mi::get_value() for details.
inline mi::Sint32 get_value( const mi::IData* data, mi::Color& value)
{
    mi::base::Handle<const mi::IColor> c( data->get_interface<mi::IColor>());
    if( c.is_valid_interface()) {
        value = c->get_value();
        return 0;
    }
    mi::base::Handle<const mi::IColor3> c3( data->get_interface<mi::IColor3>());
    if( c3.is_valid_interface()) {
        value = c3->get_value();
        return 0;
    }
    return -1;
}

/// This specialization handles #mi::ISpectrum.
///
/// It expects an #mi::Spectrum argument. See #mi::get_value() for details.
inline mi::Sint32 get_value( const mi::IData* data, mi::Spectrum& value)
{
    mi::base::Handle<const mi::ISpectrum> s( data->get_interface<mi::ISpectrum>());
    if( s.is_valid_interface()) {
        value = s->get_value();
        return 0;
    }
    return -1;
}

/// This specialization handles #mi::IBbox3.
///
/// It expects an #mi::Bbox3 argument. See #mi::get_value() for details.
inline mi::Sint32 get_value( const mi::IData* data, mi::Bbox3& value)
{
    mi::base::Handle<const mi::IBbox3> b( data->get_interface<mi::IBbox3>());
    if( b.is_valid_interface()) {
        value = b->get_value();
        return 0;
    }
    return -1;
}

/// This variant handles elements of collections identified via an additional index.
///
/// \param data            The instance of #mi::IData to read.
/// \param index           The index of the affected collection element.
/// \param value           The new value will be stored here.
/// \return
///                        -  0: Success.
///                        - -1: The dynamic type of \p data does not match the static type of
///                              \p value.
///                        - -3: The index is not valid.
template<class T>
mi::Sint32 get_value( const mi::IData* data, mi::Size index, T& value)
{
    mi::base::Handle<const mi::IData_collection> c( data->get_interface<mi::IData_collection>());
    if( c.is_valid_interface()) {
        mi::base::Handle<const mi::IData> d( c->get_value<mi::IData>( index));
        if( !d.is_valid_interface())
            return -3;
        return get_value( d.get(), value);
    }
    return -1;
}

/// This variant handles elements of collections identified via an additional key.
///
/// \param data            The instance of #mi::IData to read.
/// \param key             The key of the affected collection element.
/// \param value           The new value will be stored here.
/// \return
///                        -  0: Success.
///                        - -1: The dynamic type of \p data does not match the static type of
///                              \p value.
///                        - -3: The key is not valid.
template<class T>
mi::Sint32 get_value( const mi::IData* data, const char* key, T& value)
{
    mi::base::Handle<const mi::IData_collection> c( data->get_interface<mi::IData_collection>());
    if( c.is_valid_interface()) {
        mi::base::Handle<const mi::IData> d( c->get_value<mi::IData>( key));
        if( !d.is_valid_interface())
            return -3;
        return get_value( d.get(), value);
    }
    return -1;
}

/*@}*/ // end group mi_neuray_types

/** \addtogroup mi_neuray_scene_element
@{
*/

/// Simplifies setting the value of an attribute from the corresponding classes from the %base and
/// %math API.
///
/// \param attribute_set   The affected attribute set.
/// \param name            The name of the attribute to modify.
/// \param value           The new value to be set.
/// \return
///                        -  0: Success.
///                        - -1: The dynamic type of the attribute does not match the static type of
///                              \p value.
///                        - -2: The value of \p value is not valid.
///                        - -4: The attribute \p name does not exist.
template<class T>
mi::Sint32 set_value(
    mi::neuraylib::IAttribute_set* attribute_set, const char* name, const T& value)
{
    mi::base::Handle<mi::IData> data( attribute_set->edit_attribute<mi::IData>( name));
    if( !data.is_valid_interface())
        return -4;
    return set_value( data.get(), value);
}

/// Simplifies setting the value of an attribute from the corresponding classes from the %base and
/// %math API (variant with an index for collections).
///
/// \param attribute_set   The affected attribute set.
/// \param name            The name of the attribute to modify.
/// \param index           The index of the affected collection element.
/// \param value           The new value to be set.
/// \return
///                        -  0: Success.
///                        - -1: The dynamic type of the attribute does not match the static type of
///                              \p value.
///                        - -2: The value of \p value is not valid.
///                        - -3: The index is not valid.
///                        - -4: The attribute \p name does not exist.
template<class T>
mi::Sint32 set_value(
    mi::neuraylib::IAttribute_set* attribute_set, const char* name, mi::Size index, const T& value)
{
    mi::base::Handle<mi::IData> data( attribute_set->edit_attribute<mi::IData>( name));
    if( !data.is_valid_interface())
        return -4;
    return set_value( data.get(), index, value);
}

/// Simplifies setting the value of an attribute from the corresponding classes from the %base and
/// %math API (variant with a key for collections).
///
/// \param attribute_set   The affected attribute set.
/// \param name            The name of the attribute to modify.
/// \param key             The key of the affected collection element.
/// \param value           The new value to be set.
/// \return
///                        -  0: Success.
///                        - -1: The dynamic type of the attribute does not match the static type of
///                              \p value.
///                        - -2: The value of \p value is not valid.
///                        - -3: The key is not valid.
///                        - -4: The attribute \p name does not exist.
template<class T>
mi::Sint32 set_value(
    mi::neuraylib::IAttribute_set* attribute_set, const char* name, const char* key, const T& value)
{
    mi::base::Handle<mi::IData> data( attribute_set->edit_attribute<mi::IData>( name));
    if( !data.is_valid_interface())
        return -4;
    return set_value( data.get(), key, value);
}

/// Simplifies reading the value of an attribute into the corresponding classes from the %base and
/// %math API.
///
/// \param attribute_set   The affected attribute set.
/// \param name            The name of the attribute to read.
/// \param value           The new value will be stored here.
/// \return
///                        -  0: Success.
///                        - -1: The dynamic type of the attribute does not match the static type of
///                              \p value.
///                        - -4: The attribute \p name does not exist.
template<class T>
mi::Sint32 get_value(
    const mi::neuraylib::IAttribute_set* attribute_set, const char* name, T& value)
{
    mi::base::Handle<const mi::IData> data( attribute_set->access_attribute<mi::IData>( name));
    if( !data.is_valid_interface())
        return -4;
    return get_value( data.get(), value);
}

/// Simplifies reading the value of an attribute into the corresponding classes from the %base and
/// %math API (variant with an index for collections).
///
/// \param attribute_set   The affected attribute set.
/// \param name            The name of the attribute to read.
/// \param index           The index of the affected collection element.
/// \param value           The new value will be stored here.
/// \return
///                        -  0: Success.
///                        - -1: The dynamic type of the attribute does not match the static type of
///                              \p value.
///                        - -3: The index is not valid.
///                        - -4: The attribute \p name does not exist.
template<class T>
mi::Sint32 get_value(
    const mi::neuraylib::IAttribute_set* attribute_set, const char* name, mi::Size index, T& value)
{
    mi::base::Handle<const mi::IData> data( attribute_set->access_attribute<mi::IData>( name));
    if( !data.is_valid_interface())
        return -4;
    return get_value( data.get(), index, value);
}

/// Simplifies reading the value of an attribute into the corresponding classes from the %base and
/// %math API (variant with a key for collections).
///
/// \param attribute_set   The affected attribute set.
/// \param name            The name of the attribute to read.
/// \param key             The key of the affected collection element.
/// \param value           The new value will be stored here.
/// \return
///                        -  0: Success.
///                        - -1: The dynamic type of the attribute does not match the static type of
///                              \p value.
///                        - -3: The key is not valid.
///                        - -4: The attribute \p name does not exist.
template<class T>
mi::Sint32 get_value(
    const mi::neuraylib::IAttribute_set* attribute_set, const char* name, const char* key, T& value)
{
    mi::base::Handle<const mi::IData> data( attribute_set->access_attribute<mi::IData>( name));
    if( !data.is_valid_interface())
        return -4;
    return get_value( data.get(), key, value);
}

/*@}*/ // end group mi_neuray_types

} // namespace mi

#endif // MI_NEURAYLIB_SET_GET_H
