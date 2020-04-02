/***************************************************************************************************
 * Copyright (c) 2008-2020, NVIDIA CORPORATION. All rights reserved.
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

/// \file i_scene_scene_element.h
/// \brief The Scene_element definition.

#ifndef IO_SCENE_SCENE_I_SCENE_ELEMENT_H
#define IO_SCENE_SCENE_I_SCENE_ELEMENT_H

#include "i_scene_scene_element_base.h"

namespace MI
{
namespace SCENE
{

/// This template defines some functions for derived classes. This makes it
/// easier to define a new \c Scene_element class:
/// If the class shall directly derived from from \c Scene_element_base just do it
/// like this
/// \code
///     class Test_element : public Scene_element<Test_element, 1000>
///         ...
/// \endcode
/// If it is derived from some other class which is derived from
/// \c Scene_element_base, (e.g. \c Object) then do it like this
/// \code
///     class Test_element : public Scene_element<Test_element, 1000, Object>
///         ...
/// \endcode
/// Note that base classes (e.g. \c Object) do not have to be derived from
/// \c Scene_element but may be derived from \c Scene_element_base. This is because
/// they will not be instantiated directly. The class \c DB::Element is the
/// equivalent to \c Scene_element but without attributes. All database element
/// classes that do not have attribute-sets should be derived from \c DB::Element.
template<class T, SERIAL::Class_id ID = 0, class P = Scene_element_base>
class Scene_element: public P
{
  public:
    static const SERIAL::Class_id id = ID;		///< class id for this

    /// Construct an instance of this class.
    /// \return an instance of this class
    static SERIAL::Serializable *factory();

    /// Default constructor needed because of the copy constructor
    Scene_element();

    /// Copy constructor which will call the copy constructor of the base class.
    /// \param source element to copy construct from
    Scene_element(
        const Scene_element &source);

    /// \name NetworkingFunctions
    /// Scene_element networking functions. See description header in i_db_element.h.
    //@{
    SERIAL::Class_id get_class_id() const;
    DB::Element_base *copy() const;
    size_t get_size() const;
    //@}

    /// Check, if this object is of the given type. This is true, if either the
    /// class id of this object equals the given class id \p id, or the class is
    /// derived from another class which has the given class id \p id.
    /// \param id the class id to check
    /// \return true, if this object IS-A \p id or derives from an \p id class
    bool is_type_of(
        SERIAL::Class_id id) const;
};

}
}

#include "scene_scene_element_inline.h"

#endif
