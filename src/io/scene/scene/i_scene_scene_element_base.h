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

/// \file
/// \brief The Scene_element_base definition.

#ifndef IO_SCENE_SCENE_I_SCENE_ELEMENT_BASE_H
#define IO_SCENE_SCENE_I_SCENE_ELEMENT_BASE_H

#include <base/data/db/i_db_element.h>
#include <base/data/db/i_db_tag.h>
#include <base/data/attr/attr.h>
#include <string>

namespace MI {
namespace SERIAL { class Serializer; class Deserializer; class Serializable; }
namespace SCENE  {

/// The base class for all scene database elements. This class extends the database's
/// \c DB::Element_base with attributes. The attributes may not be a part of the \c Element
/// class because that would give a separate copy for each inheritance hierarchy level.
class Scene_element_base : public DB::Element_base
{
  public:
    /// Get attributes of this Scene_element. Attributes attach runtime flags
    /// and data such as visible flags or shader parameters.
    /// They are collected during scene preprocessing (render/render/traverse) and
    /// accessed during rendering.
    /// One const version for Access and a non-const one for Edit.
    const ATTR::Attribute_set* get_attributes() const { return &m_attributes; }
    ATTR::Attribute_set* get_attributes() { return &m_attributes; }

    /// \name NetworkingFunctions
    /// Scene_element networking functions. See description header in i_db_element.h.
    //@{
    /// Subclasses should not override this, but get_scene_element_references().
    void get_references(
        DB::Tag_set	*result) const;	// return all referenced tags
    size_t get_size() const;
    const SERIAL::Serializable	*serialize(
        SERIAL::Serializer   *ser) const;// useful functions for byte streams
    SERIAL::Serializable	*deserialize(
        SERIAL::Deserializer *deser);	// useful functions for byte streams
    //@}

    /// Return a human readable version of the class id.
    /// \return a human readable version, eg "Group"
    virtual std::string get_class_name() const;

  protected:
    /// Fast exchange of two Scene_element_base.
    /// \param other the other
    void swap(
        Scene_element_base	&other);

    /// Enforce subclass implementation of get_references() by this pure virtual method.
    /// Subclasses should not overwrite get_references(), but this method instead,
    /// it is called by get_references() here.
    /// \param tags Set for all referenced tags.
    virtual void get_scene_element_references(
        DB::Tag_set *tags) const = 0;
        
  private:
    /// \c Scene_elements may have \c ATTR::Attributes attached to them. This is the attribute
    /// anchor, with functions to attach, detach, and lookup attributes.
    ATTR::Attribute_set m_attributes;
};

}
}
#endif
