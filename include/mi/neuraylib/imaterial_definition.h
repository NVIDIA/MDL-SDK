/***************************************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
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
/// \brief      Scene element Material_definition

#ifndef MI_NEURAYLIB_IMATERIAL_DEFINITION_H
#define MI_NEURAYLIB_IMATERIAL_DEFINITION_H

#include <mi/neuraylib/iexpression.h>
#include <mi/neuraylib/iscene_element.h>
#include <mi/neuraylib/ifunction_definition.h>

namespace mi {

namespace neuraylib {

/** \addtogroup mi_neuray_mdl_elements
@{
*/

class IMaterial_instance;
class IMdl_execution_context;

/// This interface represents a material definition.
///
/// This interface is obsolete. See \ref mi_mdl_materials_are_functions and
/// #mi::neuraylib::IFunction_definition.
class IMaterial_definition : public
    mi::base::Interface_declare<0x73753e3d,0x62e4,0x41a7,0xa8,0xf5,0x37,0xeb,0xda,0xd9,0x01,0xd9,
                                neuraylib::IScene_element>
{
public:
#ifdef MI_NEURAYLIB_DEPRECATED_13_0
    virtual const char* get_module() const = 0;

    virtual const char* get_mdl_name() const = 0;

    virtual const char* get_mdl_module_name() const = 0;

    virtual const char* get_mdl_simple_name() const = 0;

    virtual const char* get_mdl_parameter_type_name( Size index) const = 0;

    virtual const char* get_prototype() const = 0;

    virtual void get_mdl_version( Mdl_version& since, Mdl_version& removed) const = 0;

    virtual IFunction_definition::Semantics get_semantic() const = 0;

    virtual bool is_exported() const = 0;

    virtual const IType* get_return_type() const = 0;

    virtual Size get_parameter_count() const = 0;

    virtual const char* get_parameter_name( Size index) const = 0;

    virtual Size get_parameter_index( const char* name) const = 0;

    virtual const IType_list* get_parameter_types() const = 0;

    virtual const IExpression_list* get_defaults() const = 0;

    virtual const IExpression_list* get_enable_if_conditions() const = 0;

    virtual Size get_enable_if_users( Size index) const = 0;

    virtual Size get_enable_if_user( Size index, Size u_index) const = 0;

    virtual const IAnnotation_block* get_annotations() const = 0;

    virtual const IAnnotation_block* get_return_annotations() const = 0;

    virtual const IAnnotation_list* get_parameter_annotations() const = 0;

    virtual const char* get_thumbnail() const = 0;

    virtual bool is_valid( IMdl_execution_context* context) const = 0;

    virtual const IExpression_direct_call* get_body() const = 0;

    virtual Size get_temporary_count() const = 0;

    virtual const IExpression* get_temporary( Size index) const = 0;

    virtual const char* get_temporary_name( Size index) const = 0;

    template<class T>
    const T* get_temporary( Size index) const
    {
        const IExpression* ptr_iexpression = get_temporary( index);
        if ( !ptr_iexpression)
            return 0;
        const T* ptr_T = static_cast<const T*>( ptr_iexpression->get_interface( typename T::IID()));
        ptr_iexpression->release();
        return ptr_T;
    }

    virtual IMaterial_instance* create_material_instance(
        const IExpression_list* arguments, Sint32* errors = 0) const = 0;
#endif // MI_NEURAYLIB_DEPRECATED_13_0
};

/**@}*/ // end group mi_neuray_mdl_elements

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_IMATERIAL_DEFINITION_H
