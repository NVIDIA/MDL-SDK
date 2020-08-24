/***************************************************************************************************
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief   Utility class for MDL annotations.

#ifndef MI_NEURAYLIB_ANNOTATION_WRAPPER_H
#define MI_NEURAYLIB_ANNOTATION_WRAPPER_H

#include <mi/base/handle.h>
#include <mi/neuraylib/assert.h>
#include <mi/neuraylib/iexpression.h>
#include <mi/neuraylib/ifunction_call.h>
#include <mi/neuraylib/imaterial_instance.h>
#include <mi/neuraylib/imdl_factory.h>
#include <mi/neuraylib/itransaction.h>
#include <mi/neuraylib/itype.h>
#include <mi/neuraylib/ivalue.h>

namespace mi {

namespace neuraylib {

/** \addtogroup mi_neuray_mdl_elements
@{
*/

/// A wrapper around the interfaces for MDL annotations.
///
/// The purpose of the MDL helper is to simplify working with MDL annotations. 
/// It is wrapping API call sequences occurring in typical tasks into one single method call.
///
/// See #mi::neuraylib::IAnnotation_block and #mi::neuraylib::IAnnotation for the underlying
/// interfaces. See also #mi::neuraylib::Definition_wrapper for a similar wrapper for
/// MDL material and function definitions.
class Annotation_wrapper
{
public:

    /// \name General methods
    //@{

    /// Constructs a helper that provides convenience methods for annotations.
    ///
    /// \param anno_block    Block of annotations attached to a module, function, etc. 
    Annotation_wrapper(const mi::neuraylib::IAnnotation_block* anno_block);

    /// Get the number of the annotations.
    ///
    /// \return             The number of annotations
    mi::Size get_annotation_count() const;

    /// Get the name of the annotation for one of the annotations.
    ///
    /// \param anno_index   The index of the annotation
    /// \return             The name of the annotation
    const char* get_annotation_name(mi::Size anno_index) const;

    /// Get the index of an annotation based on its name for one of the annotations.
    ///
    /// \param anno_name    The name of the annotation
    /// \param offset       The first index to be considered in the search. If zero, the entire
    ///                     annotation block is searched. 
    /// \return
    ///                  -  index:  of the annotation in case of success.
    ///                  -  -1:     If there is none with that name.
    mi::Size get_annotation_index(
        const char* anno_name,
        mi::Size offset = 0) const;

    /// Get the number of parameters of an annotation.
    ///
    /// \param anno_index   The index of the annotation
    /// \return             The number of parameters of an annotation.
    mi::Size get_annotation_param_count(mi::Size anno_index) const;

    /// Get the name of one of the parameters of one of the annotations.
    ///
    /// \param anno_index   The index of the annotation
    /// \param param_index  The index of the parameter value to query
    /// \return             The name, or NULL if it does not exist
    const char* get_annotation_param_name(mi::Size anno_index, mi::Size param_index) const;

    /// Get the type of one of the parameters of one of the annotations.
    ///
    /// \param anno_index   The index of the annotation
    /// \param param_index  The index of the parameter value to query
    /// \return             The type, or NULL if it does not exist
    const IType* get_annotation_param_type(mi::Size anno_index, mi::Size param_index) const;

    /// Get the value of one of the parameters of one of the annotations. 
    ///
    /// \param anno_index   The index of the annotation
    /// \param param_index  The index of the parameter value to query
    /// \return             The value, or NULL if it does not exist
    const IValue* get_annotation_param_value(
        mi::Size anno_index, mi::Size param_index) const;

    /// Get the value of one of the parameters of one of the annotations. 
    ///
    /// \param anno_name    The name of the annotation
    /// \param param_index  The index of the parameter value to query
    /// \return             The value, or NULL if it does not exist
    const IValue* get_annotation_param_value_by_name(
        const char* anno_name, mi::Size param_index ) const;

    /// Get the value of one of the parameters of one of the annotations. 
    ///
    /// \param anno_index   The index of the annotation
    /// \param param_index  The index of the parameter value to query
    /// \param value        The value returned
    /// \return
    ///                  -  0: Success.
    ///                  - -1: The type of the parameter does not match the template type of \p T.
    ///                  - -3: The index parameter is not valid for the provided annotation block.
    template <class T> 
    mi::Sint32 get_annotation_param_value(
        mi::Size anno_index, mi::Size param_index, T& value) const;

    /// Get the value of one of the parameters of one of the annotations. 
    ///
    /// \param anno_name    The name of the annotation
    /// \param param_index  The index of the parameter value to query
    /// \param value        The value returned
    /// \return
    ///                  -  0: Success.
    ///                  - -1: The type of the parameter does not match the template type of \p T.
    ///                  - -3: The name parameter is not valid for the provided annotation block.
    template <class T>
    mi::Sint32 get_annotation_param_value_by_name(
        const char* anno_name, mi::Size param_index, T& value ) const;

private:
    mi::base::Handle<const mi::neuraylib::IAnnotation_block> m_anno_block;
};

/*@}*/ // end group mi_neuray_mdl_elements

inline Annotation_wrapper::Annotation_wrapper(
    const mi::neuraylib::IAnnotation_block* anno_block)
{
    // anno_block == null is valid and will result in an annotation count of zero
    m_anno_block = make_handle_dup(anno_block);
}

inline mi::Size Annotation_wrapper::get_annotation_count() const
{
    if (!m_anno_block)
        return 0;

    return m_anno_block->get_size();
}

inline const char* Annotation_wrapper::get_annotation_name(mi::Size anno_index) const
{
    if (!m_anno_block || m_anno_block->get_size() <= anno_index)
        return NULL;

    mi::base::Handle<const mi::neuraylib::IAnnotation> anno(
        m_anno_block->get_annotation(anno_index));
    if (!anno)
        return NULL;

    return anno->get_name();
}

inline mi::Size Annotation_wrapper::get_annotation_index(
    const char* anno_name,
    mi::Size offset) const
{
    if (!anno_name || !m_anno_block)
        return static_cast<mi::Size>(-1);

    for (mi::Size i = offset, n = get_annotation_count(); i < n; ++i)
        if (strcmp(anno_name, get_annotation_name(i)) == 0) //-V575 PVS
            return i;

    return static_cast<mi::Size>(-1);
}

inline mi::Size Annotation_wrapper::get_annotation_param_count(mi::Size anno_index) const
{
    if (!m_anno_block || m_anno_block->get_size() <= anno_index)
        return 0;

    mi::base::Handle<const mi::neuraylib::IAnnotation> anno(
        m_anno_block->get_annotation(anno_index));
    if (!anno)
        return 0;

    mi::base::Handle<const mi::neuraylib::IExpression_list> expr_list(anno->get_arguments());
    if (!expr_list)
        return 0;

    return expr_list->get_size();
}

inline const char* Annotation_wrapper::get_annotation_param_name(
    mi::Size anno_index, mi::Size param_index) const
{
    if (!m_anno_block || m_anno_block->get_size() <= anno_index)
        return NULL;

    mi::base::Handle<const mi::neuraylib::IAnnotation> anno(
        m_anno_block->get_annotation(anno_index));
    if (!anno)
        return NULL;

    mi::base::Handle<const mi::neuraylib::IExpression_list> expr_list(anno->get_arguments());
    if (!expr_list)
        return NULL;

    if (expr_list->get_size() <= param_index)
        return NULL;

    return expr_list->get_name(param_index);
}

inline const IType* Annotation_wrapper::get_annotation_param_type(
    mi::Size anno_index, mi::Size param_index) const
{
    if (!m_anno_block || m_anno_block->get_size() <= anno_index)
        return NULL;

    mi::base::Handle<const mi::neuraylib::IAnnotation> anno(
        m_anno_block->get_annotation(anno_index));
    if (!anno)
        return NULL;

    mi::base::Handle<const mi::neuraylib::IExpression_list> expr_list(anno->get_arguments());
    if (!expr_list)
        return NULL;

    if (expr_list->get_size() <= param_index)
        return NULL;

    mi::base::Handle<const mi::neuraylib::IExpression> expr(expr_list->get_expression(param_index));
    if (!expr)
        return NULL;

    return expr->get_type();
}

inline const mi::neuraylib::IValue* Annotation_wrapper::get_annotation_param_value(
    mi::Size anno_index, mi::Size param_index) const
{
    if (!m_anno_block || m_anno_block->get_size() <= anno_index)
        return NULL;

    mi::base::Handle<const mi::neuraylib::IAnnotation> anno(
        m_anno_block->get_annotation(anno_index));
    if (!anno)
        return NULL;

    mi::base::Handle<const mi::neuraylib::IExpression_list> expr_list(anno->get_arguments());
    if (!expr_list)
        return NULL;

    if (expr_list->get_size() <= param_index)
        return NULL;

    mi::base::Handle<const mi::neuraylib::IExpression> expr(expr_list->get_expression(param_index));
    if (!expr)
        return NULL;

    mi::base::Handle<const mi::neuraylib::IExpression_constant> c(
        expr->get_interface<const mi::neuraylib::IExpression_constant>());
    if (!c)
        return NULL;

    return c->get_value();
}

inline const mi::neuraylib::IValue* Annotation_wrapper::get_annotation_param_value_by_name(
    const char* anno_name, mi::Size param_index) const
{
    mi::Size anno_index = get_annotation_index(anno_name);
    if (anno_index == static_cast<mi::Size>(-1))
        return NULL;

    return get_annotation_param_value(anno_index, param_index);
}

template <class T>
inline mi::Sint32 Annotation_wrapper::get_annotation_param_value(
    mi::Size anno_index, mi::Size param_index, T& value) const
{
    mi::base::Handle<const mi::neuraylib::IValue> v(get_annotation_param_value(
        anno_index, param_index));
    if (!v)
        return -3;

    return mi::neuraylib::get_value(v.get(), value);
}

template <class T>
inline mi::Sint32 Annotation_wrapper::get_annotation_param_value_by_name(
    const char* anno_name, mi::Size param_index, T& value) const
{
    mi::base::Handle<const mi::neuraylib::IValue> v(get_annotation_param_value_by_name(
        anno_name, param_index));
    if (!v)
        return -3;

    return mi::neuraylib::get_value(v.get(), value);
}

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_ANNOTATION_WRAPPER_H
