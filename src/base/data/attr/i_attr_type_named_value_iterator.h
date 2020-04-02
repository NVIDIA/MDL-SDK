/***************************************************************************************************
 * Copyright (c) 2010-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief The definition of the Type_value_iterator.

#ifndef BASE_DATA_ATTR_TYPE_NAMED_VALUE_ITERATOR_H
#define BASE_DATA_ATTR_TYPE_NAMED_VALUE_ITERATOR_H

#include "i_attr_type_value_iterator.h"
#include <string>
#include <sstream>

namespace MI {
namespace ATTR {
    

/// Extension of Type_value_iterator, provides qualified name of the
/// current value.
class Type_named_value_iterator : public Type_value_iterator
{
public:
    /// Constructor.
    /// \param type the "root" type
    /// \param values the corresponding values pointer
    Type_named_value_iterator(
        const Type* type,
        const char* values);

    /// Default constructor, creates invalid iterator, usable as end-iterator.
    Type_named_value_iterator();

    /// Copy operations.
    Type_named_value_iterator(Type_named_value_iterator const&);
    Type_named_value_iterator& operator=(Type_named_value_iterator const&);

    /// Get qualified name of current value, e.g. "x[2].y"
    std::string get_qualified_name() const;

protected:
    virtual void pre_increment();
    virtual void post_increment();

private:
    // Use vector-based stack, because vector is smaller than deque,
    // is simpler to debug, and sufficient for small stack depths.
    std::stack<int, std::vector<int> > m_array_sizes;
    std::stack<std::string, std::vector<std::string> > m_name_stack;
    mutable std::stringstream m_strbuf;
};

}
}

#endif
