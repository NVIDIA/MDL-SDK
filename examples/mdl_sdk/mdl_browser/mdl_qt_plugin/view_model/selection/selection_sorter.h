/******************************************************************************
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
 *****************************************************************************/

/// \file
/// \brief Classes for sorting selectable elements, e.g., materials, functions, ...


#ifndef MDL_SDK_EXAMPLES_MDL_BROWSER_SELECTION_SORTER_H
#define MDL_SDK_EXAMPLES_MDL_BROWSER_SELECTION_SORTER_H

#include "vm_sel_element.h"


class Index_cache_elements;
class VM_nav_package;

// base class to enable the sorting of the presented selectable elements
class Selection_sorter_base
{
public:
    explicit Selection_sorter_base(const std::string& name, bool ascending = false);
    virtual ~Selection_sorter_base() = default;

    // implementation of the actual comparison
    virtual float compare(const VM_sel_element* left, const VM_sel_element* right) const = 0;

    // name of the criteria, used for identification (UI and Settings)
    const std::string get_name() { return m_name; }

    // oder of sorting
    bool get_ascending() const { return m_ascending; }
    void set_ascending(bool value) { m_ascending = value; }

private:
    const std::string m_name;
    bool m_ascending;

};


// sort by search result ranking
class Selection_sorter_ranking 
    : public Selection_sorter_base
{
public:
    explicit Selection_sorter_ranking(const std::string& name, bool ascending = false)
        : Selection_sorter_base(name, ascending) { }
    virtual ~Selection_sorter_ranking() = default;

    float compare(const VM_sel_element* left, const VM_sel_element* right) const override;
};

// sort by name of the selectable element
class Selection_sorter_name 
    : public Selection_sorter_base
{
public:
    explicit Selection_sorter_name(const std::string& name, bool ascending = false)
        : Selection_sorter_base(name, ascending) { }
    virtual ~Selection_sorter_name() = default;

    float compare(const VM_sel_element* left, const VM_sel_element* right) const override;
};

// sort by date of last change
class Selection_sorter_date 
    : public Selection_sorter_base
{
public:
    explicit Selection_sorter_date(const std::string& name, bool ascending = false)
        : Selection_sorter_base(name, ascending) { }
    virtual ~Selection_sorter_date() = default;

    float compare(const VM_sel_element* left, const VM_sel_element* right) const override;
};


#endif 
