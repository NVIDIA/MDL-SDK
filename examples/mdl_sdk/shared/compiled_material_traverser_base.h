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

// examples/mdl_sdk/shared/compiled_material_traverser_base.h
//
// Utility class for traversing compiled materials.
// Derived classes override virtual functions to implement their logic.   

#ifndef COMPILED_MATERIAL_TRAVERSER_BASE_H
#define COMPILED_MATERIAL_TRAVERSER_BASE_H

#include "example_shared.h"

#include <string>
#include <vector>


// A base class that implements a simple traversal logic for compiled materials.
class Compiled_material_traverser_base
{
protected:

    // Possible stages of the traversal of an compiled material
    enum Traveral_stage
    {
        // Traversal has not been started
        ES_NOT_STARTED = 0,
        // Indicates that the parameters of the material are currently traversed
        ES_PARAMETERS,
        // Indicates that the temporaries of the material are currently traversed
        ES_TEMPORARIES,
        // Indicates that the main body of the material are currently traversed
        ES_BODY,
        // Traversal is done
        ES_FINISHED,

        // For alignment only.
        ES_FORCE_32_BIT = 0xffffffffU
    };


    // An internal structure that is passed to the user code while traversing.
    // This struct is used while visiting the material parameters.
    struct Parameter
    {
        explicit Parameter(const mi::neuraylib::IValue* value) 
            : value(value)
        { }

        const mi::neuraylib::IValue* value;
    };

    // An internal structure that is passed to the user code while traversing.
    // This struct is used while visiting the materials temporaries.
    struct Temporary
    {
        explicit Temporary(const mi::neuraylib::IExpression* expression) 
            : expression(expression)
        { }

        const mi::neuraylib::IExpression* expression;
    };


    // Encapsulated to current element that is visited during the traversal
    // It contains either an IExpression, an IValue, a Parameter or a Temporary while the
    // others are nullptr.
    struct Traversal_element
    {
        explicit Traversal_element(const mi::neuraylib::IExpression* expression,
                                   mi::Size sibling_count = 1, mi::Size sibling_index = 0) 
            : expression(expression)
            , value(nullptr)
            , parameter(nullptr)
            , temporary(nullptr)
            , sibling_count(sibling_count)
            , sibling_index(sibling_index)
        { }

        explicit Traversal_element(const mi::neuraylib::IValue* value,
                                   mi::Size sibling_count = 1, mi::Size sibling_index = 0)
            : expression(nullptr)
            , value(value)
            , parameter(nullptr)
            , temporary(nullptr)
            , sibling_count(sibling_count)
            , sibling_index(sibling_index)
        { }

        explicit Traversal_element(const Parameter* parameter,
                                   mi::Size sibling_count = 1, mi::Size sibling_index = 0)
            : expression(nullptr)
            , value(nullptr)
            , parameter(parameter)
            , temporary(nullptr)
            , sibling_count(sibling_count)
            , sibling_index(sibling_index)
        { }

        explicit Traversal_element(const Temporary* temporary,
                                   mi::Size sibling_count = 1, mi::Size sibling_index = 0)
            : expression(nullptr)
            , value(nullptr)
            , parameter(nullptr)
            , temporary(temporary)
            , sibling_count(sibling_count)
            , sibling_index(sibling_index)
        { }

        // Not nullptr if the current traversal element is an IExpression.
        const mi::neuraylib::IExpression* expression;

        // Not nullptr if the current traversal element is an IValue.
        const mi::neuraylib::IValue* value;

        // Not nullptr if the current traversal element is a Parameter.
        // This can happen only in the ES_PARAMETERS stage.
        const Parameter* parameter;

        // Not nullptr if the current traversal element is a Parameter.
        // This can happen only in the ES_TEMPORARAY stage.
        const Temporary* temporary;

        // Total number of children at the parent of the currently traversed element.
        mi::Size sibling_count;

        // Index of the currently traversed element in the list of children at the parent.
        mi::Size sibling_index;
    };


public:

    // virtual destructor
    virtual ~Compiled_material_traverser_base() {}; /* = default;*/


protected:

    // Traverses a compiled material and calls the corresponding virtual visit methods.
    //
    // This method is meant to be called by deriving class to start the actual traversal.
    //
    // Param:          material    The material that is traversed.
    // Param: [in,out] context     User defined context that is passed through without changes.
    void traverse(const mi::neuraylib::ICompiled_material* material, void* context);


    // Called at the beginning of each traversal stage: Parameters, Temporaries and Body.
    //
    // Param:          material    The material that is traversed.
    // Param:          stage       The stage that was entered.
    // Param: [in,out] context     User defined context that is passed through without changes.
    virtual void stage_begin(const mi::neuraylib::ICompiled_material* material,
                             Traveral_stage stage, void* context) {};


    // Called at the end of each traversal stage: Parameters, Temporaries and Body.
    //
    // Param:          material    The material that is traversed.
    // Param:          stage       The stage that was finished.
    // Param: [in,out] context     User defined context that is passed through without changes.
    virtual void stage_end(const mi::neuraylib::ICompiled_material* material,
                           Traveral_stage stage, void* context) {};


    // Called when the traversal reaches a new element.
    //
    // Param:          material    The material that is traversed.
    // Param:          element     The element that was reached.
    // Param: [in,out] context     User defined context that is passed through without changes.
    virtual void visit_begin(const mi::neuraylib::ICompiled_material* material,
                             const Traversal_element& element, void* context) {};



    // Occurs only if the current element has multiple child elements, e.g., a function call.
    // In that case, the method is called before each of the children are traversed, e.g.,
    // before each argument of a function call.
    //
    // Param:          material        The material that is traversed.
    // Param:          element         The currently traversed element with multiple children.
    // Param:          children_count  Number of children of the current element.
    // Param:          child_index     The index of the child that will be traversed next.
    // Param: [in,out] context         User defined context that is passed through without changes.
    virtual void visit_child(const mi::neuraylib::ICompiled_material* material,
                             const Traversal_element& element,
                             mi::Size children_count, mi::Size child_index,
                             void* context) {};



    // Called when the traversal reaches finishes an element.
    //
    // Param:          material    The material that is traversed.
    // Param:          element     The element that is finished.
    // Param: [in,out] context     User defined context that is passed through without changes.
    virtual void visit_end(const mi::neuraylib::ICompiled_material* material,
                           const Traversal_element& element, void* context) {};



    // Gets the name of a parameter of the traversed material.
    //
    // Param:  material        The material that is traversed.
    // Param:  index           Index of the parameter in the materials parameter list.
    // Param:  out_generated   Optional output parameter that indicates whether the parameter 
    //                         was generated by the compiler rather than defined in the 
    //                         material definition.
    //
    // Return: The parameter name.
    std::string get_parameter_name(const mi::neuraylib::ICompiled_material* material,
                                         mi::Size index, bool* out_generated = nullptr) const;

    // Gets the name of a temporary of the traversed material.
    // Since the name is usually unknown, due to optimization, a proper name is generated.
    //
    // Param:  material    The material that is traversed.
    // Param:  index       Index of the parameter in the materials temporary list.
    //
    // Return: The temporary name.
    std::string get_temporary_name(const mi::neuraylib::ICompiled_material* material,
                                         mi::Size index) const;

private:

    // Recursive function that is used for the actual traversal.
    // The names of templates are lost during compilation. Therefore, we generate numbered ones.
    //
    // Param:          material    The material.
    // Param:          element     The element that is currently visited.
    // Param: [in,out] context     User defined context that is passed through without changes.
    void traverse(const mi::neuraylib::ICompiled_material* material,
                  const Traversal_element& element, void* context);
};

#endif // COMPILED_MATERIAL_TRAVERSER_BASE_H
