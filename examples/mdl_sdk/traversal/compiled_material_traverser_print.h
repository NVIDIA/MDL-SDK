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

// examples/mdl_sdk/traversal/compiled_material_traverser_print.h
//
// Example class that implements the base traverser.
// It allows to generate MDL code from compiled materials that can be compiled 
// if all used functions are available (exported).


#ifndef COMPILED_MATERIAL_TRAVERSER_PRINT_H
#define COMPILED_MATERIAL_TRAVERSER_PRINT_H

#include "compiled_material_traverser_base.h"

#include <stack>
#include <set>
#include <map>
#include <sstream>

// An implementation of the Compiled_material_traverser_base to demonstrate a 
// valid traversal mechanism. In this case to print MDL code from a compiled material. 
class Compiled_material_traverser_print : public Compiled_material_traverser_base
{
public:

    // A custom context that is passed through while iterating the material.
    class Context
    {
        friend class Compiled_material_traverser_print;

    public:

        // Creates the context of a traversal, which includes required component and 
        // configurations.
        //
        // Param:  transaction     The DB transaction to resolve resources.
        // Param:  keep_structure  The structures produced by the compiler do not always 
        //                         match the structure of the input material, e.g., constants are 
        //                         transformed to parameters this printer focuses on producing mdl 
        //                         code. However, it could be interesting to see an output that is 
        //                         closer to the compiler output. Therefore, set true.
        Context(
            mi::neuraylib::ITransaction* transaction,
            bool keep_structure);

        // modules that have been imported directly by the module and used by the input material
        const std::set<std::string>& get_used_modules() const { return m_used_modules; }

        // resources that have been imported directly by the module and used by the input material
        const std::set<std::string>& get_used_resources() const { return m_used_resources; }

        // Indicated whether the output mdl should be valid or not.
        //
        // if we do not inline generated parameters, we want to inform about invalid mdl
        // after running the printer, this will be true if encountered no invalid case.
        // Note, this can only be false if 'keep_structure' was set to true.
        bool get_is_valid_mdl() const { return m_is_valid_mdl; }

    private:

        // reset private fields of the context to allow reuse
        void reset();

        // required to resolve resources
        mi::neuraylib::ITransaction* m_transaction;

        // stream to build up the mdl code
        std::stringstream m_print;

        // for formatting
        size_t m_indent;

        // track imported functions, types, ...
        std::set<std::string> m_imports;

        // additional information about the module 
        std::set<std::string> m_used_modules;
        std::set<std::string> m_used_resources;

        // favor compiler created structure (may create invalid mdl)
        bool m_keep_compiled_material_structure;
        std::map<std::string, std::string> m_parameters_to_inline;
        std::stringstream m_print_inline_swap;
        size_t m_indent_inline_swap;

        // relevant only in case we do not inline generated parameters
        bool m_is_valid_mdl;
        Compiled_material_traverser_base::Traveral_stage m_stage; // required for validity checking
    };


    // Generates MDL code from a compiled material.
    // 
    // Since the ICompiled_material lacks some information to generate a valid
    // module, the original module name needs to be provided for referencing
    // exported functions in the original module this material was defined in.
    // The new material also requires a name that has to be provided, too. 
    //
    // Param:          material                The material to print.
    // Param: [in,out] context                 The context that is passed through.
    // Param:          original_module_name    Name of the original module.
    // Param:          output_material_name    Name of the output material.
    //
    // Return: the generated MDL code that can be saved as module.
    std::string print_mdl(const mi::neuraylib::ICompiled_material* material,
                                Context& context,
                                const std::string& original_module_name,
                                const std::string& output_material_name);

protected:

    // Called at the beginning of each traversal stage: Parameters, Temporaries and Body.
    //
    // Param:          material    The material that is traversed.
    // Param:          stage       The stage that was entered.
    // Param: [in,out] context     The context that is passed through without changes.
    void stage_begin(const mi::neuraylib::ICompiled_material* material,
                     Compiled_material_traverser_base::Traveral_stage stage,
                     void* context) override;

    // Called at the end of each traversal stage: Parameters, Temporaries and Body.
    //
    // Param:          material    The material that is traversed.
    // Param:          stage       The stage that was finished.
    // Param: [in,out] context     The context that is passed through without changes.
    void stage_end(const mi::neuraylib::ICompiled_material* material,
                   Compiled_material_traverser_base::Traveral_stage stage,
                   void* context) override;


    // Called when the traversal reaches a new element.
    //
    // Param:          material    The material that is traversed.
    // Param:          element     The element that was reached.
    // Param: [in,out] context     The context that is passed through without changes.
    void visit_begin(const mi::neuraylib::ICompiled_material* material,
                     const Compiled_material_traverser_base::Traversal_element& element,
                     void* context) override;

    // In that case, the method is called before each of the children are traversed, e.g.,
    // before each argument of a function call.
    //
    // Param:          material        The material that is traversed.
    // Param:          element         The currently traversed element with multiple children.
    // Param:          children_count  Number of children of the current element.
    // Param:          child_index     The index of the child that will be traversed next.
    // Param: [in,out] context         The context that is passed through without changes.
    void visit_child(const mi::neuraylib::ICompiled_material* material,
                     const Compiled_material_traverser_base::Traversal_element& element,
                     mi::Size children_count, mi::Size child_index,
                     void* context) override;


    // Called when the traversal reaches finishes an element.
    //
    // Param:          material    The material that is traversed.
    // Param:          element     The element that is finished.
    // Param: [in,out] context     The context that is passed through without changes.
    void visit_end(const mi::neuraylib::ICompiled_material* material,
                   const Compiled_material_traverser_base::Traversal_element& element,
                   void* context) override;

private:

    // Helper function to generate the indentation.
    //
    // Param:  context     The context that is passed through without changes.
    // Param:  offset      (Optional) Allows to change the indent by a specified number of tabs.
    //
    // Return:             The indentation as string.
    const std::string indent(const Context* context, mi::Sint32 offset = 0) const;

    // Returns the type of an enum as string.
    static std::string enum_type_to_string(const mi::neuraylib::IType_enum* enum_type,
                                           Context* context);
    
    // Returns the type of a struct as string.
    static std::string struct_type_to_string(const mi::neuraylib::IType_struct* struct_type,
                                             Context* context,
                                             bool* out_is_material_keyword = nullptr);
    
    // Returns the type of an elemental type as string.
    static std::string atomic_type_to_string(const mi::neuraylib::IType_atomic* atomic_type,
                                             Context* context);
    
    // Returns a vector type as string.
    static std::string vector_type_to_string(const mi::neuraylib::IType_vector* vector_type,
                                             Context* context);
    
    // Returns a matrix type as string.
    static std::string matrix_type_to_string(const mi::neuraylib::IType_matrix* matrix_type,
                                             Context* context);
    
    // Returns an array type as string.
    static std::string array_type_to_string(const mi::neuraylib::IType_array* array_type,
                                            Context* context);

    // Returns the name of type as string.
    static const std::string type_to_string(const mi::neuraylib::IType* type,
                                            Context* context);
};

#endif // COMPILED_MATERIAL_TRAVERSER_PRINT_H
