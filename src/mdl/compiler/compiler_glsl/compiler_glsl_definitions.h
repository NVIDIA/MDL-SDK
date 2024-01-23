/******************************************************************************
 * Copyright (c) 2015-2024, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_COMPILER_GLSL_DEFINITIONS_H
#define MDL_COMPILER_GLSL_DEFINITIONS_H 1

#include "compiler_glsl_cc_conf.h"

#include <mi/mdl/mdl_iowned.h>

#include "compiler_glsl_declarations.h"
#include "compiler_glsl_exprs.h"
#include "compiler_glsl_types.h"

namespace mi {
namespace mdl {
namespace glsl {

class Compilation_unit;
class Definition_table;
class Location;
class Symbol;
class Value;
class Scope;

/// A GLSL definition table entry.
class Definition : public Interface_owned
{
    typedef Interface_owned Base;
    friend class Scope;
public:
    /// Definition kinds.
    enum Kind {
        DK_ERROR,         ///< This is an error definition.
        DK_TYPE,          ///< This is a type.
        DK_FUNCTION,      ///< This is a function.
        DK_VARIABLE,      ///< This is a variable.
        DK_MEMBER,        ///< This is a field member.
        DK_PARAMETER,     ///< This is a parameter.
        DK_OPERATOR,      ///< This is an operator.
    };

    /// Returns the kind of this definition.
    virtual Kind get_kind() const = 0;

    /// Get the symbol of the definition.
    Symbol *get_symbol() const { return m_sym; }

    /// Get the type of this definition.
    virtual Type *get_type() const { return m_type; }

    /// Get the declaration of the definition.
    virtual Declaration *get_declaration() const = 0;

    /// Return the location of this definition if any.
    Location const *get_location() const { return m_loc; }

    // Non interface member

    /// Return the definition scope of this definition.
    Scope *get_def_scope() const { return m_def_scope; }

    /// Link a previous definition for the same symbol.
    ///
    /// Note that the symbol table is updated, and shows a list of same
    /// definitions that occurred before the current one.
    ///
    void link_same_def(Definition *prev_def) {
        m_same_prev           = prev_def;
        prev_def->m_same_next = this;
    }

    /// Returns the next Definition in the parent scope.
    Definition *get_next_def_in_scope() const { return m_next; }

    /// Return the previous definition for the same symbol.
    Definition *get_prev_def() const { return m_same_prev; }

    /// Return the next definition for the same symbol.
    Definition *get_next_def() const { return m_same_next; }

    /// Return the definition for this symbol in the outer scope.
    Definition *get_outer_def() const { return m_outer_def; }

    /// Set the definition for this symbol in the outer scope.
    ///
    /// \param outer  the outer definition
    void set_outer_def(Definition *outer) { m_outer_def = outer; }

    /// Ge the unique ID.
    size_t get_unique_id() const { return m_id; }

protected:
    /// Constructor.
    ///
    /// \param sym           the symbol on this definition
    /// \param type          the type of this definition
    /// \param loc           the location of the definition of NULL
    /// \param parent_scope  the parent scope of the definition
    /// \param outer         the definition for the same symbol in the outer scope if any
    /// \param id            the unique id of this definition
    explicit Definition(
        Symbol         *sym,
        Type           *type,
        Location const *loc,
        Scope          *parent_scope,
        Definition     *outer,
        size_t         id);

private:
    // no copy and no assignment
    Definition(Definition const &) MDL_DELETED_FUNCTION;
    Definition &operator=(Definition const &) MDL_DELETED_FUNCTION;

protected:
    /// The symbol of this definition.
    Symbol *m_sym;

    /// The type of this definition.
    Type *m_type;

    /// The declaration of this definition if any.
    Declaration *m_decl;

    /// The location of this definition if any.
    Location const *m_loc;

    /// The definition scope of this function.
    Scope *m_def_scope;

    /// Points to the next definition in this scope.
    Definition *m_next;

    /// Points to the previous definition of the same symbol in this scope.
    Definition *m_same_prev;

    /// Points to the next definition of the same symbol in this scope.
    Definition *m_same_next;

    /// Points to the definition of the same symbol in outer scope.
    Definition *m_outer_def;

    /// The unique id of this definition.
    size_t const m_id;
};

/// An error definition.
class Def_error GLSL_FINAL : public Definition
{
    typedef Definition Base;
    friend class mi::mdl::Arena_builder;
public:
    static Kind const s_kind = DK_ERROR;

public:
    /// Returns the kind of this definition.
    Kind get_kind() const GLSL_FINAL;

    /// Get the type of this definition.
    Type_error *get_type() const GLSL_FINAL;

    /// Get the declaration of the definition.
    Declaration *get_declaration() const GLSL_FINAL;

private:
    /// Constructor.
    ///
    /// \param sym           the symbol on this definition.
    /// \param type          the error type
    /// \param loc           the location of the definition of NULL
    /// \param parent_scope  the parent scope of the definition
    /// \param outer         the definition for the same symbol in the outer scope if any
    /// \param id            the unique id of this definition
    explicit Def_error(
        Symbol         *sym,
        Type_error     *type,
        Location const *loc,
        Scope          *parent_scope,
        Definition     *outer,
        size_t         id);
};

/// A variable definition.
class Def_variable GLSL_FINAL : public Definition
{
    typedef Definition Base;
    friend class mi::mdl::Arena_builder;
public:
    static Kind const s_kind = DK_VARIABLE;

public:
    /// Returns the kind of this definition.
    Kind get_kind() const GLSL_FINAL;

    /// Get the type of this definition.
    Type *get_type() const GLSL_FINAL;

    /// Get the declaration of the definition.
    Declaration_variable *get_declaration() const GLSL_FINAL;

    /// Set the declaration.
    void set_declaration(Declaration_variable *decl) { m_decl = decl; }

private:
    /// Constructor.
    ///
    /// \param sym           the symbol on this definition.
    /// \param type          the type of this definition
    /// \param loc           the location of the definition of NULL
    /// \param parent_scope  the parent scope of the definition
    /// \param outer         the definition for the same symbol in the outer scope if any
    /// \param id            the unique id of this definition
    explicit Def_variable(
        Symbol         *sym,
        Type           *type,
        Location const *loc,
        Scope          *parent_scope,
        Definition     *outer,
        size_t         id);
};

/// A parameter definition.
class Def_param GLSL_FINAL : public Definition
{
    typedef Definition Base;
    friend class mi::mdl::Arena_builder;
public:
    static Kind const s_kind = DK_PARAMETER;

public:
    /// Returns the kind of this definition.
    Kind get_kind() const GLSL_FINAL;

    /// Get the type of this definition.
    Type *get_type() const GLSL_FINAL;

    /// Get the declaration of the definition.
    Declaration_param *get_declaration() const GLSL_FINAL;

    /// Set the declaration.
    void set_declaration(Declaration_param *decl) { m_decl = decl; }

    /// Return the parameter index of a parameter.
    size_t get_parameter_index() const;

private:
    /// Constructor.
    ///
    /// \param sym           the symbol on this definition.
    /// \param type          the type of this definition
    /// \param loc           the location of the definition of NULL
    /// \param parent_scope  the parent scope of the definition
    /// \param outer         the definition for the same symbol in the outer scope if any
    /// \param id            the unique id of this definition
    explicit Def_param(
        Symbol         *sym,
        Type           *type,
        Location const *loc,
        Scope          *parent_scope,
        Definition     *outer,
        size_t         id);
};

/// A function definition.
class Def_function GLSL_FINAL : public Definition
{
    typedef Definition Base;
    friend class mi::mdl::Arena_builder;
public:
    static Kind const s_kind = DK_FUNCTION;

    /// Built-in semantics.
    enum Semantics {
        DS_UNKNOWN = 0,
        DS_COPY_CONSTRUCTOR,               ///< copy constructor
        DS_ELEM_CONSTRUCTOR,               ///< elemental constructor
        DS_CONV_CONSTRUCTOR,               ///< conversion constructor
        DS_VECTOR_SPLAT_CONSTRUCTOR,       ///< vector splat constructor
        DS_MATRIX_ELEM_CONSTRUCTOR,        ///< matrix elemental constructor
        DS_MATRIX_DIAG_CONSTRUCTOR,        ///< matrix diagonal constructor
        DS_MATRIX_CONV_CONSTRUCTOR,        ///< matrix to matrix conversion constructor

        // Angle and Trigonometric Functions
        DS_RT_radians,                    ///< radians()
        DS_RT_degrees,                    ///< degrees()
        DS_RT_sin,                        ///< sin()
        DS_RT_cos,                        ///< cos()
        DS_RT_tan,                        ///< tan()
        DS_RT_asin,                       ///< asin()
        DS_RT_acos,                       ///< acos()
        DS_RT_atan,                       ///< atan()
        DS_RT_sinh,                       ///< sinh()
        DS_RT_cosh,                       ///< cosh()
        DS_RT_tanh,                       ///< tanh()
        DS_RT_asinh,                      ///< asinh()
        DS_RT_acosh,                      ///< acosh()
        DS_RT_atanh,                      ///< atanh()

        // Exponential Functions
        DS_RT_pow,                         ///< pow()
        DS_RT_exp,                         ///< exp()
        DS_RT_log,                         ///< log()
        DS_RT_exp2,                        ///< exp2()
        DS_RT_log2,                        ///< log2()
        DS_RT_sqrt,                        ///< sqrt()
        DS_RT_inversesqrt,                 ///< inversesqrt()

        // Common Functions
        DS_RT_abs,                         ///< abs()
        DS_RT_sign,                        ///< sign()
        DS_RT_floor,                       ///< floor()
        DS_RT_ceil,                        ///< ceil()
        DS_RT_fract,                       ///< fract()
        DS_RT_mod,                         ///< mod()
        DS_RT_min,                         ///< min()
        DS_RT_max,                         ///< max()
        DS_RT_clamp,                       ///< clamp()
        DS_RT_mix,                         ///< mix()
        DS_RT_step,                        ///< step()
        DS_RT_smoothstep,                  ///< snoothstep()
        DS_RT_trunc,                       ///< trunc()
        DS_RT_round,                       ///< round()
        DS_RT_roundEven,                   ///< roundEven()
        DS_RT_modf,                        ///< modf()
        DS_RT_isnan,                       ///< isnan()
        DS_RT_isinf,                       ///< isinf()
        DS_RT_atomicAdd,                   ///< atomicAdd()
        DS_RT_atomicMin,                   ///< atomicMin()
        DS_RT_atomicMax,                   ///< atomicMax()
        DS_RT_atomicAnd,                   ///< atomicAnd()
        DS_RT_atomicOr,                    ///< atomicOr()
        DS_RT_atomicXor,                   ///< atomicXor()
        DS_RT_atomicExchange,              ///< atomicExchange()
        DS_RT_atomicCompSwap,              ///< atomicCompSwap()
        DS_RT_floatBitsToInt,              ///< floatBitsToInt()
        DS_RT_floatBitsToUint,             ///< floatBitsToUint()
        DS_RT_intBitsToFloat,              ///< intBitsToFloat()
        DS_RT_uintBitsToFloat,             ///< uintBitsToFloat()
        DS_RT_fma,                         ///< fma()
        DS_RT_frexp,                       ///< frexp()
        DS_RT_ldexp,                       ///< ldexp()
        DS_RT_packDouble2x32,              ///< packDouble2x32()
        DS_RT_unpackDouble2x32,            ///< unpackDouble2x32()
        DS_RT_packUnorm2x16,               ///< packUnorm2x16(),
        DS_RT_unpackUnorm2x16,             ///< unpackUnorm2x16(),
        DS_RT_packSnorm2x16,               ///< packSnorm2x16()
        DS_RT_unpackSnorm2x16,             ///< unpackSnorm2x16()
        DS_RT_packHalf2x16,                ///< packHalf2x16()
        DS_RT_unpackHalf2x16,              ///< unpackHalf2x16()
        DS_RT_packSnorm4x8,                ///< packSnorm4x8()
        DS_RT_unpackSnorm4x8,              ///< unpackSnorm4x8()
        DS_RT_packUnorm4x8,                ///< packUnorm4x8()
        DS_RT_unpackUnorm4x8,              ///< unpackUnorm4x8()

        // Geometric Functions
        DS_RT_length,                      ///< length()
        DS_RT_distance,                    ///< distance()
        DS_RT_dot,                         ///< dot()
        DS_RT_cross,                       ///< cross()
        DS_RT_normalize,                   ///< normalize()
        DS_RT_faceforward,                 ///< faceforward()
        DS_RT_reflect,                     ///< reflect()
        DS_RT_refract,                     ///< refract()

        // Matrix Functions
        DS_RT_matrixCompMult,              ///< matrixCompMult()
        DS_RT_outerProduct,                ///< outerProduct()
        DS_RT_transpose,                   ///< transpose()
        DS_RT_determinant,                 ///< determinant()
        DS_RT_inverse,                     ///< inverse()

        // Vector relational functions
        DS_RT_lessThan,                    ///< lessThan()
        DS_RT_lessThanEqual,               ///< lessThanEqual()
        DS_RT_greaterThan,                 ///< greaterThan()
        DS_RT_greaterThanEqual,            ///< greaterThanEqual()
        DS_RT_equal,                       ///< equal()
        DS_RT_notEqual,                    ///< notEqual()
        DS_RT_any,                         ///< any()
        DS_RT_all,                         ///< all()
        DS_RT_not,                         ///< not()

        // Texture Functions
        DS_RT_texture2D,                   ///< texture2D()
        DS_RT_texture2DProj,               ///< texture2DProj()
        DS_RT_texture3D,                   ///< texture3D()
        DS_RT_texture3DProj,               ///< texture3DProj()
        DS_RT_textureCube,                 ///< textureCube()
        DS_RT_texture1D,                   ///< texture1D()
        DS_RT_texture1DProj,               ///< texture1DProj()
        DS_RT_shadow1D,                    ///< shadow1D()
        DS_RT_shadow2D,                    ///< shadow2D()
        DS_RT_shadow1DProj,                ///< shadow1DProj()
        DS_RT_shadow2DProj,                ///< shadow2DProj()
        DS_RT_texture2DRect,               ///< texture2DRect()
        DS_RT_texture2DRectProj,           ///< texture2DRectProj()
        DS_RT_shadow2DRect,                ///< shadow2DRect()
        DS_RT_shadow2DRectProj,            ///< shadow2DRectProj()
        DS_RT_texture2DGradEXT,            ///< texture2DGradEXT()
        DS_RT_texture2DProjGradEXT,        ///< texture2DProjGradEXT()
        DS_RT_textureCubeGradEXT,          ///< textureCubeGradEXT()

        // Noise Functions
        DS_RT_noise1,                      ///< noise1()
        DS_RT_noise2,                      ///< noise2()
        DS_RT_noise3,                      ///< noise3()
        DS_RT_noise4,                      ///< noise4()

        // Atomic counter Functions
        DS_RT_atomicCounterIncrement,      ///< atomicCounterIncrement()
        DS_RT_atomicCounterDecrement,      ///< atomicCounterDecrement()
        DS_RT_atomicCounter,               ///< atomicCounter()

        // Bitfield Functions
        DS_RT_uaddCarry,                   ///< uaddCarry()
        DS_RT_usubBorrow,                  ///< usubBorrow()
        DS_RT_umulExtended,                ///< umulExtended()
        DS_RT_imulExtended,                ///< imulExtended()
        DS_RT_bitfieldExtract,             ///< bitfieldExtract()
        DS_RT_bitfieldInsert,              ///< bitfieldInsert()
        DS_RT_bitfieldReverse,             ///< bitfieldReverse()
        DS_RT_bitCount,                    ///< bitCount()
        DS_RT_findLSB,                     ///< findLSB()
        DS_RT_findMSB,                     ///< findMSB()

        // Texture LOD Functions
        DS_RT_texture2DLod,                ///< texture2DLod()
        DS_RT_texture2DProjLod,            ///< texture2DProjLod()
        DS_RT_textureCubeLod,              ///< textureCubeLod()
        DS_RT_texture3DLod,                ///< texture3DLod()
        DS_RT_texture3DProjLod,            ///< texture3DProjLod()
        DS_RT_texture1DLod,                ///< texture1DLod()
        DS_RT_texture1DProjLod,            ///< texture1DProjLod()
        DS_RT_shadow1DLod,                 ///< shadow1DLod()
        DS_RT_shadow2DLod,                 ///< shadow2DLod()
        DS_RT_shadow1DProjLod,             ///< shadow1DProjLod()
        DS_RT_shadow2DProjLod,             ///< shadow2DProjLod()
        DS_RT_texture1DGradARB,            ///< texture1DGradARB()
        DS_RT_texture1DProjGradARB,        ///< texture1DProjGradARB()
        DS_RT_texture2DGradARB,            ///< texture2DGradARB()
        DS_RT_texture2DProjGradARB,        ///< texture2DProjGradARB()
        DS_RT_texture3DGradARB,            ///< texture3DGradARB()
        DS_RT_texture3DProjGradARB,        ///< texture3DProjGradARB()
        DS_RT_textureCubeGradARB,          ///< textureCubeGradARB()
        DS_RT_shadow1DGradARB,             ///< shadow1DGradARB()
        DS_RT_shadow1DProjGradARB,         ///< shadow1DProjGradARB()
        DS_RT_shadow2DGradARB,             ///< shadow2DGradARB()
        DS_RT_shadow2DProjGradARB,         ///< shadow2DProjGradARB()
        DS_RT_texture2DRectGradARB,        ///< texture2DRectGradARB()
        DS_RT_texture2DRectProjGradARB,    ///< texture2DRectProjGradARB()
        DS_RT_shadow2DRectGradARB,         ///< shadow2DRectGradARB()
        DS_RT_shadow2DRectProjGradARB,     ///< shadow2DRectProjGradARB()

        // functions seen by geometry shaders only
        DS_RT_EmitStreamVertex,            ///< EmitStreamVertex()
        DS_RT_EndStreamPrimitive,          ///< EndStreamPrimitive()
        DS_RT_EmitVertex,                  ///< EmitVertex()
        DS_RT_EndPrimitive,                ///< EndPrimitive()

        // control functions
        DS_RT_barrier,                    ///< barrier()
        DS_RT_memoryBarrier,              ///< memoryBarrier()
        DS_RT_memoryBarrierAtomicCounter, ///< memoryBarrierAtomicCounter()
        DS_RT_memoryBarrierBuffer,        ///< memoryBarrierBuffer()
        DS_RT_memoryBarrierImage,         ///< memoryBarrierImage()
        DS_RT_memoryBarrierShared,        ///< memoryBarrierShared()
        DS_RT_groupMemoryBarrier,         ///< groupMemoryBarrier()

        // functions seen by fragment shaders only
        DS_RT_texture2DLodEXT,           ///< texture2DLodEXT()
        DS_RT_texture2DProjLodEXT,       ///< texture2DProjLodEXT()
        DS_RT_textureCubeLodEXT,         ///< textureCubeLodEXT()
        DS_RT_dFdx,                      ///< dFdx()
        DS_RT_dFdy,                      ///< dFdy()
        DS_RT_fwidth,                    ///< fwidth()
        DS_RT_dFdxFine,                  ///< dFdxFine()
        DS_RT_dFdyFine,                  ///< dFdyFine()
        DS_RT_fwidthFine,                ///< fwidthFine()
        DS_RT_dFdxCoarse,                ///< dFdxCoarse()
        DS_RT_dFdyCoarse,                ///< dFdyCoarse()
        DS_RT_fwidthCoarse,              ///< fwidthCoarse()
    };

public:
    /// Returns the kind of this definition.
    Kind get_kind() const GLSL_FINAL;

    /// Get the type of this definition.
    Type_function *get_type() const GLSL_FINAL;

    /// Get the declaration of the definition.
    Declaration_function *get_declaration() const GLSL_FINAL;

    /// Get the semantics of this definition.
    Semantics get_semantics() const { return m_semantics; }

    /// Get the prototype declaration of the definition if any.
    Declaration_function const *get_prototype_declaration() const;

    /// Return scope that this definition creates.
    Scope *get_own_scope() const { return m_own_scope; }

    /// Set the scope that this definition creates.
    void set_own_scope(Scope *scope) { m_own_scope = scope; }

    /// Set the declaration.
    void set_declaration(Declaration_function *decl) { m_decl = decl; }

private:
    /// Constructor.
    ///
    /// \param sym           the symbol on this definition.
    /// \param type          the type of this definition
    /// \param semantics     the semantics of this definition
    /// \param loc           the location of the definition of NULL
    /// \param parent_scope  the parent scope of the definition
    /// \param outer         the definition for the same symbol in the outer scope if any
    /// \param id            the unique id of this definition
    explicit Def_function(
        Symbol         *sym,
        Type_function  *type,
        Semantics      semantics,
        Location const *loc,
        Scope          *parent_scope,
        Definition     *outer,
        size_t         id);

private:
    /// The scope that owns this definition.
    Scope *m_own_scope;

    /// The semantics of this function definition.
    Semantics m_semantics;
};

/// A type definition.
class Def_type GLSL_FINAL : public Definition
{
    typedef Definition Base;
    friend class mi::mdl::Arena_builder;
public:
    static Kind const s_kind = DK_TYPE;

public:
    /// Returns the kind of this definition.
    Kind get_kind() const GLSL_FINAL;

    /// Get the declaration of the definition if this is a struct definition.
    Declaration_struct *get_declaration() const GLSL_FINAL;

    /// Return scope that this definition creates.
    Scope *get_own_scope() const { return m_own_scope; }

    /// Set the scope that this definition creates.
    void set_own_scope(Scope *scope) { m_own_scope = scope; }

    /// Set the declaration.
    void set_declaration(Declaration_struct *decl) { m_decl = decl; }

    /// Change the type of this definition.
    void set_type(Type *type);

private:
    /// Constructor.
    ///
    /// \param sym           the symbol on this definition.
    /// \param type          the type of this definition
    /// \param loc           the location of the definition of NULL
    /// \param parent_scope  the parent scope of the definition
    /// \param outer         the definition for the same symbol in the outer scope if any
    /// \param id            the unique id of this definition
    explicit Def_type(
        Symbol         *sym,
        Type           *type,
        Location const *loc,
        Scope          *parent_scope,
        Definition     *outer,
        size_t         id);

private:
    /// The scope that owns this definition.
    Scope *m_own_scope;
};

/// A struct member definition.
class Def_member GLSL_FINAL : public Definition
{
    typedef Definition Base;
    friend class mi::mdl::Arena_builder;
public:
    static Kind const s_kind = DK_MEMBER;

public:
    /// Returns the kind of this definition.
    Kind get_kind() const GLSL_FINAL;

    /// Get the type of this definition.
    Type *get_type() const GLSL_FINAL;

    /// Get the declaration of the definition.
    Declaration_field *get_declaration() const GLSL_FINAL;

    /// Set the field index of a field member.
    void set_field_index(size_t idx) { m_field_index = idx; }

    /// Get the field index of a field member.
    size_t get_field_index() const { return m_field_index; }

private:
    /// Constructor.
    ///
    /// \param sym           the symbol on this definition.
    /// \param type          the type of this definition
    /// \param loc           the location of the definition of NULL
    /// \param parent_scope  the parent scope of the definition
    /// \param outer         the definition for the same symbol in the outer scope if any
    /// \param id            the unique id of this definition
    /// \param field_index   the field index of this member (if it is a struct field
    explicit Def_member(
        Symbol         *sym,
        Type           *type,
        Location const *loc,
        Scope          *parent_scope,
        Definition     *outer,
        size_t         id,
        size_t         field_index);

private:
    /// If this member is a field, its index inside the enclosing type.
    size_t m_field_index;
};

/// A operator definition.
class Def_operator GLSL_FINAL : public Definition
{
    typedef Definition Base;
    friend class mi::mdl::Arena_builder;
public:
    static Kind const s_kind = DK_OPERATOR;

public:
    /// Returns the kind of this definition.
    Kind get_kind() const GLSL_FINAL;

    /// Get the type of this definition.
    Type *get_type() const GLSL_FINAL;

    /// Get the declaration of the definition.
    ///
    /// \note Always returns NULL for operators.
    Declaration *get_declaration() const GLSL_FINAL;

private:
    /// Constructor.
    ///
    /// \param sym           the symbol on this definition.
    /// \param type          the type of this definition
    /// \param loc           the location of the definition of NULL
    /// \param parent_scope  the parent scope of the definition
    /// \param outer         the definition for the same symbol in the outer scope if any
    /// \param id            the unique id of this definition
    explicit Def_operator(
        Symbol         *sym,
        Type           *type,
        Location const *loc,
        Scope          *parent_scope,
        Definition     *outer,
        size_t         id);
};

/// Cast to subtype or return NULL if types do not match.
template<typename T>
T *as(Definition *def) {
    return def->get_kind() == T::s_kind ? static_cast<T *>(def) : NULL;
}

/// Cast to subtype or return NULL if types do not match.
template<typename T>
T const *as(Definition const *def) {
    return def->get_kind() == T::s_kind ? static_cast<T const *>(def) : NULL;
}

/// Cast to subtype or return NULL if types do not match or the definition is NULL.
template<typename T>
T *as_or_null(Definition *def) {
    if (def == NULL) {
        return NULL;
    }
    return def->get_kind() == T::s_kind ? static_cast<T *>(def) : NULL;
}

/// Cast to subtype or return NULL if types do not match or the definition is NULL.
template<typename T>
T const *as_or_null(Definition const *def) {
    if (def == NULL) {
        return NULL;
    }
    return def->get_kind() == T::s_kind ? static_cast<T const *>(def) : NULL;
}

/// Check if a statement is of a certain type.
template<typename T>
bool is(Definition const *def) {
    return as<T const>(def) != NULL;
}

/// A static_cast with check in debug mode.
template <typename T>
inline T *cast(Definition *arg) {
    GLSL_ASSERT(arg == NULL || is<T>(arg));
    return static_cast<T *>(arg);
}

/// A static_cast with check in debug mode.
template <typename T>
inline T *cast(Definition const *arg) {
    GLSL_ASSERT(arg == NULL || is<T>(arg));
    return static_cast<T const *>(arg);
}

/// An interface for visiting definitions
class IDefinition_visitor {
public:
    /// Called for every visited definition.
    ///
    /// \param def  the definition
    virtual void visit(Definition const *def) const = 0;
};

///
/// An environment scope.
///
/// Holds all entities declared or imported inside a scope.
class Scope : public Interface_owned
{
    typedef Interface_owned Base;
    friend class mi::mdl::Arena_builder;
    friend class Definition_table;

public:
    typedef list<Definition const *>::Type Definition_list;

    /// Return the unique id of this scope.
    size_t get_unique_id() const {
        return m_id;
    }

    /// Return the parent scope.
    Scope *get_parent() const {
        return m_parent;
    }

    /// Return the owner definition of this scope.
    Definition const *get_owner_definition() const {
        return m_owner_definition;
    }

    /// Set the owner definition of this scope.
    ///
    /// \param owner_def  the owner definition of this scope
    void set_owner_definition(Definition const *owner_def);

    /// Return the associated type of this scope.
    Type *get_scope_type() const {
        return m_scope_type;
    }

    /// Return the associated name of this scope if any.
    Symbol *get_scope_name() const {
        if (m_scope_type != NULL)
            return m_scope_type->get_sym();
        return NULL;
    }

    /// Add a definition to this scope.
    ///
    /// \param def  the definition to add
    void add_definition(Definition *def) {
        def->m_next   = m_definitions;
        m_definitions = def;
    }

    /// Returns the first definition in this scope.
    Definition *get_first_definition_in_scope() const {
        return m_definitions;
    }

    /// Re-enter the definitions for all defined symbols in the scope.
    ///
    /// \param owner_table  the definition table that owns this scope
    void enter_definitions(Definition_table *owner_table) const;

    /// Restore the definitions for all defined symbols in the scope.
    ///
    /// \param owner_table  the definition table that owns this scope
    void restore_definitions(Definition_table *owner_table);

    /// Find a definition inside this scope only.
    ///
    /// \param sym  the name of the entity to lookup
    Definition *find_definition_in_scope(Symbol *sym) const;

    /// Find the definition of the given ID in this scope only.
    ///
    /// \param ID  the ID of the definition
    Definition *find_definition_in_scope(size_t id) const;

    /// Find a definition inside this scope or parent scopes.
    ///
    /// \param sym  the name of the entity to lookup
    Definition *find_def_in_scope_or_parent(Symbol *sym) const;

    /// Walk over all definition inside this scope and all its sub-scopes.
    ///
    /// \param visitor  the visitor
    void walk(IDefinition_visitor const *visitor) const;

    /// Returns true if this is an empty scope that can be thrown away.
    bool is_empty() const;

    /// Get the first sub scope of a scope.
    Scope const *get_first_subscope() const { return m_sub_scopes; }

    /// Get the next (sibling) sub-scope.
    Scope const *get_next_subscope() const { return m_next_subscope; }

private:
    /// Creates a new environmental scope.
    ///
    /// \param parent     the parent scope or NULL
    /// \param id         an unique id for identifying this scope
    /// \param owner_def  the owner definition of this scope or NULL
    /// \param type       the type creating the scope or NULL
    explicit Scope(
        Scope            *parent,
        size_t           id,
        Definition const *owner_def,
        Type             *type);

    /// Remove this scope from its parent sub-scopes.
    void remove_from_parent();

private:
    /// List of all definitions in this scope.
    Definition *m_definitions;

    /// The owner definition of this scope if any.
    Definition const *m_owner_definition;

    /// Points to the parent scope.
    Scope *m_parent;

    /// Points to the head of all sub-scopes.
    Scope *m_sub_scopes;

    /// Points to the last sub-scope of the parent.
    Scope *m_last_sub_scope;

    /// Points to the next sub-scope of the parent.
    Scope *m_next_subscope;

    /// Points to the previous sub-scope of the parent.
    Scope *m_prev_subscope;

    /// If this scope represents a type, the associated one, else NULL.
    Type *m_scope_type;

    /// An unique id of this scope.
    size_t m_id;
};

/// The definition table.
class Definition_table {
    friend class Compilation_unit;
public:
    /// Enter a new scope owned by a definition.
    ///
    /// \param def  the definition that owns this scope or NULL
    Scope *enter_scope(Definition const *def);

    /// Enter a new scope created by a type declaration.
    ///
    /// \param type      the type defined by this scope
    /// \param type_def  the definition of the type
    Scope *enter_scope(Type *type, Definition const *type_def);

    /// Leave the current scope.
    ///
    /// This restores the definitions that were current before this scope was entered.
    ///
    /// The current scope will be set to the parent scope of the leaved one.
    void leave_scope();

    /// Reopen an already leaved scope.
    ///
    /// \param scope  the scope to open
    ///
    /// Node that scope must be a sub-scope of the current scope.
    void reopen_scope(Scope *scope);

    /// Do a transition to the given scope.
    ///
    /// \param scope  this scope will be the new current one, can be NULL
    ///
    /// Leave and enter scopes until the given scope is reached.
    void transition_to_scope(Scope *scope);

    /// Enter a new (entity) definition.
    ///
    /// \param kind       the kind of the definition to enter
    /// \param symbol     the symbol of the definition
    /// \param type       the type of the entity
    /// \param semantic   the semantic of this definition, if this is a function definition
    /// \param loc        the location of the definition
    Definition *enter_definition(
        Definition::Kind        kind,
        Symbol                  *symbol,
        Type                    *type,
        Def_function::Semantics semantic,
        Location const          *loc);

    /// Enter a new type definition.
    ///
    /// \param symbol     the symbol of the type
    /// \param type       the type of the type
    /// \param loc        the location of the type
    Def_type *enter_type_definition(
        Symbol         *symbol,
        Type           *type,
        Location const *loc);

    /// Enter a new function definition.
    ///
    /// \param symbol     the symbol of the function
    /// \param type       the type of the function
    /// \param semantics  the semantics of this definition
    /// \param loc        the location of the function
    Def_function *enter_function_definition(
        Symbol                   *symbol,
        Type_function            *type,
        Def_function::Semantics   semantics,
        Location const           *loc);

    /// Enter a new variable definition.
    ///
    /// \param symbol  the symbol of the variable
    /// \param type    the type of the variable
    /// \param loc     the location of the variable
    Def_variable *enter_variable_definition(
        Symbol         *symbol,
        Type           *type,
        Location const *loc);

    /// Enter a new parameter definition.
    ///
    /// \param symbol  the symbol of the parameter
    /// \param type    the type of the parameter
    /// \param loc     the location of the parameter
    Def_param *enter_parameter_definition(
        Symbol         *symbol,
        Type           *type,
        Location const *loc);

    /// Enter a new member definition.
    ///
    /// \param symbol       the symbol of the member
    /// \param type         the type of the member
    /// \param field_index  if this member is a field, its index
    /// \param loc          the location of the member
    Def_member *enter_member_definition(
        Symbol         *symbol,
        Type           *type,
        size_t         field_index,
        Location const *loc);

    /// Enter a new operator definition.
    ///
    /// \param kind    the operator kind of the definition to enter
    /// \param symbol  the symbol of the definition
    /// \param type    the type of the entity
    Def_operator *enter_operator_definition(
        Expr::Operator kind,
        Symbol         *symbol,
        Type           *type);

    /// Enter an error definition for the given symbol.
    ///
    /// \param symbol    this symbol will be defined in the current scope as an error
    /// \param err_type  the error type
    Definition *enter_error(
        Symbol     *symbol,
        Type_error *err_type);

    /// Return the current scope of the compilation unit.
    Scope *get_curr_scope() const { return m_curr_scope; }

    /// Return the outer scope of the compilation unit.
    /// This scope contains the predefined definitions.
    Scope *get_predef_scope() const { return m_predef_scope; }

    /// Return the global scope of the compilation unit.
    Scope *get_global_scope() const { return m_global_scope; }

    /// Return a scope by its type.
    ///
    /// \param type  the type to lookup
    Scope *get_type_scope(
        Type *type) const;

    /// Return the current definition for a symbol in this definition table.
    ///
    /// \param sym  the symbol
    Definition *get_definition(
        Symbol *sym) const;

    /// Set the current definition for a symbol in this definition table.
    ///
    /// \param sym  the symbol
    /// \param def  the current definition for this symbol
    void set_definition(
        Symbol     *sym,
        Definition *def);

    /// Restore the current definition for a symbol in this definition table.
    ///
    /// \param sym  the symbol
    /// \param def  the restored definition for this symbol
    void restore_definition(
        Symbol     *sym,
        Definition *def);

    /// Return the definition for an operator in this definition table.
    ///
    /// \param op  the operator kind
    Definition *get_operator_definition(
        Expr::Operator op) const;

    /// Set the definition for an operator in this definition table.
    ///
    /// \param op   the operator kind
    /// \param def  the current definition for this operator
    void set_operator_definition(
        Expr::Operator op,
        Definition     *def);

    /// Walk over all definitions of this definition table.
    ///
    /// \param visitor  the visitor
    void walk(
        IDefinition_visitor const *visitor) const;

    /// Clear the definition table.
    void clear();

    /// Remove an empty scope from the scope tree.
    ///
    /// \param scope  An empty scope (i.e. scope->is_empty() == true).
    void remove_empty_scope(Scope *scope);

private:
    /// Create a new definition table.
    ///
    /// \param owner  the owner module of this definition table
    explicit Definition_table(Compilation_unit &owner);

private:
    // non copyable
    Definition_table(Definition_table const &) GLSL_DELETED_FUNCTION;
    Definition_table &operator=(Definition_table const &) GLSL_DELETED_FUNCTION;

private:
    /// Create a new scope.
    ///
    /// \param parent     the parent scope or NULL
    /// \param id         an unique id for identifying this scope
    /// \param owner_def  the owner definition of this scope or NULL
    /// \param type       the type creating the scope or NULL
    Scope *create_scope(
        Scope            *parent,
        size_t           id,
        Definition const *owner_def = NULL,
        Type             *type = NULL);

private:
    /// The owner unit of this definition table
    Compilation_unit &m_unit;

    /// Points to the top of the scope stack.
    Scope *m_curr_scope;

    /// The outer scope of the compilation unit. Contains all predefined entities.
    Scope *m_predef_scope;

    /// The global scope of the compilation unit. Contains all definitions of this unit.
    Scope *m_global_scope;

    /// The list of free scopes.
    Scope *m_free_scopes;

    /// The next id for a definition.
    size_t m_next_definition_id;

    /// Memory arena for all sub objects.
    Memory_arena m_arena;

    /// Builder for sub objects.
    Arena_builder m_builder;

    typedef ptr_hash_map<Type, Scope *>::Type Type_scope_map;

    /// Associate types to scopes.
    Type_scope_map m_type_scopes;

    typedef vector<Definition *>::Type Def_vector;

    /// Vector of entity Definitions in this table for lookup, index by its symbol ids.
    Def_vector m_definitions;

    /// The definitions of all operators.
    Definition *m_operator_definitions[Expr::OK_LAST];

public:
    /// Helper class for scope transitions using RAII.
    class Scope_transition {
    public:
        /// Remember the current scope and transition to another scope.
        ///
        /// \param def_table   the definition table
        /// \param scope       the destination scope
        Scope_transition(
            Definition_table &def_tab,
            Scope            *scope)
        : m_deftab(def_tab), m_curr_scope(def_tab.get_curr_scope())
        {
            def_tab.transition_to_scope(scope);
        }

        /// Return to the previous scope.
        ~Scope_transition()
        {
            m_deftab.transition_to_scope(m_curr_scope);
        }

    private:
        /// The definition table.
        Definition_table &m_deftab;

        /// The current scope before the transition.
        Scope            *m_curr_scope;
    };

    /// Helper class for scope entering using RAII.
    class Scope_enter {
    public:
        /// Enter a new scope.
        ///
        /// \param def_table   the definition table
        /// \param def         the (function) definition that "owns" the newly created scope
        Scope_enter(
            Definition_table &def_tab,
            Definition       *def)
        : m_def_tab(def_tab)
        {
            Scope *scope = m_def_tab.enter_scope(def);
            if (def != NULL && is<Def_function>(def)) {
                cast<Def_function>(def)->set_own_scope(scope);
            }
        }

        /// Reopen given scope.
        ///
        /// \param def_table   the definition table
        /// \param scope       the scope that will be reopened
        Scope_enter(
            Definition_table &def_tab,
            Scope            *scope)
        : m_def_tab(def_tab)
        {
            m_def_tab.reopen_scope(scope);
        }

        /// Enter a new struct type scope.
        ///
        /// \param def_table   the definition table
        /// \param type        the type that "owns" the new scope
        /// \param type_def    the definition of the struct type
        Scope_enter(
            Definition_table &def_tab,
            Type             *type,
            Definition       *type_def)
        : m_def_tab(def_tab)
        {
            Scope *scope = m_def_tab.enter_scope(type, type_def);
            if (is<Def_type>(type_def)) {
                cast<Def_type>(type_def)->set_own_scope(scope);
            }
        }

        /// Leave current scope.
        ~Scope_enter() { m_def_tab.leave_scope(); }

    private:
        /// The definition table.
        Definition_table &m_def_tab;
    };
};

}  // glsl
}  // mdl
}  // mi

#endif
