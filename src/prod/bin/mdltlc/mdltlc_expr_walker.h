/******************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDLTLC_EXPR_WALKER_H
#define MDLTLC_EXPR_WALKER_H 1

#include "mdltlc_exprs.h"

class Expr_visitor {
public:
    virtual void visit(Expr_invalid *expr) {}
    virtual void visit(Expr_literal *expr) {}
    virtual void visit(Expr_ref *expr) {}
    virtual void visit(Expr_unary *expr) {}
    virtual void visit(Expr_binary *expr) {}
    virtual void visit(Expr_conditional *expr) {}
    virtual void visit(Expr_call *expr) {}
    virtual void visit(Expr_type_annotation *expr) {}
    virtual void visit(Expr_attribute *expr) {}
};

class Const_expr_visitor {
public:
    virtual void visit(Expr_invalid const *expr) {}
    virtual void visit(Expr_literal const *expr) {}
    virtual void visit(Expr_ref const *expr) {}
    virtual void visit(Expr_unary const *expr) {}
    virtual void visit(Expr_binary const *expr) {}
    virtual void visit(Expr_conditional const *expr) {}
    virtual void visit(Expr_call const *expr) {}
    virtual void visit(Expr_type_annotation const *expr) {}
    virtual void visit(Expr_attribute const *expr) {}
};

class Expr_walker {
public:
    Expr_walker(Expr_visitor &visitor)
        : m_visitor(visitor) {}

    void walk(Expr *expr);

private:
    Expr_visitor &m_visitor;
};

class Const_expr_walker {
public:
    Const_expr_walker(Const_expr_visitor &visitor)
        : m_visitor(visitor) {}

    void walk(Expr const *expr);

private:
    Const_expr_visitor &m_visitor;
};


#endif
