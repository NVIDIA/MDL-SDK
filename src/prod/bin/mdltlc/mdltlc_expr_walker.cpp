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

#include "pch.h"

#include "mdltlc_expr_walker.h"

void Expr_walker::walk(Expr *expr) {
    switch (expr->get_kind()) {
    case Expr::EK_INVALID:
    {
        Expr_invalid *e = cast<Expr_invalid>(expr);
        m_visitor.visit(e);
        break;
    }

    case Expr::EK_LITERAL:
    {
        Expr_literal *e = cast<Expr_literal>(expr);
        m_visitor.visit(e);
        break;
    }

    case Expr::EK_REFERENCE:
    {
        Expr_ref *e = cast<Expr_ref>(expr);
        m_visitor.visit(e);
        break;
    }

    case Expr::EK_TYPE_ANNOTATION:
    {
        Expr_type_annotation *e = cast<Expr_type_annotation>(expr);
        m_visitor.visit(e);

        walk(e->get_argument());
        break;
    }

    case Expr::EK_BINARY:
    {
        Expr_binary *e = cast<Expr_binary>(expr);
        m_visitor.visit(e);

        walk(e->get_left_argument());
        walk(e->get_right_argument());
        break;
    }

    case Expr::EK_UNARY:
    {
        Expr_unary *e = cast<Expr_unary>(expr);
        m_visitor.visit(e);

        walk(e->get_argument());
        break;
    }

    case Expr::EK_CONDITIONAL:
    {
        Expr_conditional *e = cast<Expr_conditional>(expr);
        m_visitor.visit(e);

        walk(e->get_condition());
        walk(e->get_true());
        walk(e->get_false());
        break;
    }

    case Expr::EK_CALL:
    {
        Expr_call *e = cast<Expr_call>(expr);
        m_visitor.visit(e);

        // Not going down the callee, because it is always a reference.

        for (int i = 0; i < e->get_argument_count(); i++) {
            walk(e->get_argument(i));
        }
        break;
    }

    case Expr::EK_ATTRIBUTE:
    {
        Expr_attribute *e = cast<Expr_attribute>(expr);
        m_visitor.visit(e);

        walk(e->get_argument());

        Expr_attribute::Expr_attribute_vector &attrs = e->get_attributes();
        for (size_t i = 0; i < attrs.size(); i++) {
            Expr_attribute::Expr_attribute_entry &attr = attrs[i];
            if (attr.expr)
                walk(attr.expr);
        }
        break;
    }
    }
}

void Const_expr_walker::walk(Expr const *expr) {
    switch (expr->get_kind()) {
    case Expr::EK_INVALID:
    {
        Expr_invalid const *e = cast<Expr_invalid>(expr);
        m_visitor.visit(e);
        break;
    }

    case Expr::EK_LITERAL:
    {
        Expr_literal const *e = cast<Expr_literal>(expr);
        m_visitor.visit(e);
        break;
    }

    case Expr::EK_REFERENCE:
    {
        Expr_ref const *e = cast<Expr_ref>(expr);
        m_visitor.visit(e);
        break;
    }

    case Expr::EK_TYPE_ANNOTATION:
    {
        Expr_type_annotation const *e = cast<Expr_type_annotation>(expr);
        m_visitor.visit(e);

        walk(e->get_argument());
        break;
    }

    case Expr::EK_BINARY:
    {
        Expr_binary const *e = cast<Expr_binary>(expr);
        m_visitor.visit(e);

        walk(e->get_left_argument());
        walk(e->get_right_argument());
        break;
    }

    case Expr::EK_UNARY:
    {
        Expr_unary const *e = cast<Expr_unary>(expr);
        m_visitor.visit(e);

        walk(e->get_argument());
        break;
    }

    case Expr::EK_CONDITIONAL:
    {
        Expr_conditional const *e = cast<Expr_conditional>(expr);
        m_visitor.visit(e);

        walk(e->get_condition());
        walk(e->get_true());
        walk(e->get_false());
        break;
    }

    case Expr::EK_CALL:
    {
        Expr_call const *e = cast<Expr_call>(expr);
        m_visitor.visit(e);

        // Not going down the callee, because it is always a reference.

        for (int i = 0; i < e->get_argument_count(); i++) {
            walk(e->get_argument(i));
        }
        break;
    }

    case Expr::EK_ATTRIBUTE:
    {
        Expr_attribute const *e = cast<Expr_attribute>(expr);
        m_visitor.visit(e);

        walk(e->get_argument());

        Expr_attribute::Expr_attribute_vector const &attrs = e->get_attributes();
        for (size_t i = 0; i < attrs.size(); i++) {
            Expr_attribute::Expr_attribute_entry const &attr = attrs[i];
            if (attr.expr) {
                walk(attr.expr);
            }
        }
        break;
    }
    }
}

