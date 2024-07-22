/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

/** \file
 ** \brief
 **/


#include "pch.h"

#define MI_TEST_AUTO_SUITE_NAME "Regression Test Suite for prod/lib/neuray"
#define MI_TEST_IMPLEMENT_TEST_MAIN_INSTEAD_OF_MAIN

#include <base/system/test/i_test_auto_driver.h>
#include <base/system/test/i_test_auto_case.h>

#include <mi/base/config.h>
#include <mi/base/handle.h>
#include <mi/base/interface_implement.h>

#include <mi/neuraylib/idatabase.h>
#include <mi/neuraylib/ifactory.h>
#include <mi/neuraylib/imdl_configuration.h>
#include <mi/neuraylib/imdl_factory.h>
#include <mi/neuraylib/imdl_execution_context.h>
#include <mi/neuraylib/imdl_impexp_api.h>
#include <mi/neuraylib/ineuray.h>
#include <mi/neuraylib/itransaction.h>
#include <mi/neuraylib/iscope.h>

#include <optional>
#include <vector>

#include "test_shared.h"

char const *ok_export_category_and_struct_mdl = R"(
// OK test case: export category and struct with category definition
mdl 1.9;

export struct_category cat;
export struct stru in cat {
  float g;
};
)";

char const *ok_import_category_and_struct_mdl = R"(
// OK test case: import category and struct with category
mdl 1.9;

import ::ok::export_category_and_struct::cat;
import ::ok::export_category_and_struct::stru;
)";

char const *ok_use_category_and_struct_wc_mdl = R"(
// OK test case: import and use category and struct with category using wildcard
mdl 1.9;

import ::ok::export_category_and_struct::*;

export struct t in ::ok::export_category_and_struct::cat {
  float h;
};

export float f(::ok::export_category_and_struct::stru s1, t s2) = s1.g + s2.h;
)";

char const *ok_use_category_and_struct1_mdl = R"(
// OK test case: import and use category and struct with category
mdl 1.9;

import ::ok::export_category_and_struct::cat;
import ::ok::export_category_and_struct::stru;

export struct t in ::ok::export_category_and_struct::cat {
  float h;
};

export float f(::ok::export_category_and_struct::stru s1, t s2) = s1.g + s2.h;
)";

char const *ok_use_category_and_struct2_mdl = R"(
// OK test case: import and use category and struct with category
mdl 1.9;

using ::ok::export_category_and_struct import cat, stru;

export struct t in cat {
  float h;
};

export float f(stru s1, t s2) = s1.g + s2.h;
)";

char const *ok_annotated_struct_with_cat1_mdl = R"(
// OK test case: annotated struct definition with category
mdl 1.9;

using ::anno import *;

export struct_category bar_struct_category;

// fails
export struct s2 in bar_struct_category [[ description("struct annotation") ]] {
    int a;
};
)";

char const *ok_overloaded_decl_func_mdl = R"(
// OK test case: overloaded declarative function
mdl 1.9;

using ::anno import *;

export declarative int f(int i) = i;
export declarative int f(float f) = int(f);
)";

char const *ok_overloaded_decl_fn1_mdl = R"(
// OK test case: overloading of declarative function where first one is explicity declarative.
mdl 1.9;

export declarative int f() = 1;
export int f(int i) = i;
)";

char const *ok_overloaded_decl_fn2_mdl = R"(
// OK test case: overloading of declarative where second one is explicitly declarative.
mdl 1.9;

export int f() = 1;
export declarative int f(int i) = i;
)";

char const *ok_overloaded_decl_fn3_mdl = R"(
// OK test case: overloading of declarative function where first one is implicitly declarative.
mdl 1.9;

export declarative struct s {
  int x;
};

export int f(s s0) = s0.x;
export int f() = 2;
)";

char const *ok_overloaded_decl_fn4_mdl = R"(
// OK test case: overloading of declarative and non-declarative function.
mdl 1.9;

export declarative struct s {
  int x;
};

export int f() = 2;
export int f(s s0) = s0.x;
)";

char const *ok_nested_decl_structs_mdl = R"(
// OK test case: nested declarative structs.
mdl 1.9;

declarative struct s1 {
};

declarative struct s2 {
  s1 s;
};
)";

char const *ok_typedef_of_decl_struct_mdl = R"(
// OK test case: typedef of declarative struct
mdl 1.9;

using ::ok::export_category_and_struct import cat;

export struct mat in cat {
  float f;
};

typedef mat bat;

export float g(bat b) =
  b.f;
  
export float h(mat m = mat(2.0)) =
  g(m);
)";

char const *ok_decl_struct_with_bsdf_field_mdl = R"(
// OK test case: declarative struct with bsdf field
mdl 1.9;

import ::state::*;

export declarative struct aov_material in material_category {
    float3 example = float3(0.315f, 0.625f, 1.0f) * state::normal();
    int magic = 42;
    bsdf scattering = bsdf();
};
)";

char const *ok_bsdf_return_type_mdl = R"(
// OK test case: BSDF as return type of declarative function.
mdl 1.9;

import ::df::*;

export bsdf f1() = bsdf();
export declarative bsdf f2() = ::df::diffuse_reflection_bsdf();
)";

char const *warning_bad_annotation1_mdl = R"(
// OK test case: annotated category definition
mdl 1.9;

import ::anno::noinline;

struct_category cat [[ ::anno::noinline() ]];
)";

char const *warning_bad_annotation2_mdl = R"(
// OK test case: annotated category definition
mdl 1.9;

struct_category cat [[ unknown() ]];
)";

char const *warning_bad_annotation3_mdl = R"(
// WARNING test case: unsupported annotation on categories
mdl 1.9;

import ::anno::soft_range;

struct_category cat [[ ::anno::soft_range(1, 9) ]];
)";

char const *error_redeclaration1_mdl = R"(
// ERROR test case: redeclaration
mdl 1.9;

struct cat {
  int i;
};

// Structs and categories are in the same namespace, redeclaration is not allowed.
struct_category cat;
)";

char const *error_redeclaration3_mdl = R"(
// ERROR test case: redeclaration
mdl 1.9;

struct_category cat;

// Functions and categories are in the same namespace, redeclaration is not allowed.
int cat() = 1;
)";

char const *error_multi_declaration1_mdl = R"(
// ERROR test case: multiple category declarations
mdl 1.9;

struct_category cat;
struct_category cat;
)";

char const *error_multi_declaration2_mdl = R"(
// ERROR test case: multiple category declarations with import
mdl 1.9;

using ::ok::export_category_and_struct import cat;
struct_category cat;
)";

char const *error_decl_var_in_assign_mdl = R"(
// ERROR test case: declarative variable in assignment.
mdl 1.9;

declarative struct mat {
  float f;
};

export int f() {
    mat m;
    m = mat(1.0);
    return 23;
}
)";

char const *error_annotated_struct_with_cat1_mdl = R"(
// ERROR test case: wrongly annotated struct definition with category
mdl 1.9;

using ::anno import *;

export struct_category bar_struct_category;

// should fail
export struct s1 [[ description("struct annotation") ]] in bar_struct_category {
    int a;
};
)";

char const *_mdl = R"(
)";

struct Test_module {
    char const *name;
    char const *src;
    mi::Sint32 counts[3]; // result, error count, msg count
    std::optional<std::vector<std::string>> exp_msgs;
};

Test_module module_sources[] = {
    // name                               // source                           // result, 
                                                                              // error cnt,
                                                                              // msg cnt
    { "::ok::export_category_and_struct", ok_export_category_and_struct_mdl,  {0, 0, 1}, {} },
    { "::ok::import_category_and_struct", ok_import_category_and_struct_mdl,  {0, 0, 1}, {} },
    { "::ok::use_category_and_struct_wc", ok_use_category_and_struct_wc_mdl,  {0, 0, 1}, {} },
    { "::ok::use_category_and_struct1",   ok_use_category_and_struct1_mdl,    {0, 0, 1}, {} },
    { "::ok::use_category_and_struct2",   ok_use_category_and_struct2_mdl,    {0, 0, 1}, {} },
    { "::ok::annotated_struct_with_cat1", ok_annotated_struct_with_cat1_mdl,  {0, 0, 1}, {} },
    { "::ok::overloaded_decl_func",       ok_overloaded_decl_func_mdl,        {0, 0, 1}, {} },
    { "::ok::overloaded_decl_fn1",        ok_overloaded_decl_fn1_mdl,         {0, 0, 1}, {} },
    { "::ok::overloaded_decl_fn2",        ok_overloaded_decl_fn2_mdl,         {0, 0, 1}, {} },
    { "::ok::overloaded_decl_fn3",        ok_overloaded_decl_fn3_mdl,         {0, 0, 1}, {} },
    { "::ok::overloaded_decl_fn4",        ok_overloaded_decl_fn4_mdl,         {0, 0, 1}, {} },
    { "::ok::nested_decl_structs",        ok_nested_decl_structs_mdl,         {0, 0, 1}, {} },
    { "::ok::typedef_of_decl_struct",     ok_typedef_of_decl_struct_mdl,      {0, 0, 1}, {} },
    { "::ok::decl_struct_with_bsdf_field", ok_decl_struct_with_bsdf_field_mdl, {0, 0, 1}, {} },
    { "::ok::bsdf_return_type_mdl",       ok_bsdf_return_type_mdl,            {0, 0, 1}, {} },
    { "::warning::bad_annotation1",       warning_bad_annotation1_mdl,        {0, 0, 3}, {} },
    { "::warning::bad_annotation2",       warning_bad_annotation2_mdl,        {0, 0, 5}, {} },
    { "::warning::bad_annotation3",       warning_bad_annotation3_mdl,        {0, 0, 3}, {} },
    { "::error::redeclaration1",          error_redeclaration1_mdl,          {-2, 1, 2},
        {{"C107 redeclaration of 'cat' as a different kind of symbol"}} },
    { "::error::redeclaration3",          error_redeclaration3_mdl,          {-2, 1, 2},
        {{"C107 redeclaration of 'cat' as a different kind of symbol"}} },
    { "::error::multi_declaration1",      error_multi_declaration1_mdl,      {-2, 1, 2},
        {{"C104 redeclaration of 'cat'"}} },
    { "::error::multi_declaration2",      error_multi_declaration2_mdl,      {-2, 1, 2},
        {{"C104 redeclaration of 'cat'"}} },
    { "::error::decl_decl_var_in_assign_fn", error_decl_var_in_assign_mdl,   {-2, 3, 5},
        {{"C361 declarative type 'struct ::error::decl_decl_var_in_assign_fn::mat' used in non-declarative context"}}},
   { "::error::annotated_struct_with_cat1", error_annotated_struct_with_cat1_mdl,  {-2, 2, 2},
        {{ "C100 \"{\" expected" } } },
};

void check_modules(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IFactory* factory,
    mi::neuraylib::IMdl_factory* mdl_factory,
    mi::neuraylib::IMdl_impexp_api* mdl_impexp_api)
{
    for (Test_module &tm : module_sources) {
        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
            mdl_factory->create_execution_context());
#define SHOW_TRACE 0
#if SHOW_TRACE
        printf("[+] testing %s\n", tm.name);
#endif
        int result = mdl_impexp_api->load_module_from_string(transaction, tm.name,
            tm.src, context.get());
#if SHOW_TRACE
        printf("[+] testing %s: %d (err: %d (exp: %d), msg: %d (exp: %d))\n", tm.name, result,
            int(context->get_error_messages_count()), int(tm.counts[1]), int(context->get_messages_count()), int(tm.counts[2]));
        for (mi::Size i = 0; i < context->get_messages_count(); ++i) {
            printf("[!] %s\n", context->get_message(i)->get_string());
        }
#endif
        if (tm.exp_msgs.has_value()) {
            for (std::string &exp_msg : tm.exp_msgs.value()) {
                bool found = false;
                for (mi::Size i = 0; i < context->get_messages_count(); ++i) {
                    mi::base::Handle<const mi::neuraylib::IMessage> act_msg_imessage(
                        context->get_message(i));
                    std::string act_msg{ act_msg_imessage->get_string() };
#if SHOW_TRACE
                    printf("[=] %s <-> %s\n", exp_msg.c_str(), act_msg.c_str());
#endif
                    size_t p = act_msg.find(exp_msg);
                    found = found || p != act_msg.npos;
                }
                MI_CHECK(found && "expected message not found");
            }

        }
        MI_CHECK_EQUAL(result, tm.counts[0]);
        MI_CHECK_EQUAL(context->get_error_messages_count(), tm.counts[1]);
        MI_CHECK_EQUAL(context->get_messages_count(), tm.counts[2]);
    }
}

void run_tests(mi::neuraylib::INeuray* neuray)
{
    MI_CHECK_EQUAL(0, neuray->start());

    {
        mi::base::Handle<mi::neuraylib::IDatabase> database(
            neuray->get_api_component<mi::neuraylib::IDatabase>());
        mi::base::Handle<mi::neuraylib::IScope> global_scope(database->get_global_scope());
        mi::base::Handle<mi::neuraylib::ITransaction> transaction(
            global_scope->create_transaction());

        mi::base::Handle<mi::neuraylib::IFactory> factory(
            neuray->get_api_component<mi::neuraylib::IFactory>());
        MI_CHECK(factory);

        mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
            neuray->get_api_component<mi::neuraylib::IMdl_factory>());
        MI_CHECK(mdl_factory);

        mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api(
            neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());
        MI_CHECK(mdl_impexp_api);

        mi::base::Handle<mi::neuraylib::IMdl_configuration> mdl_configuration(
            neuray->get_api_component<mi::neuraylib::IMdl_configuration>());
        MI_CHECK(mdl_configuration);

        check_modules(transaction.get(), factory.get(), mdl_factory.get(),
            mdl_impexp_api.get());

        MI_CHECK_EQUAL(0, transaction->commit());
    }
    MI_CHECK_EQUAL(0, neuray->shutdown());
}

MI_TEST_AUTO_FUNCTION( test_declarative_structs )
{
    mi::base::Handle<mi::neuraylib::INeuray> neuray( load_and_get_ineuray());
    MI_CHECK( neuray);

    {
        run_tests(neuray.get());
        // MDL SDK must be able to run the test a second time, test that
        run_tests(neuray.get());
    }

    neuray = nullptr;
    MI_CHECK( unload());
}

MI_TEST_MAIN_CALLING_TEST_MAIN();

