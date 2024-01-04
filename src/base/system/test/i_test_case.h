/******************************************************************************
 * Copyright (c) 2007-2023, NVIDIA CORPORATION. All rights reserved.
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
/// \brief regression test infrastructure
///
/// - class Test_case                 abstract test case class interface
/// - class Named_test_case           abstract interface for named test cases
/// - class Function_test_case        class wrapper for function test cases
/// - class Method_test_case          class wrapper for method test cases
///
/// - MI_TEST_FUNCTION()              define a function test case
/// - MI_TEST_METHOD()                define a method test case
///
/// - MI_CHECK()                      check a boolean expression
/// - MI_CHECK_CLOSE()                check |lhs - rhs| < epsilon
/// - MI_CHECK_CLOSE_COLLECTIONS()    check that two containers are approx. equal
/// - MI_CHECK_EQUAL()                check lhs == rhs
/// - MI_CHECK_EQUAL_COLLECTIONS()    check [b1,e1) == [b2,e2)
/// - MI_CHECK_NOT_EQUAL()            check lhs != rhs
/// - MI_CHECK_LESS()                 check lhs < rhs
/// - MI_CHECK_GREATER()              check lhs > rhs
/// - MI_CHECK_LESS_OR_EQUAL()        check lhs <= rhs
/// - MI_CHECK_GREATER_OR_EQUAL()     check lhs >= rhs
///
/// - MI_REQUIRE()                    require a boolean expression
/// - MI_REQUIRE_CLOSE()              require |lhs - rhs| < epsilon
/// - MI_REQUIRE_CLOSE_COLLECTIONS()  require that two containers are approx. equal
/// - MI_REQUIRE_EQUAL()              require lhs == rhs
/// - MI_REQUIRE_EQUAL_COLLECTIONS()  require that two containers are equal
/// - MI_REQUIRE_NOT_EQUAL()          require lhs != rhs
/// - MI_REQUIRE_LESS()               require lhs < rhs
/// - MI_REQUIRE_GREATER()            require lhs > rhs
/// - MI_REQUIRE_LESS_OR_EQUAL()      require lhs <= rhs
/// - MI_REQUIRE_GREATER_OR_EQUAL()   require lhs >= rhs


#ifndef BASE_SYSTEM_TEST_CASE_H
#define BASE_SYSTEM_TEST_CASE_H

#include "i_test_pretty_print.h"
#include "i_test_environment.h"
#include "i_test_current_location.h"
#include <base/system/stlext/i_stlext_concepts.h>
#include <cassert>
#include <iterator>
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace MI { namespace TEST {

struct Test_case : public STLEXT::Abstract_interface
{
    virtual std::string const & name() const = 0;
    virtual void                  run()        = 0;
};

class Named_test_case : public Test_case
{
    std::string const _name;

public:
    explicit Named_test_case(std::string const & name) : _name(name)
    {
    }
    std::string const & name() const
    {
        return _name;
    }
};

class Function_test_case : public Named_test_case
{
    void (* const _f)();

public:
    Function_test_case(std::string const & name, void (*f)()) : Named_test_case(name), _f(f)
    {
        assert(f);
    }
    void run()
    {
        (*_f)();
    }
};

template <class T>
class Method_test_case : public Named_test_case
{
    T & _object;
    void (T::*_method)();

public:
    Method_test_case(std::string const & name, T & obj, void (T::*method)())
        : Named_test_case(name), _object(obj), _method(method)
    {
        assert(_method);
    }
    void run()
    {
        (_object.*_method)();
    }
};

template <class T>
inline Method_test_case<T> * new_method_test_case(std::string const & name, T & obj, void (T::*method)())
{
    return new Method_test_case<T>(name, obj, method);
}

class Runtime_error : public std::runtime_error
{
    std::string const _msg;

public:
    Runtime_error(std::string const & ctx) : std::runtime_error(""), _msg(ctx)
    {
        std::cerr << what() << std::endl;
    }
    ~Runtime_error() throw()
    {
    }
    char const * what() const throw()
    {
        return _msg.c_str();
    }
};

struct Test_case_failure : public Runtime_error
{
    Test_case_failure( std::string const &    lit
                     , std::string const &    exp
                     , std::string const &    ctx
                     )
    : Runtime_error( ctx + ": failed check \"" + lit + "\""
                   + (exp.empty() ? std::string() : std::string(" [") + exp + "]")
                   )
    {
    }
};

struct Test_suite_failure : public Runtime_error
{
    Test_suite_failure( std::string const &    lit
                      , std::string const &    exp
                      , std::string const &    ctx
                      )
    : Runtime_error( ctx + ": failed critical check \"" + lit + "\""
                   + (exp.empty() ? std::string("") : std::string(" [") + exp + "]")
                   )
    {
    }
};

struct Test_case_skipped : public Runtime_error
{
    Test_case_skipped(const std::string& msg = {})
    : Runtime_error(msg)
    {}
};

#define MI_TEST_IMPL(result_type,result,literal,expression,extra_scope)                 \
    do                                                                                  \
    {                                                                                   \
        if (!(result))                                                                  \
        {                                                                               \
            std::ostringstream __mi_test_os;                                          \
            __mi_test_os.setf( std::ios::fixed );                                     \
            __mi_test_os.unsetf( std::ios::showpoint );                               \
            __mi_test_os.precision( 16 );                                               \
            __mi_test_os << extra_scope;                                                \
            std::string __mi_test_scope( __mi_test_os.str() );                        \
            if (!__mi_test_scope.empty())                                               \
                __mi_test_scope.insert(__mi_test_scope.begin(), ':');                   \
            __mi_test_scope.insert(0u, MI::TEST::show(MI_CURRENT_LOCATION));            \
            result_type __mi_test_r(literal, expression, __mi_test_scope);              \
            throw __mi_test_r;                                                          \
        }                                                                               \
    }                                                                                   \
    while(false)

template <class Number_type>
inline Number_type abs_distance(Number_type const & lhs, Number_type const & rhs)
{
    return lhs > rhs ? lhs - rhs : rhs - lhs;
}

template <class Result_type, class Iter1, class Iter2>
inline void is_equal_collection( std::string const &    literal
                               , Current_location const & location
                               , Iter1 lb, Iter1 le
                               , Iter2 rb, Iter2 re
                               )
{
    std::string errors;

    for (size_t i(0); lb != le && rb != re; ++i, ++lb, ++rb)
    {
        if (!( *lb == *rb ))
        {
            if (!errors.empty()) errors += "; ";
            errors += std::string("index ") + show(i) + ": " + show(*lb) + " != " + show(*rb);
        }
    }
    if (!errors.empty())
    {
        Result_type r(literal, errors, show(location));
        throw r;
    }
    else if (lb != le || rb != re)
    {
        Result_type r(literal, "collections differ in length", show(location));
        throw r;
    }
}

template <class Result_type, class Iter1, class Iter2, class Number_type>
inline void is_close_collection( std::string const &    literal
                               , Current_location const & location
                               , Iter1 lb, Iter1 le
                               , Iter2 rb, Iter2 re
                               , Number_type eps
                               )
{
    std::string errors;

    for (size_t i(0); lb != le && rb != re; ++i, ++lb, ++rb)
    {
        if (!( abs_distance( *lb, *rb) < eps))
        {
            if (!errors.empty()) errors += "; ";
            errors += std::string("index ") + show(i) + ": abs(" + show(*lb) + " - " + show(*rb) + ") >= " + show(eps);
        }
    }
    if (!errors.empty())
    {
        Result_type r(literal, errors, show(location));
        throw r;
    }
    else if (lb != le || rb != re)
    {
        Result_type r(literal, "collections differ in length", show(location));
        throw r;
    }
}

}} // MI::TEST

#define MI_TEST_FUNCTION(func)                                                          \
    new MI::TEST::Function_test_case(MI::TEST::pretty_function_name(#func), &(func))

#define MI_TEST_METHOD(object, method)                                                  \
    new MI::TEST::Method_test_case<object>( MI::TEST::pretty_function_name(#method)     \
                                          , *this, &object::method                      \
                                          )

//
// MI_CHECK Macros
//

#define MI_CHECK_MSG(expr,extra_scope)                                                  \
  MI_TEST_IMPL( MI::TEST::Test_case_failure                                             \
              , (expr)                                                                  \
              , #expr                                                                   \
              , ""                                                                      \
              , extra_scope                                                             \
              )

#define MI_CHECK_EQUAL_MSG(lhs,rhs,extra_scope)                                         \
  MI_TEST_IMPL( MI::TEST::Test_case_failure                                             \
              , (lhs) == (rhs)                                                          \
              , std::string(#lhs) + " == " + std::string(#rhs)                      \
              , MI::TEST::show(lhs) + " != " + MI::TEST::show(rhs)                      \
              , extra_scope                                                             \
              )

#define MI_CHECK_NOT_EQUAL_MSG(lhs,rhs,extra_scope)                                     \
  MI_TEST_IMPL( MI::TEST::Test_case_failure                                             \
              , (lhs) != (rhs)                                                          \
              , std::string(#lhs) + " != " + std::string(#rhs)                      \
              , MI::TEST::show(lhs) + " == " + MI::TEST::show(rhs)                      \
              , extra_scope                                                             \
              )

#define MI_CHECK_LESS_MSG(lhs,rhs,extra_scope)                                          \
  MI_TEST_IMPL( MI::TEST::Test_case_failure                                             \
              , (lhs) < (rhs)                                                           \
              , std::string(#lhs) + " < " + std::string(#rhs)                       \
              , MI::TEST::show(lhs) + " >= " + MI::TEST::show(rhs)                      \
              , extra_scope                                                             \
              )

#define MI_CHECK_LESS_OR_EQUAL_MSG(lhs,rhs,extra_scope)                                 \
  MI_TEST_IMPL( MI::TEST::Test_case_failure                                             \
              , (lhs) <= (rhs)                                                          \
              , std::string(#lhs) + " <= " + std::string(#rhs)                      \
              , MI::TEST::show(lhs) + " > " + MI::TEST::show(rhs)                       \
              , extra_scope                                                             \
              )

#define MI_CHECK_GREATER_MSG(lhs,rhs,extra_scope)                                       \
  MI_TEST_IMPL( MI::TEST::Test_case_failure                                             \
              , (lhs) > (rhs)                                                           \
              , std::string(#lhs) + " > " + std::string(#rhs)                       \
              , MI::TEST::show(lhs) + " <= " + MI::TEST::show(rhs)                      \
              , extra_scope                                                             \
              )

#define MI_CHECK_GREATER_OR_EQUAL_MSG(lhs,rhs,extra_scope)                              \
  MI_TEST_IMPL( MI::TEST::Test_case_failure                                             \
              , (lhs) >= (rhs)                                                          \
              , std::string(#lhs) + " >= " + std::string(#rhs)                      \
              , MI::TEST::show(lhs) + " < " + MI::TEST::show(rhs)                       \
              , extra_scope                                                             \
              )

#define MI_CHECK_CLOSE_MSG(lhs,rhs,eps,extra_scope)                                     \
  MI_TEST_IMPL( MI::TEST::Test_case_failure                                             \
              , MI::TEST::abs_distance((lhs), (rhs)) < (eps)                            \
              , std::string("abs(") + #lhs + " - " + #rhs + ") < " #eps               \
              , std::string("abs(") + MI::TEST::show(lhs) + " - " + MI::TEST::show(rhs) + ") >= " + MI::TEST::show(eps) \
              , extra_scope                                                             \
              )

#define MI_CHECK_EQUAL_COLLECTIONS(lb,le,rb,re)                                         \
    MI::TEST::is_equal_collection<MI::TEST::Test_case_failure>                          \
      ( std::string("{ ") + #lb + ", " + #le + " } == { " + #rb + ", " + #re + " }"   \
      , MI_CURRENT_LOCATION                                                             \
      , (lb), (le), (rb), (re)                                                          \
      )

#define MI_CHECK_CLOSE_COLLECTIONS(lb,le,rb,re,eps)                                     \
    MI::TEST::is_close_collection<MI::TEST::Test_case_failure>                          \
      ( std::string("{ ") + #lb + ", " + #le + " } == { " + #rb + ", " + #re + " }"   \
      , MI_CURRENT_LOCATION                                                             \
      , (lb), (le), (rb), (re), (eps)                                                   \
      )

#define MI_CHECK_EQUAL_CSTR_MSG(lhs,rhs,extra_scope)                                    \
  MI_TEST_IMPL( MI::TEST::Test_case_failure                                             \
              , strcmp(lhs, rhs) == 0                                                   \
              , std::string(#lhs) + " == " + std::string(#rhs)                      \
              , '"' + MI::TEST::show(lhs) + "\" != \"" + MI::TEST::show(rhs) + '"'      \
              , extra_scope                                                             \
              )

#define MI_CHECK_NOT_EQUAL_CSTR_MSG(lhs,rhs,extra_scope)                                \
  MI_TEST_IMPL( MI::TEST::Test_case_failure                                             \
              , strcmp(lhs, rhs) != 0                                                   \
              , std::string(#lhs) + " != " + std::string(#rhs)                      \
              , '"' + MI::TEST::show(lhs) + "\" == \"" + MI::TEST::show(rhs) + '"'      \
              , extra_scope                                                             \
              )

#define MI_CHECK_ZERO_MSG(expr,extra_scope)                                             \
  MI_TEST_IMPL( MI::TEST::Test_case_failure                                             \
              , (expr) == 0                                                             \
              , std::string(#expr) + " == 0"                                            \
              , MI::TEST::show(expr) + " != 0"                                          \
              , extra_scope                                                             \
              )

#define MI_CHECK(expr)                     MI_CHECK_MSG(expr,"")
#define MI_CHECK_EQUAL(lhs,rhs)            MI_CHECK_EQUAL_MSG(lhs,rhs,"")
#define MI_CHECK_NOT_EQUAL(lhs,rhs)        MI_CHECK_NOT_EQUAL_MSG(lhs,rhs,"")
#define MI_CHECK_LESS(lhs,rhs)             MI_CHECK_LESS_MSG(lhs,rhs,"")
#define MI_CHECK_LESS_OR_EQUAL(lhs,rhs)    MI_CHECK_LESS_OR_EQUAL_MSG(lhs,rhs,"")
#define MI_CHECK_GREATER(lhs,rhs)          MI_CHECK_GREATER_MSG(lhs,rhs,"")
#define MI_CHECK_GREATER_OR_EQUAL(lhs,rhs) MI_CHECK_GREATER_OR_EQUAL_MSG(lhs,rhs,"")
#define MI_CHECK_CLOSE(lhs,rhs,eps)        MI_CHECK_CLOSE_MSG(lhs,rhs,eps,"")
#define MI_CHECK_EQUAL_CSTR(lhs,rhs)       MI_CHECK_EQUAL_CSTR_MSG(lhs,rhs,"")
#define MI_CHECK_NOT_EQUAL_CSTR(lhs,rhs)   MI_CHECK_NOT_EQUAL_CSTR_MSG(lhs,rhs,"")
#define MI_CHECK_ZERO(expr)                MI_CHECK_ZERO_MSG(expr,"")


//
// MI_REQUIRE Macros
//

#define MI_REQUIRE_MSG(expr,extra_scope)                                                \
  MI_TEST_IMPL( MI::TEST::Test_suite_failure                                            \
              , (expr)                                                                  \
              , #expr                                                                   \
              , ""                                                                      \
              , extra_scope                                                             \
              )

#define MI_REQUIRE_EQUAL_MSG(lhs,rhs,extra_scope)                                       \
  MI_TEST_IMPL( MI::TEST::Test_suite_failure                                            \
              , lhs == rhs                                                              \
              , std::string(#lhs) + " == " + std::string(#rhs)                      \
              , MI::TEST::show(lhs) + " != " + MI::TEST::show(rhs)                      \
              , extra_scope                                                             \
              )

#define MI_REQUIRE_NOT_EQUAL_MSG(lhs,rhs,extra_scope)                                   \
  MI_TEST_IMPL( MI::TEST::Test_suite_failure                                            \
              , lhs != rhs                                                              \
              , std::string(#lhs) + " != " + std::string(#rhs)                      \
              , MI::TEST::show(lhs) + " == " + MI::TEST::show(rhs)                      \
              , extra_scope                                                             \
              )

#define MI_REQUIRE_LESS_MSG(lhs,rhs,extra_scope)                                        \
  MI_TEST_IMPL( MI::TEST::Test_suite_failure                                            \
              , (lhs) < (rhs)                                                           \
              , std::string(#lhs) + " < " + std::string(#rhs)                       \
              , MI::TEST::show(lhs) + " >= " + MI::TEST::show(rhs)                      \
              , extra_scope                                                             \
              )

#define MI_REQUIRE_LESS_OR_EQUAL_MSG(lhs,rhs,extra_scope)                               \
  MI_TEST_IMPL( MI::TEST::Test_suite_failure                                            \
              , (lhs) <= (rhs)                                                          \
              , std::string(#lhs) + " <= " + std::string(#rhs)                      \
              , MI::TEST::show(lhs) + " > " + MI::TEST::show(rhs)                       \
              , extra_scope                                                             \
              )

#define MI_REQUIRE_GREATER_MSG(lhs,rhs,extra_scope)                                     \
  MI_TEST_IMPL( MI::TEST::Test_suite_failure                                            \
              , (lhs) > (rhs)                                                           \
              , std::string(#lhs) + " > " + std::string(#rhs)                       \
              , MI::TEST::show(lhs) + " <= " + MI::TEST::show(rhs)                      \
              , extra_scope                                                             \
              )

#define MI_REQUIRE_GREATER_OR_EQUAL_MSG(lhs,rhs,extra_scope)                            \
  MI_TEST_IMPL( MI::TEST::Test_suite_failure                                            \
              , (lhs) >= (rhs)                                                          \
              , std::string(#lhs) + " >= " + std::string(#rhs)                      \
              , MI::TEST::show(lhs) + " < " + MI::TEST::show(rhs)                       \
              , extra_scope                                                             \
              )

#define MI_REQUIRE_CLOSE_MSG(lhs,rhs,eps,extra_scope)                                   \
  MI_TEST_IMPL( MI::TEST::Test_suite_failure                                            \
              , MI::TEST::abs_distance((lhs), (rhs)) < (eps)                            \
              , std::string("abs(") + #lhs + " - " + #rhs + ") < " #eps               \
              , std::string("abs(") + MI::TEST::show(lhs) + " - " + MI::TEST::show(rhs) + ") >= " + MI::TEST::show(eps) \
              , extra_scope                                                             \
              )

#define MI_REQUIRE_EQUAL_COLLECTIONS(lb,le,rb,re)                                       \
    MI::TEST::is_equal_collection<MI::TEST::Test_suite_failure>                         \
      ( std::string("{ ") + #lb + ", " + #le + " } == { " + #rb + ", " + #re + " }"   \
      , MI_CURRENT_LOCATION                                                             \
      , (lb), (le), (rb), (re)                                                          \
      )

#define MI_REQUIRE_CLOSE_COLLECTIONS(lb,le,rb,re,eps)                                   \
    MI::TEST::is_close_collection<MI::TEST::Test_suite_failure>                         \
      ( std::string("{ ") + #lb + ", " + #le + " } == { " + #rb + ", " + #re + " }"   \
      , MI_CURRENT_LOCATION                                                             \
      , (lb), (le), (rb), (re), (eps)                                                   \
      )

#define MI_REQUIRE(expr)                        MI_REQUIRE_MSG(expr,"")
#define MI_REQUIRE_EQUAL(lhs,rhs)               MI_REQUIRE_EQUAL_MSG(lhs,rhs,"")
#define MI_REQUIRE_NOT_EQUAL(lhs,rhs)           MI_REQUIRE_NOT_EQUAL_MSG(lhs,rhs,"")
#define MI_REQUIRE_LESS(lhs,rhs)                MI_REQUIRE_LESS_MSG(lhs,rhs,"")
#define MI_REQUIRE_LESS_OR_EQUAL(lhs,rhs)       MI_REQUIRE_LESS_OR_EQUAL_MSG(lhs,rhs,"")
#define MI_REQUIRE_GREATER(lhs,rhs)             MI_REQUIRE_GREATER_MSG(lhs,rhs,"")
#define MI_REQUIRE_GREATER_OR_EQUAL(lhs,rhs)    MI_REQUIRE_GREATER_OR_EQUAL_MSG(lhs,rhs,"")
#define MI_REQUIRE_CLOSE(lhs,rhs,eps)           MI_REQUIRE_CLOSE_MSG(lhs,rhs,eps,"")

#endif // BASE_SYSTEM_TEST_CASE_H

