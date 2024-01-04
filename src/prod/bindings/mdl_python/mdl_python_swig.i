/******************************************************************************
 * Copyright 2023 NVIDIA Corporation. All rights reserved.
 *****************************************************************************/

/* File : example.i */
// 1- python instantiate_templates_inline.py example_swig.i "" ..\..\..\\include processed_headers processed_headers_dummy
// 2- swig -I./processed_headers -I..\..\..\\include -c++ -python -cppext cpp example_swig.i

%module pymdlsdk

%begin %{
#ifdef _MSC_VER
// use the 'Release' build of the Python interpreter with a 'Debug' build of the wrappers
// adding the `corecrt.h` here because problems when un-defining and re-defining '_DEBUG` done by SWIG
#include <corecrt.h>
#define SWIG_PYTHON_INTERPRETER_NO_DEBUG
#endif
%}

%include <typemaps.i>
%apply unsigned long long{ uint64_t };

// TODO: Doxygen
// Extrract aliases from Doxyfile automatically
// Figure out: %base, %math, $(MI_PRODUCT_VERSION)
// DOES NOT WORK  %import "../doc/mdl_sdkapi/Doxyfile"
%feature("doxygen:alias:BaseApiName") "Base API"
// %feature("doxygen:alias:baseApiName") "%base API"
%feature("doxygen:alias:baseApiName") "base API"
%feature("doxygen:alias:MathApiName") "Math API"
// %feature("doxygen:alias:mathApiName") "%math API"
%feature("doxygen:alias:mathApiName") "math API"
%feature("doxygen:alias:NeurayApiName") "MDL SDK API"
%feature("doxygen:alias:neurayApiName") "MDL SDK API"
%feature("doxygen:alias:NeurayProductName") "The MDL SDK"
%feature("doxygen:alias:neurayProductName") "the MDL SDK"
%feature("doxygen:alias:NeurayLibraryName") "MDL SDK library"
%feature("doxygen:alias:neurayLibraryName") "MDL SDK library"
%feature("doxygen:alias:NeurayAdjectiveName") "MDL SDK"
%feature("doxygen:alias:neurayAdjectiveName") "MDL SDK"
// %feature("doxygen:alias:MiProductVersion") "$(MI_PRODUCT_VERSION)"

// When enabled, the python binding will be able to load neuray or the MDL SDK.
// Otherwise, only existing neuray instance pointers can be attached from a running process.
#ifndef STANDALONE_SUPPORT_ENABLED
#define STANDALONE_SUPPORT_ENABLED 1
#endif

%{
/******************************************************************************
 * Copyright 2023 NVIDIA Corporation. All rights reserved.
 *****************************************************************************/

#ifdef IRAY_SDK
    #include "mi/neuraylib.h"
#else
    #include "mi/mdl_sdk.h"
#endif

#include "mdl_python.h"

namespace {
#if __cplusplus >= 201703L
    constexpr bool starts_with(const char* str, const char* prefix)
    {
        size_t n = std::char_traits<char>::length(str);
        size_t pn = std::char_traits<char>::length(prefix);
        return (pn > n) ? false : std::char_traits<char>::compare(str, prefix, pn) == 0;
    }
    constexpr bool ends_with(const char* str, const char* suffix)
    {
        size_t n = std::char_traits<char>::length(str);
        size_t sn = std::char_traits<char>::length(suffix);
        return (sn > n) ? false : std::char_traits<char>::compare(str + n - sn, suffix, sn) == 0;
    }
#else
    bool starts_with(const char* str, const char* prefix)
    {
        size_t n = strlen(str);
        size_t pn = strlen(prefix);
        return (pn > n) ? false : strncmp(str, prefix, pn) == 0;
    }
    bool ends_with(const char* str, const char* suffix)
    {
        size_t n = strlen(str);
        size_t sn = strlen(suffix);
        return (sn > n) ? false : strncmp(str + n - sn, suffix, sn) == 0;
    }
#endif
}

std::mutex SmartPtrBase::s_lock_open_handles;
std::map<void*, SmartPtrBase::Open_handle>SmartPtrBase::s_open_handles;
bool SmartPtrBase::s_print_ref_counts = false;

void _print_open_handle_statistic()
{
    SmartPtrBase::_print_open_handle_statistic();
}

void _enable_print_ref_counts(bool enabled)
{
    SmartPtrBase::_enable_print_ref_counts(enabled);
}

using mi::Uint32;
using mi::IData;
using mi::base::Uuid;
using mi::neuraylib::IFunction_definition;
%}

#if STANDALONE_SUPPORT_ENABLED
%{
extern mi::neuraylib::INeuray* load_and_get_ineuray(const char*);
extern bool load_plugin(mi::neuraylib::INeuray*, const char*);
extern bool unload();
%}
#endif

%pythoncode {
    from enum import Enum
    import warnings
    import gc
}

// this adds 'with` support to the smart pointer
%ignore SmartPtrBase;
%ignore SmartPtrBase::Open_handle;
%ignore SmartPtr::operator=; // not supported by swig
%import "mdl_python.h"
%extend SmartPtr
{
    // enter function for `with` blocks
    SmartPtr<T>* __enter__() {
        return $self;
    }

    // exit function for `with` blocks
    bool __exit__(const void* exc_type, const void* exc_value, const void* exc_traceback) {
        $self->drop(true);
        return true;
    }

    // release function exposed to python.
    // decrements the ref count internally and makes the python proxy invalid (holds nullptr).
    void release() {
        $self->drop(false);
    }

    // Reference counter of the native objects.
    size_t __iinterface_refs__() const {
        const T* pointee = $self->get();
        if (pointee) {
            pointee->retain();
            return pointee->release();
        }
        return 0;
    }

    // Get the native pointer as uint64. E.g. for comparison.
    uint64_t __iinterface_ptr_as_uint64__() const {
        const T* pointee = $self->get();
        if (pointee) {
            return reinterpret_cast<uint64_t>(pointee);
        }
        return 0;
    }
}

// To handle out-parameters in combination with context managers we make the out value an object
// that can be modified from the wrapper code and read afterwards similar to the C++c syntax.
%pythoncode {
    class ReturnCode() :
        value: int = 0
}

// Workaround for some issues with template arguments that contain 'enum' or 'struct' in their names
%{
namespace mi
{
    namespace neuraylib
    {
        using IType_enumeration = mi::neuraylib::IType_enum;
        using IType_structure = mi::neuraylib::IType_struct;
        using IValue_enumeration = mi::neuraylib::IValue_enum;
        using IValue_structure = mi::neuraylib::IValue_struct;
    }
}
%}

#define IType_enum IType_enumeration
#define IType_struct IType_structure
#define IValue_enum IValue_enumeration
#define IValue_struct IValue_structure

// Init step for the wrapper generation
%define NEURAY_INIT_INTERFACE(CLASS_TYPE)

    // We don't want to generate wrappers for our interfaces directly, only for the ones wrapped in handles
    //%ignore CLASS_TYPE;
    %nodefault CLASS_TYPE;

    // handle retain and release a bit different:
    // - we hide the internal ref countering from the python interface
    // - we add a new release function, that just like the __exit__,
    //   invalidates the python proxy using `drop` (which decrements the ref count internally)
    %ignore CLASS_TYPE::retain;
    %ignore CLASS_TYPE::release;

    // hide the get/retain/release/drop functions, this is not supposed to be used from python
    %ignore SmartPtr<CLASS_TYPE>::retain;
    // %ignore SmartPtr<CLASS_TYPE>::release; // we actually do expose this for manual releasing of objects (deprecated in the future?)
    %ignore SmartPtr<CLASS_TYPE>::drop;
    %ignore SmartPtr<CLASS_TYPE>::get;

    // hide debugging support
    %ignore SmartPtr<CLASS_TYPE>::get_debug_str;
    %ignore SmartPtr<CLASS_TYPE>::assign_open_handle_typename;

    // we don't want people to create objects using the constructor
    %nodefault SmartPtr<CLASS_TYPE>;
    %ignore SmartPtr<CLASS_TYPE>::SmartPtr<CLASS_TYPE>;

    // we re-implement get_interface here
    %ignore CLASS_TYPE::get_interface;

    %ignore CLASS_TYPE::s_kind;

    %rename(_get_iid) SmartPtr<CLASS_TYPE>::get_iid;
    %extend SmartPtr<CLASS_TYPE> {

        /// Declares the interface ID (IID) of this interface.
        ///
        /// A local type in each interface type, which is distinct and unique for each interface. The
        /// type has a default constructor and the constructed value represents the universally unique
        /// identifier (UUID) for this interface. The local type is readily convertible to a
        /// #mi::base::Uuid.
        static const mi::base::Uuid IID()
        {
            return CLASS_TYPE::IID();
        }

        /// this creates a static function at the target interface class to convert an interface pointer
        /// into an object of the own class, e.g.:
        /// [python] cfg = pymdlsdk.IMdl_configuration.get_interface(iinterface)
        static SmartPtr<CLASS_TYPE>* _get_interface(mi::base::IInterface* iface) {
            auto ptr = new SmartPtr<CLASS_TYPE>(iface->get_interface<CLASS_TYPE>(), "CLASS_TYPE");
            return ptr;
        }

        // Allow "foo.get_interface(Bar)" instead of "Bar.get_interface(foo)"
        // Note, this will return an invalid pointer if casting fails
        %pythoncode {
            def get_interface(self, type):
                r"""
                Acquires an interfaces of a given other ``type``.
                If the type hierarchy of the calling object does not contain the specified ``type``, 
                an invalid interface is returned.
                """
                typed_interface = type._get_interface(self)
                typed_interface.thisown = True
                return typed_interface
        }

        // %pythoncode {
        //     def __repr__(self) -> str:
        //         r"""Create a string representation."""
        //         refCount: int = self.__iinterface_refs__()
        //         return f"CLASS_TYPE (RefCount: {refCount})"
        // }
    };
%enddef

%define DICE_INTERFACE(CLASS_TYPE)
    NEURAY_INIT_INTERFACE(mi::neuraylib::CLASS_TYPE)
%enddef

%define DICE_INTERFACE_MI(CLASS_TYPE)
    NEURAY_INIT_INTERFACE(mi::CLASS_TYPE)
%enddef

%define DICE_INTERFACE_BASE(CLASS_TYPE)
    NEURAY_INIT_INTERFACE(mi::base::CLASS_TYPE)
%enddef

// Wrapping Neuray IInterfaces
%define NEURAY_DEFINE_HANDLE_TYPEMAP(IINTERFACE_TYPE, ARGS...)
    // Convert IInterface* return types to SmartPtr<IInterface>
    // %typemap(out) IINTERFACE_TYPE ARGS* {
    %typemap(out) IINTERFACE_TYPE ARGS*, IINTERFACE_TYPE ARGS&, IINTERFACE_TYPE ARGS&&, IINTERFACE_TYPE ARGS[] {
        auto ptr = new SmartPtr<IINTERFACE_TYPE>($1, "IINTERFACE_TYPE");
        %set_output(SWIG_NewPointerObj(ptr, $descriptor(SmartPtr<IINTERFACE_TYPE> ARGS*), SWIG_POINTER_OWN));
    }
    // TODO: needs to be tested when used (ask Joerg about this)
    %typemap(out) IINTERFACE_TYPE ARGS *const& {
        assert(false && "Untested typemap out!");
        auto ptr = new SmartPtr<IINTERFACE_TYPE>($1, "IINTERFACE_TYPE");
        %set_output(SWIG_NewPointerObj(ptr, $descriptor(SmartPtr<IINTERFACE_TYPE> ARGS*), SWIG_POINTER_OWN));
    }

    // Convert SmartPtr<IInterface> to IInterface* when passed as arguments
    %typemap(in) IINTERFACE_TYPE ARGS * {
        void* arg_ptr = 0;
        int res = SWIG_ConvertPtr($input, (void**)&arg_ptr, $descriptor(SmartPtr<IINTERFACE_TYPE> ARGS*), 0);
        mi::base::Handle<IINTERFACE_TYPE> tmp_$1_name;
        if (SWIG_IsOK(res))
        {
            auto sptr = reinterpret_cast<SmartPtr<IINTERFACE_TYPE> ARGS*>(arg_ptr);
            $1 = (sptr != nullptr && sptr->is_valid_interface()) ? sptr->get() : nullptr;
        }
        else
        {
            // get the base ptr for type infos
            res = SWIG_ConvertPtr($input, (void**)&arg_ptr, SWIG_TypeQuery("SmartPtrBase*"), 0);
            if (SWIG_IsOK(res))
            {
                if (!arg_ptr)
                {
                    $1 = nullptr; // usually valid
                }
                else
                {
                    // use the base pointer class to cast to the expected type
                    auto base_sptr = reinterpret_cast<SmartPtrBase ARGS*>(arg_ptr);
                    tmp_$1_name = base_sptr->get_iinterface<IINTERFACE_TYPE>();
                    $1 = tmp_$1_name.get();
                }
            }
            else {
                SWIG_exception_fail(SWIG_ArgError(res), "in method '" "$symname" "', argument " "$1_name"" of type '" "$1_type""'");
            }
        }
    }

    // Comes in pairs with the typemap(in). This is required to resolver type-mapped parameters in overloads
    %typemap(typecheck, precedence = SWIG_TYPECHECK_POINTER) IINTERFACE_TYPE ARGS* {
        void* arg_ptr = 0;
        int res = SWIG_ConvertPtr($input, (void**)&arg_ptr, SWIG_TypeQuery("SmartPtrBase*"), 0);
        if (!SWIG_IsOK(res))
        {
            $1 = 0;
            PyErr_Clear();
        }
        else
        {
            if (!arg_ptr)
            {
                $1 = 1; // usually valid
            }
            else
            {
                // use the base pointer class to cast to the expected type
                auto base_sptr = reinterpret_cast<SmartPtrBase ARGS*>(arg_ptr);
                mi::base::Handle<IINTERFACE_TYPE> tmp_$1_name(base_sptr->get_iinterface<IINTERFACE_TYPE>());
                $1 = tmp_$1_name ? 1 : 0;
            }
        }
    }

    %typemap(check) SmartPtr<IINTERFACE_TYPE ARGS> *self {
        // filter cases that are supposed to work with nullptrs
        %#if __cplusplus >= 201703L
            constexpr bool filtered = 
        %#else
            const bool filtered =
        %#endif
            ends_with("$symname", "is_valid_interface") ||  // when an returned IInterface is null, we get a nullptr in the SmartPtr, which is expected
            ends_with("$symname", "__enter__") ||           // when an returned IInterface is used in a `with .. as ..` statement
            ends_with("$symname", "__exit__") ||            // when an returned IInterface is used in a `with .. as ..` statement
            starts_with("$symname", "delete_");             // when exiting a `with .. as ..` statement, the refcount of the SmartPtr is already zero
        if (!$1->is_valid_interface() && !filtered)
        {
            // __debugbreak(); // can be used for debugging, don't enable in production because users could use these exceptions intentionally
            SWIG_exception_fail(SWIG_ArgError(SWIG_NullReferenceError), "called wrapped " "IINTERFACE_TYPE" " which is not a valid interface (None)");
        }
    }
    %typemap(check) IINTERFACE_TYPE ARGS *self {
        if (!$1) {
            // __debugbreak();
            PyErr_SetString(PyExc_RuntimeError, "called " "IINTERFACE_TYPE" " that is not a valid interface (None)");
            SWIG_fail;
        }
    }

%enddef

// Instantiate Handle template for Neuray IInterfaces
%define NEURAY_CREATE_HANDLE_TEMPLATE(ORG_NAMESPACE, IINTERFACE_TYPE)
    //#warning Instantiated: IINTERFACE_TYPE
    %template(IINTERFACE_TYPE) SmartPtr<ORG_NAMESPACE::IINTERFACE_TYPE>;
%enddef

// Adds a template-like casting helper as we have in the neuray headers for c++
%define EXTEND_FUNCTION_AS(IINTERFACE_TYPE, FUNCTION_NAME)
    %extend SmartPtr<IINTERFACE_TYPE> {
        %pythoncode{
            def FUNCTION_NAME ## _as(self, type, *args):
                r"""
                Calls FUNCTION_NAME and acquires an interface of ``type`` for the return value.
                """
                iinterface = self. ## FUNCTION_NAME ## (*args)
                if iinterface.is_valid_interface():
                    return iinterface.get_interface(type)
                else:
                    return iinterface
        }
    }
%enddef


// Wrap mi data types manually for the templated get_value function.
// Addionally, the data types are defined above.
%define WRAP_MI_NUMBER_DATA_TYPES(IINTERFACE_TYPE, DATA_TYPE)
    %extend SmartPtr<IINTERFACE_TYPE> {
        DATA_TYPE get_value() const
        {
            return $self->get()->get_value<DATA_TYPE>();
        }
        void set_value( DATA_TYPE value)
        {
            $self->get()->set_value(value);
        }
    }
%enddef
%define WRAP_MI_COMPOUND_DATA_TYPES(IINTERFACE_TYPE, DATA_TYPE)
    %extend SmartPtr<IINTERFACE_TYPE> {
        DATA_TYPE get_value() const
        {
            DATA_TYPE v;
            $self->get()->get_value(v);
            return v;
        }
        void set_value(DATA_TYPE value)
        {
            $self->get()->set_value(value);
        }
    }
%enddef
%define WRAP_MI_DATA_COLLECTION_TYPES(IINTERFACE_TYPE)
    %extend SmartPtr<IINTERFACE_TYPE> {
        const mi::base::IInterface* _get_value_by_name(const char* name) const
        {
            return $self->get()->get_value(name);
        }
        const mi::base::IInterface* _get_value_by_index(mi::Size index) const
        {
            return $self->get()->get_value(index);
        }
        %pythoncode {
            def get_value(self, index_or_name):
                if isinstance(index_or_name, str):
                    return self._get_value_by_name(index_or_name)
                if isinstance(index_or_name, int) :
                    return self._get_value_by_index(index_or_name)
                raise IndexError('Collections can only be addressed by name (str) or index (int).')

            def get_value_as(self, type, index_or_name):
                iinterface = self.get_value(index_or_name)
                if iinterface.is_valid_interface():
                    return iinterface.get_interface(type)
                else:
                    return iinterface
        }

        mi::Sint32 _set_value_by_name(const char* name, mi::base::IInterface* value)
        {
            return $self->get()->set_value(name, value);
        }
        mi::Sint32 _set_value_by_index(mi::Size index, mi::base::IInterface* value)
        {
            return $self->get()->set_value(index, value);
        }
        %pythoncode {
            def set_value(self, index_or_name, value):
                if isinstance(index_or_name, str):
                    return self._set_value_by_name(index_or_name, value)
                if isinstance(index_or_name, int):
                    return self._set_value_by_index(index_or_name, value)
                raise IndexError('Collections can only be addressed by name (str) or index (int).')
        }
    }
%enddef

namespace mi {

typedef signed   char      Sint8;   ///<  8-bit   signed integer.
typedef signed   short     Sint16;  ///< 16-bit   signed integer.
typedef signed   int       Sint32;  ///< 32-bit   signed integer.
typedef unsigned char      Uint8;   ///<  8-bit unsigned integer.
typedef unsigned short     Uint16;  ///< 16-bit unsigned integer.
typedef unsigned int       Uint32;  ///< 32-bit unsigned integer.

#if defined(MI_COMPILER_MSC)
typedef signed __int64     Sint64;  ///< 64-bit   signed integer.
typedef unsigned __int64   Uint64;  ///< 64-bit unsigned integer.
#elif defined(MI_COMPILER_GCC)
typedef long long          Sint64;  ///< 64-bit   signed integer.
typedef unsigned long long Uint64;  ///< 64-bit unsigned integer.
#else
typedef signed   long long Sint64;  ///< 64-bit   signed integer.
typedef unsigned long long Uint64;  ///< 64-bit unsigned integer.
#endif

typedef float              Float32; ///< 32-bit float.
typedef double             Float64; ///< 64-bit float.

// Define Size to be signed, otherwise Java would use BigInteger
#ifdef MI_ARCH_64BIT
typedef Sint64             Size;
typedef Sint64             Difference;
#else // MI_ARCH_64BIT
typedef Sint32             Size;
typedef Sint32             Difference;
#endif // MI_ARCH_64BIT

} // namespace

%extend mi::base::Uuid {
    %pythoncode {
        def __eq__(self, rhs):
            return self.m_id1 == rhs.m_id1 and self.m_id2 == rhs.m_id2 and self.m_id3 == rhs.m_id3 and self.m_id4 == rhs.m_id4
        def __str__(self):
            return '-'.join('%02x' % i for i in {self.m_id1, self.m_id2, self.m_id3, self.m_id4})
    }
}


// special handling for: mi::neuraylib::IDatabase::Garbage_collection_priority
// ----------------------------------------------------------------------------

// We manually define the enums
%pythoncode{
    class Clip_mode(Enum) :
        r"""
        Supported clipping modes
        See #mi::Color::clip() function.
        """

        CLIP_RGB = _pymdlsdk.CLIP_RGB
        r""" First clip RGB to [0,1], then clip A to [max(R,G,B),1]."""
        CLIP_ALPHA = _pymdlsdk.CLIP_ALPHA
        r""" First clip A to [0,1], then clip RGB to [0,A]."""
        CLIP_RAW = _pymdlsdk.CLIP_RAW
        r""" Clip RGB and A to [0,1]."""
}

// unwrap all usages of the enum values in arguments
%feature("pythonprepend") mi::Color::clip%{
    mode = mode.value  # unwrap python enum and pass the integer value
%}

namespace mi {
namespace base {
    struct Uuid
    {
        Uint32 m_id1; ///< First  value.
        Uint32 m_id2; ///< Second value.
        Uint32 m_id3; ///< Third  value.
        Uint32 m_id4; ///< Fourth value.
    };
} // base

namespace math {
    struct Color_struct { Float32 r, g, b, a; };
    enum Clip_mode {
        CLIP_RGB,   ///< First clip RGB to [0,1], then clip A to [max(R,G,B),1].
        CLIP_ALPHA, ///< First clip A to [0,1], then clip RGB to [0,A].
        CLIP_RAW    ///< Clip RGB and A to [0,1].
    };
} // math

    struct Boolean_2_struct { bool x, y; };
    struct Boolean_3_struct { bool x, y, z; };
    struct Boolean_4_struct { bool x, y, z, w; };

    struct Sint32_2_struct { Sint32 x, y; };
    struct Sint32_3_struct { Sint32 x, y, z; };
    struct Sint32_4_struct { Sint32 x, y, z, w; };

    struct Uint32_2_struct { Uint32 x, y; };
    struct Uint32_3_struct { Uint32 x, y, z; };
    struct Uint32_4_struct { Uint32 x, y, z, w; };

    struct Float32_2_struct { Float32 x, y; };
    struct Float32_3_struct { Float32 x, y, z; };
    struct Float32_4_struct { Float32 x, y, z, w; };

    struct Float64_2_struct { Float64 x, y; };
    struct Float64_3_struct { Float64 x, y, z; };
    struct Float64_4_struct { Float64 x, y, z, w; };

    struct Boolean_2_2_struct { bool xx, xy, yx, yy; };
    struct Boolean_2_3_struct { bool xx, xy, xz, yx, yy, yz; };
    struct Boolean_2_4_struct { bool xx, xy, xz, xw, yx, yy, yz, yw; };
    struct Boolean_3_2_struct { bool xx, xy, yx, yy, zx ,zy; };
    struct Boolean_3_3_struct { bool xx, xy, xz, yx, yy, yz, zx ,zy, zz; };
    struct Boolean_3_4_struct { bool xx, xy, xz, xw, yx, yy, yz, yw, zx, zy, zz, zw; };
    struct Boolean_4_2_struct { bool xx, xy, yx, yy, zx ,zy, wx, wy; };
    struct Boolean_4_3_struct { bool xx, xy, xz, yx, yy, yz, zx ,zy, zz, wx, wy, wz; };
    struct Boolean_4_4_struct { bool xx, xy, xz, xw, yx, yy, yz, yw, zx ,zy, zz, zw, wx, wy, wz, ww; };

    struct Sint32_2_2_struct { Sint32 xx, xy, yx, yy; };
    struct Sint32_2_3_struct { Sint32 xx, xy, xz, yx, yy, yz; };
    struct Sint32_2_4_struct { Sint32 xx, xy, xz, xw, yx, yy, yz, yw; };
    struct Sint32_3_2_struct { Sint32 xx, xy, yx, yy, zx, zy; };
    struct Sint32_3_3_struct { Sint32 xx, xy, xz, yx, yy, yz, zx, zy, zz; };
    struct Sint32_3_4_struct { Sint32 xx, xy, xz, xw, yx, yy, yz, yw, zx, zy, zz, zw; };
    struct Sint32_4_2_struct { Sint32 xx, xy, yx, yy, zx, zy, wx, wy; };
    struct Sint32_4_3_struct { Sint32 xx, xy, xz, yx, yy, yz, zx, zy, zz, wx, wy, wz; };
    struct Sint32_4_4_struct { Sint32 xx, xy, xz, xw, yx, yy, yz, yw, zx, zy, zz, zw, wx, wy, wz, ww; };

    struct Uint32_2_2_struct { Uint32 xx, xy, yx, yy; };
    struct Uint32_2_3_struct { Uint32 xx, xy, xz, yx, yy, yz; };
    struct Uint32_2_4_struct { Uint32 xx, xy, xz, xw, yx, yy, yz, yw; };
    struct Uint32_3_2_struct { Uint32 xx, xy, yx, yy, zx, zy; };
    struct Uint32_3_3_struct { Uint32 xx, xy, xz, yx, yy, yz, zx, zy, zz; };
    struct Uint32_3_4_struct { Uint32 xx, xy, xz, xw, yx, yy, yz, yw, zx, zy, zz, zw; };
    struct Uint32_4_2_struct { Uint32 xx, xy, yx, yy, zx, zy, wx, wy; };
    struct Uint32_4_3_struct { Uint32 xx, xy, xz, yx, yy, yz, zx, zy, zz, wx, wy, wz; };
    struct Uint32_4_4_struct { Uint32 xx, xy, xz, xw, yx, yy, yz, yw, zx, zy, zz, zw, wx, wy, wz, ww; };

    struct Float32_2_2_struct { Float32 xx, xy, yx, yy; };
    struct Float32_2_3_struct { Float32 xx, xy, xz, yx, yy, yz; };
    struct Float32_2_4_struct { Float32 xx, xy, xz, xw, yx, yy, yz, yw; };
    struct Float32_3_2_struct { Float32 xx, xy, yx, yy, zx, zy; };
    struct Float32_3_3_struct { Float32 xx, xy, xz, yx, yy, yz, zx, zy, zz; };
    struct Float32_3_4_struct { Float32 xx, xy, xz, xw, yx, yy, yz, yw, zx, zy, zz, zw; };
    struct Float32_4_2_struct { Float32 xx, xy, yx, yy, zx, zy, wx, wy; };
    struct Float32_4_3_struct { Float32 xx, xy, xz, yx, yy, yz, zx, zy, zz, wx, wy, wz; };
    struct Float32_4_4_struct { Float32 xx, xy, xz, xw, yx, yy, yz, yw, zx, zy, zz, zw, wx, wy, wz, ww; };

    struct Float64_2_2_struct { Float64 xx, xy, yx, yy; };
    struct Float64_2_3_struct { Float64 xx, xy, xz, yx, yy, yz; };
    struct Float64_2_4_struct { Float64 xx, xy, xz, xw, yx, yy, yz, yw; };
    struct Float64_3_2_struct { Float64 xx, xy, yx, yy, zx, zy; };
    struct Float64_3_3_struct { Float64 xx, xy, xz, yx, yy, yz, zx, zy, zz; };
    struct Float64_3_4_struct { Float64 xx, xy, xz, xw, yx, yy, yz, yw, zx, zy, zz, zw; };
    struct Float64_4_2_struct { Float64 xx, xy, yx, yy, zx, zy, wx, wy; };
    struct Float64_4_3_struct { Float64 xx, xy, xz, yx, yy, yz, zx, zy, zz, wx, wy, wz; };
    struct Float64_4_4_struct { Float64 xx, xy, xz, xw, yx, yy, yz, yw, zx, zy, zz, zw, wx, wy, wz, ww; };
} // mi

#define mi_static_assert(ignored)


// ----------------------------------------------------------------------------
// mi::base
// ----------------------------------------------------------------------------

// Do not run the dice script on the base interface
%ignore mi::base::IInterface;
NEURAY_INIT_INTERFACE(mi::base::IInterface);
%include "mi/base/iinterface.h"
%include "mi/base/interface_declare.h"
%include "mi/base/enums.h"


NEURAY_DEFINE_HANDLE_TYPEMAP(mi::base::IInterface)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::base, IInterface)

// ----------------------------------------------------------------------------
// mi
// ----------------------------------------------------------------------------

DICE_INTERFACE_MI(IArray);
DICE_INTERFACE_MI(IMap);
DICE_INTERFACE_MI(ICompound);
DICE_INTERFACE_MI(IBoolean);
DICE_INTERFACE_MI(IBoolean_2);
DICE_INTERFACE_MI(IBoolean_3);
DICE_INTERFACE_MI(IBoolean_4);
DICE_INTERFACE_MI(ISint32_2);
DICE_INTERFACE_MI(ISint32_3);
DICE_INTERFACE_MI(ISint32_4);
DICE_INTERFACE_MI(IUint32_2);
DICE_INTERFACE_MI(IUint32_3);
DICE_INTERFACE_MI(IUint32_4);
DICE_INTERFACE_MI(IFloat32_2);
DICE_INTERFACE_MI(IFloat32_3);
DICE_INTERFACE_MI(IFloat32_4);
DICE_INTERFACE_MI(IFloat64_2);
DICE_INTERFACE_MI(IFloat64_3);
DICE_INTERFACE_MI(IFloat64_4);
DICE_INTERFACE_MI(IDynamic_array);
DICE_INTERFACE_MI(IDifference);
DICE_INTERFACE_MI(IString);
DICE_INTERFACE_MI(IStructure);
DICE_INTERFACE_MI(IData);
DICE_INTERFACE_MI(IData_simple);
DICE_INTERFACE_MI(IData_collection);
DICE_INTERFACE_MI(IFloat32);
DICE_INTERFACE_MI(IColor);
DICE_INTERFACE_MI(IColor3);
DICE_INTERFACE_MI(IFloat64);
DICE_INTERFACE_MI(INumber);
DICE_INTERFACE_MI(ISint8);
DICE_INTERFACE_MI(ISint16);
DICE_INTERFACE_MI(ISint32);
DICE_INTERFACE_MI(ISint64);
DICE_INTERFACE_MI(ISize);
DICE_INTERFACE_MI(IUint8);
DICE_INTERFACE_MI(IUint16);
DICE_INTERFACE_MI(IUint32);
DICE_INTERFACE_MI(IUint64);
DICE_INTERFACE_MI(IVoid);
DICE_INTERFACE_MI(IEnum_decl);
DICE_INTERFACE_MI(IEnum);
DICE_INTERFACE_MI(IBoolean_2_2)
DICE_INTERFACE_MI(IBoolean_2_3)
DICE_INTERFACE_MI(IBoolean_2_4)
DICE_INTERFACE_MI(IBoolean_3_2)
DICE_INTERFACE_MI(IBoolean_3_3)
DICE_INTERFACE_MI(IBoolean_3_4)
DICE_INTERFACE_MI(IBoolean_4_2)
DICE_INTERFACE_MI(IBoolean_4_3)
DICE_INTERFACE_MI(IBoolean_4_4)
DICE_INTERFACE_MI(ISint32_2_2)
DICE_INTERFACE_MI(ISint32_2_3)
DICE_INTERFACE_MI(ISint32_2_4)
DICE_INTERFACE_MI(ISint32_3_2)
DICE_INTERFACE_MI(ISint32_3_3)
DICE_INTERFACE_MI(ISint32_3_4)
DICE_INTERFACE_MI(ISint32_4_2)
DICE_INTERFACE_MI(ISint32_4_3)
DICE_INTERFACE_MI(ISint32_4_4)
DICE_INTERFACE_MI(IUint32_2_2)
DICE_INTERFACE_MI(IUint32_2_3)
DICE_INTERFACE_MI(IUint32_2_4)
DICE_INTERFACE_MI(IUint32_3_2)
DICE_INTERFACE_MI(IUint32_3_3)
DICE_INTERFACE_MI(IUint32_3_4)
DICE_INTERFACE_MI(IUint32_4_2)
DICE_INTERFACE_MI(IUint32_4_3)
DICE_INTERFACE_MI(IUint32_4_4)
DICE_INTERFACE_MI(IFloat32_2_2)
DICE_INTERFACE_MI(IFloat32_2_3)
DICE_INTERFACE_MI(IFloat32_2_4)
DICE_INTERFACE_MI(IFloat32_3_2)
DICE_INTERFACE_MI(IFloat32_3_3)
DICE_INTERFACE_MI(IFloat32_3_4)
DICE_INTERFACE_MI(IFloat32_4_2)
DICE_INTERFACE_MI(IFloat32_4_3)
DICE_INTERFACE_MI(IFloat32_4_4)
DICE_INTERFACE_MI(IFloat64_2_2)
DICE_INTERFACE_MI(IFloat64_2_3)
DICE_INTERFACE_MI(IFloat64_2_4)
DICE_INTERFACE_MI(IFloat64_3_2)
DICE_INTERFACE_MI(IFloat64_3_3)
DICE_INTERFACE_MI(IFloat64_3_4)
DICE_INTERFACE_MI(IFloat64_4_2)
DICE_INTERFACE_MI(IFloat64_4_3)
DICE_INTERFACE_MI(IFloat64_4_4)
DICE_INTERFACE_MI(IStructure_decl)

%ignore mi::INumber::get_value;
%ignore mi::INumber::set_value;
%ignore mi::ICompound::get_value;
%ignore mi::ICompound::set_value;
%ignore mi::IData_collection::get_value;
%ignore mi::IData_collection::set_value;
%ignore mi::ICompound::get_values;
%ignore mi::ICompound::set_values;

WRAP_MI_NUMBER_DATA_TYPES(mi::IBoolean, bool);

WRAP_MI_NUMBER_DATA_TYPES(mi::ISint8, mi::Sint8);
WRAP_MI_NUMBER_DATA_TYPES(mi::ISint16, mi::Sint16);
WRAP_MI_NUMBER_DATA_TYPES(mi::ISint32, mi::Sint32);
WRAP_MI_NUMBER_DATA_TYPES(mi::ISint64, mi::Sint64);

WRAP_MI_NUMBER_DATA_TYPES(mi::IUint8, mi::Uint8);
WRAP_MI_NUMBER_DATA_TYPES(mi::IUint16, mi::Uint16);
WRAP_MI_NUMBER_DATA_TYPES(mi::IUint32, mi::Uint32);
WRAP_MI_NUMBER_DATA_TYPES(mi::IUint64, mi::Uint64);
WRAP_MI_NUMBER_DATA_TYPES(mi::ISize, mi::Size);
WRAP_MI_NUMBER_DATA_TYPES(mi::IDifference, mi::Difference);

WRAP_MI_NUMBER_DATA_TYPES(mi::IFloat32, mi::Float32);
WRAP_MI_NUMBER_DATA_TYPES(mi::IFloat64, mi::Float64);

WRAP_MI_COMPOUND_DATA_TYPES(mi::IBoolean_2, mi::Boolean_2_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::IBoolean_3, mi::Boolean_3_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::IBoolean_4, mi::Boolean_4_struct);

WRAP_MI_COMPOUND_DATA_TYPES(mi::ISint32_2, mi::Sint32_2_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::ISint32_3, mi::Sint32_3_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::ISint32_4, mi::Sint32_4_struct);

WRAP_MI_COMPOUND_DATA_TYPES(mi::IUint32_2, mi::Uint32_2_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::IUint32_3, mi::Uint32_3_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::IUint32_4, mi::Uint32_4_struct);

WRAP_MI_COMPOUND_DATA_TYPES(mi::IFloat32_2, mi::Float32_2_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::IFloat32_3, mi::Float32_3_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::IFloat32_4, mi::Float32_4_struct);

WRAP_MI_COMPOUND_DATA_TYPES(mi::IFloat64_2, mi::Float64_2_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::IFloat64_3, mi::Float64_3_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::IFloat64_4, mi::Float64_4_struct);

WRAP_MI_COMPOUND_DATA_TYPES(mi::IBoolean_2_2, mi::Boolean_2_2_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::IBoolean_2_3, mi::Boolean_2_3_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::IBoolean_2_4, mi::Boolean_2_4_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::IBoolean_3_2, mi::Boolean_3_2_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::IBoolean_3_3, mi::Boolean_3_3_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::IBoolean_3_4, mi::Boolean_3_4_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::IBoolean_4_2, mi::Boolean_4_2_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::IBoolean_4_3, mi::Boolean_4_3_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::IBoolean_4_4, mi::Boolean_4_4_struct);

WRAP_MI_COMPOUND_DATA_TYPES(mi::ISint32_2_2, mi::Sint32_2_2_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::ISint32_2_3, mi::Sint32_2_3_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::ISint32_2_4, mi::Sint32_2_4_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::ISint32_3_2, mi::Sint32_3_2_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::ISint32_3_3, mi::Sint32_3_3_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::ISint32_3_4, mi::Sint32_3_4_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::ISint32_4_2, mi::Sint32_4_2_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::ISint32_4_3, mi::Sint32_4_3_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::ISint32_4_4, mi::Sint32_4_4_struct);

WRAP_MI_COMPOUND_DATA_TYPES(mi::IUint32_2_2, mi::Uint32_2_2_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::IUint32_2_3, mi::Uint32_2_3_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::IUint32_2_4, mi::Uint32_2_4_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::IUint32_3_2, mi::Uint32_3_2_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::IUint32_3_3, mi::Uint32_3_3_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::IUint32_3_4, mi::Uint32_3_4_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::IUint32_4_2, mi::Uint32_4_2_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::IUint32_4_3, mi::Uint32_4_3_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::IUint32_4_4, mi::Uint32_4_4_struct);

WRAP_MI_COMPOUND_DATA_TYPES(mi::IFloat32_2_2, mi::Float32_2_2_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::IFloat32_2_3, mi::Float32_2_3_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::IFloat32_2_4, mi::Float32_2_4_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::IFloat32_3_2, mi::Float32_3_2_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::IFloat32_3_3, mi::Float32_3_3_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::IFloat32_3_4, mi::Float32_3_4_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::IFloat32_4_2, mi::Float32_4_2_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::IFloat32_4_3, mi::Float32_4_3_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::IFloat32_4_4, mi::Float32_4_4_struct);

WRAP_MI_COMPOUND_DATA_TYPES(mi::IFloat64_2_2, mi::Float64_2_2_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::IFloat64_2_3, mi::Float64_2_3_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::IFloat64_2_4, mi::Float64_2_4_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::IFloat64_3_2, mi::Float64_3_2_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::IFloat64_3_3, mi::Float64_3_3_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::IFloat64_3_4, mi::Float64_3_4_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::IFloat64_4_2, mi::Float64_4_2_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::IFloat64_4_3, mi::Float64_4_3_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::IFloat64_4_4, mi::Float64_4_4_struct);

WRAP_MI_COMPOUND_DATA_TYPES(mi::IColor, mi::Color_struct);
WRAP_MI_COMPOUND_DATA_TYPES(mi::IColor3, mi::Color_struct);


WRAP_MI_DATA_COLLECTION_TYPES(mi::IData_collection)
WRAP_MI_DATA_COLLECTION_TYPES(mi::IArray)
WRAP_MI_DATA_COLLECTION_TYPES(mi::IDynamic_array)
WRAP_MI_DATA_COLLECTION_TYPES(mi::IMap)
WRAP_MI_DATA_COLLECTION_TYPES(mi::IStructure)

EXTEND_FUNCTION_AS(mi::IArray, get_element)
EXTEND_FUNCTION_AS(mi::IDynamic_array, get_element)
EXTEND_FUNCTION_AS(mi::IDynamic_array, front)
EXTEND_FUNCTION_AS(mi::IDynamic_array, back)

%include "mi/neuraylib/vector_typedefs.h"
%include "mi/neuraylib/typedefs.h"
%include "mi/neuraylib/idata.h"
%include "mi/neuraylib/istring.h"
%include "mi/neuraylib/istructure.h"
%include "mi/neuraylib/istructure_decl.h"
%include "mi/neuraylib/iarray.h"
%include "mi/neuraylib/idynamic_array.h"
%include "mi/neuraylib/inumber.h"
%include "mi/neuraylib/icompound.h"
%include "mi/neuraylib/icolor.h"
%include "mi/neuraylib/ivector.h"
%include "mi/neuraylib/imatrix.h"
%include "mi/neuraylib/imap.h"

NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IArray)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IMap)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::ICompound);
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IBoolean)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IBoolean_2)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IBoolean_3)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IBoolean_4)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::ISint32_2)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::ISint32_3)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::ISint32_4)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IUint32_2)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IUint32_3)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IUint32_4)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IFloat32_2)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IFloat32_3)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IFloat32_4)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IFloat64_2)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IFloat64_3)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IFloat64_4)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IDifference)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IDynamic_array)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IString)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IStructure)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IStructure_decl)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IData)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IData_simple)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IData_collection)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IFloat32)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IColor)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IColor3)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IFloat64)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::INumber)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::ISint8)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::ISint16)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::ISint32)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::ISint64)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::ISize)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IUint8)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IUint16)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IUint32)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IUint64)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IVoid)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IEnum_decl)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IEnum)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IBoolean_2_2)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IBoolean_2_3)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IBoolean_2_4)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IBoolean_3_2)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IBoolean_3_3)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IBoolean_3_4)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IBoolean_4_2)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IBoolean_4_3)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IBoolean_4_4)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::ISint32_2_2)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::ISint32_2_3)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::ISint32_2_4)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::ISint32_3_2)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::ISint32_3_3)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::ISint32_3_4)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::ISint32_4_2)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::ISint32_4_3)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::ISint32_4_4)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IUint32_2_2)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IUint32_2_3)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IUint32_2_4)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IUint32_3_2)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IUint32_3_3)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IUint32_3_4)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IUint32_4_2)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IUint32_4_3)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IUint32_4_4)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IFloat32_2_2)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IFloat32_2_3)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IFloat32_2_4)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IFloat32_3_2)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IFloat32_3_3)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IFloat32_3_4)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IFloat32_4_2)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IFloat32_4_3)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IFloat32_4_4)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IFloat64_2_2)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IFloat64_2_3)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IFloat64_2_4)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IFloat64_3_2)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IFloat64_3_3)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IFloat64_3_4)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IFloat64_4_2)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IFloat64_4_3)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IFloat64_4_4)



NEURAY_CREATE_HANDLE_TEMPLATE(mi, IArray)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IMap)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, ICompound)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IBoolean)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IBoolean_2)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IBoolean_3)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IBoolean_4)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, ISint32_2)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, ISint32_3)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, ISint32_4)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IUint32_2)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IUint32_3)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IUint32_4)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IFloat32_2)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IFloat32_3)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IFloat32_4)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IFloat64_2)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IFloat64_3)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IFloat64_4)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IDifference)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IDynamic_array)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IString)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IStructure)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IStructure_decl)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IData)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IData_simple)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IData_collection)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IFloat32)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IColor)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IColor3)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IFloat64)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, INumber)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, ISint8)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, ISint16)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, ISint32)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, ISint64)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, ISize)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IUint8)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IUint16)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IUint32)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IUint64)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IVoid)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IEnum_decl)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IEnum)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IBoolean_2_2)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IBoolean_2_3)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IBoolean_2_4)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IBoolean_3_2)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IBoolean_3_3)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IBoolean_3_4)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IBoolean_4_2)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IBoolean_4_3)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IBoolean_4_4)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, ISint32_2_2)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, ISint32_2_3)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, ISint32_2_4)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, ISint32_3_2)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, ISint32_3_3)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, ISint32_3_4)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, ISint32_4_2)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, ISint32_4_3)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, ISint32_4_4)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IUint32_2_2)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IUint32_2_3)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IUint32_2_4)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IUint32_3_2)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IUint32_3_3)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IUint32_3_4)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IUint32_4_2)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IUint32_4_3)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IUint32_4_4)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IFloat32_2_2)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IFloat32_2_3)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IFloat32_2_4)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IFloat32_3_2)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IFloat32_3_3)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IFloat32_3_4)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IFloat32_4_2)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IFloat32_4_3)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IFloat32_4_4)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IFloat64_2_2)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IFloat64_2_3)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IFloat64_2_4)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IFloat64_3_2)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IFloat64_3_3)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IFloat64_3_4)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IFloat64_4_2)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IFloat64_4_3)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IFloat64_4_4)


// ----------------------------------------------------------------------------
// mi::neuray
// ----------------------------------------------------------------------------
// one note on the order here, currently trying to keep the blocks and in each
// block we have alphabetical sorting

DICE_INTERFACE(IAttribute_set);
DICE_INTERFACE(IBaker)
DICE_INTERFACE(IBsdf_measurement)
DICE_INTERFACE(ICanvas)
DICE_INTERFACE(ICanvas_base)
DICE_INTERFACE(ICompiled_material)
DICE_INTERFACE(IDatabase);
DICE_INTERFACE(IDeserialized_function_name);
DICE_INTERFACE(IDeserialized_module_name);
DICE_INTERFACE(IImage_api);
DICE_INTERFACE(IMdl_configuration);
DICE_INTERFACE(IMdl_evaluator_api);
DICE_INTERFACE(IMessage);
DICE_INTERFACE(IMdl_distiller_api);
DICE_INTERFACE(IMdl_execution_context);
DICE_INTERFACE(IMdl_resolved_module);
DICE_INTERFACE(IMdl_resolved_resource);
DICE_INTERFACE(IMdl_resolved_resource_element);
DICE_INTERFACE(IMdl_entity_resolver);
DICE_INTERFACE(IMdl_factory);
DICE_INTERFACE(IMdl_impexp_api);
DICE_INTERFACE(IMdl_module_builder)
DICE_INTERFACE(IMdle_deserialization_callback)
DICE_INTERFACE(IMdle_serialization_callback)
DICE_INTERFACE(INeuray);
DICE_INTERFACE(IPlugin_configuration)
DICE_INTERFACE(IScene_element);
DICE_INTERFACE(IScope);
DICE_INTERFACE(ISerialized_function_name);
DICE_INTERFACE(ITransaction);
DICE_INTERFACE(IAnnotation)
DICE_INTERFACE(IAnnotation_block)
DICE_INTERFACE(IAnnotation_definition)
DICE_INTERFACE(IAnnotation_list)
DICE_INTERFACE(IExpression)
DICE_INTERFACE(IExpression_call)
DICE_INTERFACE(IExpression_constant)
DICE_INTERFACE(IExpression_direct_call)
DICE_INTERFACE(IExpression_factory)
DICE_INTERFACE(IExpression_list)
DICE_INTERFACE(IExpression_parameter)
DICE_INTERFACE(IExpression_temporary)
DICE_INTERFACE(IFunction_call)
DICE_INTERFACE(IFunction_definition)
DICE_INTERFACE(IImage)
DICE_INTERFACE(ILightprofile)
DICE_INTERFACE(IMaterial_instance)
DICE_INTERFACE(IModule)
DICE_INTERFACE(IStream_position)
DICE_INTERFACE(IReader_writer_base)
DICE_INTERFACE(IReader)
DICE_INTERFACE(IWriter)
DICE_INTERFACE(ITexture)
DICE_INTERFACE(ITile)
DICE_INTERFACE(IType)
DICE_INTERFACE(IType_alias)
DICE_INTERFACE(IType_array)
DICE_INTERFACE(IType_atomic)
DICE_INTERFACE(IType_bool)
DICE_INTERFACE(IType_bsdf)
DICE_INTERFACE(IType_bsdf_measurement)
DICE_INTERFACE(IType_color)
DICE_INTERFACE(IType_compound)
DICE_INTERFACE(IType_df)
DICE_INTERFACE(IType_double)
DICE_INTERFACE(IType_edf)
DICE_INTERFACE(IType_enum)
DICE_INTERFACE(IType_factory)
DICE_INTERFACE(IType_float)
DICE_INTERFACE(IType_hair_bsdf)
DICE_INTERFACE(IType_int)
DICE_INTERFACE(IType_light_profile)
DICE_INTERFACE(IType_list)
DICE_INTERFACE(IType_matrix)
DICE_INTERFACE(IType_reference)
DICE_INTERFACE(IType_resource)
DICE_INTERFACE(IType_string)
DICE_INTERFACE(IType_struct)
DICE_INTERFACE(IType_texture)
DICE_INTERFACE(IType_vdf)
DICE_INTERFACE(IType_vector)
DICE_INTERFACE(IValue)
DICE_INTERFACE(IValue_array)
DICE_INTERFACE(IValue_atomic)
DICE_INTERFACE(IValue_bool)
DICE_INTERFACE(IValue_bsdf_measurement)
DICE_INTERFACE(IValue_color)
DICE_INTERFACE(IValue_compound)
DICE_INTERFACE(IValue_double)
DICE_INTERFACE(IValue_enum)
DICE_INTERFACE(IValue_factory)
DICE_INTERFACE(IValue_float)
DICE_INTERFACE(IValue_int)
DICE_INTERFACE(IValue_invalid_df)
DICE_INTERFACE(IValue_light_profile)
DICE_INTERFACE(IValue_list)
DICE_INTERFACE(IValue_matrix)
DICE_INTERFACE(IValue_resource)
DICE_INTERFACE(IValue_string)
DICE_INTERFACE(IValue_string_localized)
DICE_INTERFACE(IValue_struct)
DICE_INTERFACE(IValue_texture)
DICE_INTERFACE(IValue_vector)


// Enums are handled manually because otherwise they would just end up as constants in the global scope
// Make sure this include is at this position, right after DICE_INTERFACE and in front of the other
// specializations but definitely before the template instantiation.
%include <mi_neuraylib_enums.i>


// special handling for: mi::neuraylib::IAttribute_set
// ----------------------------------------------------------------------------
// Since this is not used by the MDL SDK, we drop this from the bindings.

%ignore mi::neuraylib::IAttribute_set::create_attribute;
%ignore mi::neuraylib::IAttribute_set::destroy_attribute;
%ignore mi::neuraylib::IAttribute_set::access_attribute;
%ignore mi::neuraylib::IAttribute_set::edit_attribute;
%ignore mi::neuraylib::IAttribute_set::is_attribute;
%ignore mi::neuraylib::IAttribute_set::get_attribute_type_name;
%ignore mi::neuraylib::IAttribute_set::set_attribute_propagation;
%ignore mi::neuraylib::IAttribute_set::get_attribute_propagation;
%ignore mi::neuraylib::IAttribute_set::enumerate_attributes;


// special handling for: mi::neuraylib::INeuray
// ----------------------------------------------------------------------------
// Rewrite of special template functions to make life easier
// - note, this has to happen before the types are processed
// - of the original function is needed, just rename it with an underscore in front
// - then extent the wrapped class by a handwritten python or c++ function using the original name
%rename(_get_api_component) mi::neuraylib::INeuray::get_api_component;
%rename(_shutdown) mi::neuraylib::INeuray::shutdown;
%extend SmartPtr< mi::neuraylib::INeuray > {
    %pythoncode {
        def get_api_component(self, type):
            r"""
            Returns an API component from the MDL SDK API.
            See also: 'mi_neuray_api_components' for a list of built - in API components.
            See also: #register_api_component(), #unregister_api_component()
            """
            iinterface = self._get_api_component(type.IID())
            if iinterface.is_valid_interface():
                return iinterface.get_interface(type)
            else:
                return iinterface

        def shutdown(self, blocking: bool = True, run_garbage_collection: bool = True) -> "mi::Sint32":
            if run_garbage_collection:
                gc.collect()
            return self._shutdown(blocking)
    }
}




// special handling for: mi::neuraylib::IExpression
// ----------------------------------------------------------------------------

EXTEND_FUNCTION_AS(mi::neuraylib::IExpression, get_type)
EXTEND_FUNCTION_AS(mi::neuraylib::IExpression_constant, get_type)
EXTEND_FUNCTION_AS(mi::neuraylib::IExpression_call, get_type)
EXTEND_FUNCTION_AS(mi::neuraylib::IExpression_parameter, get_type)
EXTEND_FUNCTION_AS(mi::neuraylib::IExpression_direct_call, get_type)
EXTEND_FUNCTION_AS(mi::neuraylib::IExpression_temporary, get_type)

// special handling for: mi::neuraylib::IExpression_constant
// ----------------------------------------------------------------------------
EXTEND_FUNCTION_AS(mi::neuraylib::IExpression_constant, get_value)

// special handling for: mi::neuraylib::IExpression_list
// ----------------------------------------------------------------------------
EXTEND_FUNCTION_AS(mi::neuraylib::IExpression_list, get_expression)

// special handling for: mi::neuraylib::IExpression_factory
// ----------------------------------------------------------------------------
%ignore mi::neuraylib::IExpression_factory::create_cast(IExpression*, IType const*, char const*, bool) const;
%rename(_create_cast) mi::neuraylib::IExpression_factory::create_cast(IExpression*, IType const*, char const*, bool, Sint32*) const;
%extend SmartPtr<mi::neuraylib::IExpression_factory> {
    %pythoncode {
        def create_cast(self, src_expr, target_type, cast_db_name, force_cast, errors: ReturnCode = None):
            iinterface, ret = self._create_cast(src_expr, target_type, cast_db_name, force_cast)
            if errors != None:
                errors.value = ret
            iinterface.thisown = True
            return iinterface

        def create_cast_with_ret(self, src_expr, target_type, cast_db_name, force_cast):
            warnings.warn("Use `create_cast` instead using the `errors: ReturnCode` parameter.", DeprecationWarning)
            return self._create_cast(src_expr, target_type, cast_db_name, force_cast)
    }
}

%ignore mi::neuraylib::IExpression_factory::create_direct_call(const char*, IExpression_list*) const;
%rename(_create_direct_call) mi::neuraylib::IExpression_factory::create_direct_call(const char*, IExpression_list*, Sint32*) const;
%extend SmartPtr<mi::neuraylib::IExpression_factory> {
    %pythoncode {
        def create_direct_call(self, name, arguments, errors: ReturnCode = None):
            iinterface, ret = self._create_direct_call(name, arguments)
            if errors != None:
                errors.value = ret
            iinterface.thisown = True
            return iinterface

        def create_direct_call_with_ret(self, name, arguments):
            warnings.warn("Use `create_direct_call` instead using the `errors: ReturnCode` parameter.", DeprecationWarning)
            return self._create_direct_call(name, arguments)
    }
}

// special handling for: mi::neuraylib::IExpression_factory
// ----------------------------------------------------------------------------
%ignore mi::neuraylib::IExpression_factory::create_constant(IValue*) const; // omit the const overload

// special handling for: mi::neuraylib::IFunction_definition
// ----------------------------------------------------------------------------
%ignore mi::neuraylib::IFunction_definition::create_function_call(IExpression_list const*) const;
%rename(_create_function_call) mi::neuraylib::IFunction_definition::create_function_call(IExpression_list const*, Sint32*) const;
%extend SmartPtr<mi::neuraylib::IFunction_definition> {
    %pythoncode {
        def create_function_call(self, arguments, errors: ReturnCode = None):
            iinterface, ret = self._create_function_call(arguments)
            if errors != None:
                errors.value = ret
            iinterface.thisown = True
            return iinterface

        def create_function_call_with_ret(self, arguments):
            warnings.warn("Use `create_function_call` instead using the `errors: ReturnCode` parameter.", DeprecationWarning)
            return self._create_function_call(arguments)
    }
}


// special handling for: mi::neuraylib::IMdl_distiller_api
// ----------------------------------------------------------------------------
%ignore mi::neuraylib::IMdl_distiller_api::distill_material(ICompiled_material const*, char const*) const;
%ignore mi::neuraylib::IMdl_distiller_api::distill_material(ICompiled_material const*, char const*, IMap const*) const;
%rename(_distill_material) mi::neuraylib::IMdl_distiller_api::distill_material(ICompiled_material const*, char const*, IMap const*, Sint32*) const;
%extend SmartPtr<mi::neuraylib::IMdl_distiller_api> {
    %pythoncode {
        def distill_material(self, material, target, distiller_options = None, errors: ReturnCode = None):
            iinterface, ret = self._distill_material(material, target, distiller_options)
            if errors != None:
                errors.value = ret
            iinterface.thisown = True
            return iinterface

        def distill_material_with_ret(self, material, target, distiller_options = None) :
            warnings.warn("Use `distill_material` instead using the `errors: ReturnCode` parameter.", DeprecationWarning)
            return self._distill_material(material, target, distiller_options)
    }
}

// special handling for: mi::neuraylib::ICompiled_material
// ----------------------------------------------------------------------------
%ignore mi::neuraylib::ICompiled_material::get_connected_function_db_name(char const*, Size) const;
%rename(_get_connected_function_db_name) mi::neuraylib::ICompiled_material::get_connected_function_db_name(char const*, Size, Sint32*) const;
%extend SmartPtr<mi::neuraylib::ICompiled_material> {
    %pythoncode {
        def get_connected_function_db_name(self, material_instance_name, parameter_index, errors: ReturnCode = None):
            iinterface, ret = self._get_connected_function_db_name(material_instance_name, parameter_index)
            if errors != None:
                errors.value = ret
            iinterface.thisown = True
            return iinterface

        def get_connected_function_db_name_with_ret(self, material_instance_name, parameter_index):
            warnings.warn("Use `get_connected_function_db_name` instead using the `errors: ReturnCode` parameter.", DeprecationWarning)
            return self._get_connected_function_db_name(material_instance_name, parameter_index)
    }
}
// make the return of mi::neuraylib::ICompiled_material::get_cutout_opacity() a tuple
%apply mi::Float32* OUTPUT{ float* cutout_opacity };

// special handling for: mi::neuraylib::IType_enum
// ----------------------------------------------------------------------------
%rename(get_value_code_with_ret) mi::neuraylib::IType_enumeration::get_value_code(Size, Sint32*) const;


// special handling for: mi::neuraylib::IValue
// ----------------------------------------------------------------------------

EXTEND_FUNCTION_AS(mi::neuraylib::IValue_compound, get_value)
EXTEND_FUNCTION_AS(mi::neuraylib::IValue_color, get_value)
EXTEND_FUNCTION_AS(mi::neuraylib::IValue_vector, get_value)
EXTEND_FUNCTION_AS(mi::neuraylib::IValue_matrix, get_value)
EXTEND_FUNCTION_AS(mi::neuraylib::IValue_array, get_value)
EXTEND_FUNCTION_AS(mi::neuraylib::IValue_structure, get_value)
EXTEND_FUNCTION_AS(mi::neuraylib::IValue_structure, get_field)

// special handling for: mi::neuraylib::ITransaction
// ----------------------------------------------------------------------------
EXTEND_FUNCTION_AS(mi::neuraylib::ITransaction, access)
EXTEND_FUNCTION_AS(mi::neuraylib::ITransaction, edit)

%rename(_create) mi::neuraylib::ITransaction::create; // omit the last two parameters of the create function
%extend SmartPtr<mi::neuraylib::ITransaction> {
    %pythoncode{
        def create(self, type_name: str, argc = 0, argv = None):
            return  self._create(type_name, argc, argv)

        def create_as(self, type, type_name:str, argc = 0, argv = None):
            iinterface = self.create(type_name, argc, argv)
            if iinterface.is_valid_interface():
                return iinterface.get_interface(type)
            else:
                return iinterface
    }
}


// special handling for: mi::neuraylib::IImage
// ----------------------------------------------------------------------------
%ignore mi::neuraylib::IImage::set_from_canvas(ICanvas const*);
%ignore mi::neuraylib::IImage::set_from_canvas(ICanvas*);
%ignore mi::neuraylib::IImage::set_from_canvas(IArray const*);
%ignore mi::neuraylib::IImage::set_from_canvas(IArray*);
%ignore mi::neuraylib::IImage::set_from_canvas(ICanvas const*,char const *);
%ignore mi::neuraylib::IImage::set_from_canvas(ICanvas*,char const *);
%ignore mi::neuraylib::IImage::set_from_canvas(IArray const*,char const *);
%ignore mi::neuraylib::IImage::set_from_canvas(IArray*,char const *);

// insert default parameters, which seems to not work correctly otherwise with overloads
%feature("pythonprepend") mi::neuraylib::IImage::set_from_canvas%{
    argc: int = len(args)
    if argc == 1:
        args = (args[0], None, False)
    elif argc == 2:
        args = (args[0], args[1], False)
%}

%ignore mi::neuraylib::IImage::get_uvtile_uv;
%ignore mi::neuraylib::IImage::get_uvtile_uv_ranges;
%extend SmartPtr<mi::neuraylib::IImage> {

    /// Checks if there are valid uv-coordinates for given index ``frame_id`` and ``uvtile_id``.
    /// 
    /// \param frame_id     The frame ID of the frame.
    /// \param uvtile_id    The uv-tile ID of the uv-tile.
    /// \return             ``false`` if ``frame_id`` or ``uvtile_id`` out of range.
    bool has_uvtile_uv(mi::Size frame_id, mi::Size uvtile_id) const
    {
        mi::Sint32 u = 0;
        mi::Sint32 v = 0;
        return $self->get()->get_uvtile_uv(frame_id, uvtile_id, u, v) == 0;
    }

    /// Returns the u-coordinates corresponding to a uv-tile ID.
    /// 
    /// \param frame_id     The frame ID of the frame.
    /// \param uvtile_id    The uv-tile ID of the uv-tile.
    /// \return             The u-coordinate or undefined if out of range.
    ///                     See also ``has_uvtile_uv``.
    mi::Sint32 get_uvtile_u(mi::Size frame_id, mi::Size uvtile_id) const
    {
        mi::Sint32 u = 0;
        mi::Sint32 v = 0;
        mi::Sint32 res = $self->get()->get_uvtile_uv(frame_id, uvtile_id, u, v);
        return res == 0 ? u : 0;
    }

    /// Returns the v-coordinates corresponding to a uv-tile ID.
    /// 
    /// \param frame_id     The frame ID of the frame.
    /// \param uvtile_id    The uv-tile ID of the uv-tile.
    /// \return             The v-coordinate or undefined if out of range.
    ///                     See also ``has_uvtile_uv``.
    mi::Sint32 get_uvtile_v(mi::Size frame_id, mi::Size uvtile_id) const
    {
        mi::Sint32 u = 0;
        mi::Sint32 v = 0;
        mi::Sint32 res = $self->get()->get_uvtile_uv(frame_id, uvtile_id, u, v);
        return res == 0 ? v : 0;
    }

    /// Get the smallest u-coordinate for that frame.
    mi::Sint32 get_uvtile_uv_ranges_min_u(mi::Size frame_id) const
    {
        mi::Sint32 min_u, min_v, max_u, max_v;
        $self->get()->get_uvtile_uv_ranges(frame_id, min_u, min_v, max_u, max_v);
        return min_u;
    }

    /// Get the smallest v-coordinate for that frame.
    mi::Sint32 get_uvtile_uv_ranges_min_v(mi::Size frame_id) const
    {
        mi::Sint32 min_u, min_v, max_u, max_v;
        $self->get()->get_uvtile_uv_ranges(frame_id, min_u, min_v, max_u, max_v);
        return min_v;
    }

    /// Get the largest u-coordinate for that frame.
    mi::Sint32 get_uvtile_uv_ranges_max_u(mi::Size frame_id) const
    {
        mi::Sint32 min_u, min_v, max_u, max_v;
        $self->get()->get_uvtile_uv_ranges(frame_id, min_u, min_v, max_u, max_v);
        return max_u;
    }

    /// Get the largest v-coordinate for that frame.
    mi::Sint32 get_uvtile_uv_ranges_max_v(mi::Size frame_id) const
    {
        mi::Sint32 min_u, min_v, max_u, max_v;
        $self->get()->get_uvtile_uv_ranges(frame_id, min_u, min_v, max_u, max_v);
        return max_v;
    }
}

// special handling for: mi::neuraylib::ITile
// ----------------------------------------------------------------------------
%ignore mi::neuraylib::ITile::get_pixel const;
%ignore mi::neuraylib::ITile::set_pixel;
%ignore mi::neuraylib::ITile::get_data const;   // Not supported at this point, TODO involve numpy here
%ignore mi::neuraylib::ITile::get_data;         // Not supported at this point, TODO involve numpy here
%extend SmartPtr<mi::neuraylib::ITile> {

    mi::math::Color_struct get_pixel(mi::Uint32 x_offset, mi::Uint32 y_offset) const
    {
        mi::math::Color_struct color;
        $self->get()->get_pixel(x_offset, y_offset, (mi::Float32*)(&color.r));
        return color;
    }

    void set_pixel(mi::Uint32 x_offset, mi::Uint32 y_offset, const mi::math::Color_struct* color)
    {
        $self->get()->set_pixel(x_offset, y_offset, (mi::Float32*)(&color->r));
    }
}

// special handling for: mi::neuraylib::IMdl_resolved_resource_element
// ----------------------------------------------------------------------------
%ignore mi::neuraylib::IMdl_resolved_resource_element::get_uvtile_uv;
%extend SmartPtr<mi::neuraylib::IMdl_resolved_resource_element> {

    /// Checks if there are valid uv-coordinates for a given index ``i``.
    /// 
    /// \param i        The index of the requested resource entity (from 0 to #get_count()-1).
    /// \return         ``true`` if the uvtile mode is not #mi::neuraylib::UVTILE_MODE_NONE 
    ///                 and \c i is not out of range.
    bool has_uvtile_uv(mi::Size i) const
    {
        mi::Sint32 u = 0;
        mi::Sint32 v = 0;
        return $self->get()->get_uvtile_uv(i, u, v);
    }

    /// Returns the u tile index for a resource entity.
    /// 
    /// \param i        The index of the requested resource entity (from 0 to #get_count()-1).
    /// \return         The u-coordinate of the resource entity or undefined if the uvtile mode
    ///                 is not #mi::neuraylib::UVTILE_MODE_NONE or \c i is out of range.
    ///                 See also ``has_uvtile_uv``.
    mi::Sint32 get_uvtile_u(mi::Size i) const
    {
        mi::Sint32 u = 0;
        mi::Sint32 v = 0;
        bool res = $self->get()->get_uvtile_uv(i, u, v);
        return res ? u : 0;
    }

    /// Returns the v tile index for a resource entity.
    /// 
    /// \param i        The index of the requested resource entity (from 0 to #get_count()-1).
    /// \return         The v-coordinate of the resource entity or undefined if the uvtile mode
    ///                 is not #mi::neuraylib::UVTILE_MODE_NONE or \c i is out of range.
    ///                 See also ``has_uvtile_uv``.
    mi::Sint32 get_uvtile_v(mi::Size i) const
    {
        mi::Sint32 u = 0;
        mi::Sint32 v = 0;
        bool res = $self->get()->get_uvtile_uv(i, u, v);
        return res ? v : 0;
    }
}

// special handling for: mi::neuraylib::IReader/IWriter
// ----------------------------------------------------------------------------
// Data access is not supported at this point. The interface is exposed to be passed between components only.
// TODO involve numpy here

%ignore mi::neuraylib::IReader::read;
%ignore mi::neuraylib::IReader::lookahead const;
%ignore mi::neuraylib::IWriter::write;

%ignore mi::neuraylib::IReader::readline;
%extend SmartPtr<mi::neuraylib::IReader> {

    /// Reads a line from the stream.
    ///
    /// Reads at most \p size - 1 characters from the stream and stores returns them as string.
    /// Reading stops after a newline character or an end-of-file.
    /// If a newline is read, it is stored at the end of the string.
    /// 
    /// \param size     The maximum number of bytes to be read.
    /// \return         The string in case of success, or None in case of errors.
    std::string readline(mi::Sint32 size)
    {
        if (size == 0)
            return "";
        char* buffer = new char[size];
        $self->get()->readline(buffer, size);
        std::string result = std::string(buffer);
        delete[] buffer;
        return result;
    }
}

// ----------------------------------------------------------------------------

// from now on we handle mi::Sint32* as out parameter (this could be changed or later)
%apply mi::Sint32* OUTPUT{ int* errors };

// Note, the order of the includes is important here !
#ifdef IRAY_SDK
    %include "mi/neuraylib.h"
#else
    %include "mi/mdl_sdk.h"
#endif

%include "mi/neuraylib/version.h"
%include "mi/neuraylib/iattribute_set.h"
%include "mi/neuraylib/iscene_element.h"
%ignore mi::neuraylib::IJob_execution_context;
%include "mi/neuraylib/idatabase.h"
%include "mi/neuraylib/itype.h"
%include "mi/neuraylib/ivalue.h"
%include "mi/neuraylib/iexpression.h"
%include "mi/neuraylib/ifunction_call.h"
%include "mi/neuraylib/ifunction_definition.h"
%include "mi/neuraylib/iimage.h"
%include "mi/neuraylib/imaterial_instance.h"
%include "mi/neuraylib/iimage_api.h"
%include "mi/neuraylib/imdl_configuration.h"
%include "mi/neuraylib/imdl_distiller_api.h"
%include "mi/neuraylib/imdl_evaluator_api.h"
%include "mi/neuraylib/imdl_entity_resolver.h"
%include "mi/neuraylib/imdl_execution_context.h"
%include "mi/neuraylib/imdl_factory.h"
%include "mi/neuraylib/imdl_impexp_api.h"
%include "mi/neuraylib/imdl_module_builder.h"
%include "mi/neuraylib/imodule.h"
%include "mi/neuraylib/ineuray.h"
%include "mi/neuraylib/iplugin_configuration.h"
%include "mi/neuraylib/iscope.h"
%include "mi/neuraylib/itexture.h"
%include "mi/neuraylib/itile.h"
%include "mi/neuraylib/ibsdf_measurement.h"
%include "mi/neuraylib/icanvas.h"
%include "mi/neuraylib/icompiled_material.h"
%include "mi/neuraylib/ilightprofile.h"
%include "mi/neuraylib/itransaction.h"
%include "mi/neuraylib/ienum_decl.h"
%include "mi/neuraylib/ienum.h"
%include "mi/neuraylib/istream_position.h"
%include "mi/neuraylib/ireader_writer_base.h"
%include "mi/neuraylib/ireader.h"
%include "mi/neuraylib/iwriter.h"


// NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IAttribute_set)  Not added because we don't use the IAttribute_set in the MDL SDK
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IBaker)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IBsdf_measurement)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::ICanvas)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::ICanvas_base)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::ICompiled_material)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IDatabase)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IDeserialized_function_name)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IDeserialized_module_name)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IImage_api)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IMdl_configuration)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IMdl_distiller_api)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IMdl_evaluator_api)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IMdl_execution_context)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IMdl_resolved_module)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IMdl_resolved_resource)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IMdl_resolved_resource_element)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IMdl_entity_resolver)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IMdl_factory)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IMdl_impexp_api)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IMdl_module_builder)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IMdle_deserialization_callback)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IMdle_serialization_callback)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::INeuray)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IPlugin_configuration)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IScene_element)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IScope)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::ISerialized_function_name)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::ITransaction)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IAnnotation)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IAnnotation_block)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IAnnotation_definition)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IAnnotation_list)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IExpression)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IExpression_call)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IExpression_constant)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IExpression_direct_call)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IExpression_factory)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IExpression_list)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IExpression_parameter)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IExpression_temporary)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IFunction_call)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IFunction_definition)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IImage)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::ILightprofile)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IMaterial_instance)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IMessage)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IModule)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::ITexture)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::ITile)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IType)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IType_alias)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IType_array)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IType_atomic)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IType_bool)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IType_bsdf)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IType_bsdf_measurement)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IType_color)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IType_compound)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IType_df)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IType_double)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IType_edf)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IType_enum)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IType_factory)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IType_float)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IType_hair_bsdf)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IType_int)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IType_light_profile)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IType_list)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IType_matrix)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IType_reference)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IType_resource)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IType_string)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IType_struct)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IType_texture)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IType_vdf)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IType_vector)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IValue)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IValue_array)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IValue_atomic)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IValue_bool)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IValue_bsdf_measurement)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IValue_color)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IValue_compound)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IValue_double)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IValue_enum)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IValue_factory)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IValue_float)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IValue_int)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IValue_invalid_df)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IValue_light_profile)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IValue_list)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IValue_matrix)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IValue_resource)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IValue_string)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IValue_string_localized)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IValue_struct)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IValue_texture)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IValue_vector)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IStream_position)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IReader_writer_base)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IReader)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IWriter)


// NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IAttribute_set)  Not added because we don't use the IAttribute_set in the MDL SDK
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IBaker)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IBsdf_measurement)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, ICanvas)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, ICanvas_base)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, ICompiled_material)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IDatabase)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IDeserialized_function_name)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IDeserialized_module_name)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IImage_api)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IMdl_configuration)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IMdl_distiller_api)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IMdl_evaluator_api)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IMdl_execution_context)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IMdl_resolved_module)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IMdl_resolved_resource)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IMdl_resolved_resource_element)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IMdl_entity_resolver)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IMdle_deserialization_callback)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IMdle_serialization_callback)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IMdl_factory)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IMdl_impexp_api)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IMdl_module_builder)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IMessage)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IModule)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, INeuray)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IPlugin_configuration)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IScene_element)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IScope)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, ISerialized_function_name)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, ITransaction)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IAnnotation)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IAnnotation_block)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IAnnotation_definition)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IAnnotation_list)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IExpression)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IExpression_call)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IExpression_constant)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IExpression_direct_call)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IExpression_factory)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IExpression_list)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IExpression_parameter)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IExpression_temporary)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IFunction_call)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IFunction_definition)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IImage)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, ILightprofile)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IMaterial_instance)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, ITexture)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, ITile)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IType)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IType_alias)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IType_array)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IType_atomic)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IType_bool)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IType_bsdf)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IType_bsdf_measurement)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IType_color)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IType_compound)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IType_df)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IType_double)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IType_edf)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IType_enum)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IType_factory)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IType_float)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IType_hair_bsdf)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IType_int)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IType_light_profile)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IType_list)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IType_matrix)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IType_reference)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IType_resource)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IType_string)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IType_struct)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IType_texture)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IType_vdf)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IType_vector)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IValue)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IValue_array)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IValue_atomic)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IValue_bool)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IValue_bsdf_measurement)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IValue_color)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IValue_compound)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IValue_double)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IValue_enum)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IValue_factory)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IValue_float)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IValue_int)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IValue_invalid_df)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IValue_light_profile)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IValue_list)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IValue_matrix)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IValue_resource)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IValue_string)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IValue_string_localized)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IValue_struct)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IValue_texture)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IValue_vector)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IStream_position)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IReader_writer_base)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IReader)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IWriter)

extern void _print_open_handle_statistic();
extern void _enable_print_ref_counts(bool);

// add additional functions to support inter-op with existing neuray instances
// Maybe move that separate files
%apply unsigned long long{ uint64_t }
%{
template<typename IInterface>
IInterface* attach(uint64_t ptr_as_uint64)
{
    return reinterpret_cast<IInterface*>(ptr_as_uint64);
}
%}

template<typename IInterface>
IInterface* attach(uint64_t ptr_as_uint64);

// Alternative entry-point to the API that can be used when a neuray instance
// already exists in the current process. In that case, DO NOT use `get_neuray()`
// before calling this functions and DO NOT shutdown neuray when finished.
%template(attach_ineuray) attach<mi::neuraylib::INeuray>;

// Similar to the main neuray instance, we can share existing transactions.
%template(attach_itransaction) attach<mi::neuraylib::ITransaction>;

#if STANDALONE_SUPPORT_ENABLED
// external entry points for loading the library
extern mi::neuraylib::INeuray* load_and_get_ineuray(const char*);
extern bool load_plugin(mi::neuraylib::INeuray*, const char*);
extern int unload();
#endif
