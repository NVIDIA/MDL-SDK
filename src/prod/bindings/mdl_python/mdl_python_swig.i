/******************************************************************************
 * Copyright 2022 NVIDIA Corporation. All rights reserved.
 *****************************************************************************/

/* File : example.i */
// 1- python instantiate_templates_inline.py example_swig.i "" ..\..\..\\include processed_headers processed_headers_dummy
// 2- swig -I./processed_headers -I..\..\..\\include -c++ -python -cppext cpp example_swig.i

%module pymdlsdk

%begin %{
#ifdef _MSC_VER
#define SWIG_PYTHON_INTERPRETER_NO_DEBUG
#endif
%}


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
#ifndef ADD_STANDALONE_SUPPORT
#define ADD_STANDALONE_SUPPORT 1
#endif

%{
/***************************************************************************************************
 * Copyright 2022 NVIDIA Corporation. All rights reserved.
 **************************************************************************************************/

#ifdef IRAY_SDK
    #include "mi/neuraylib.h"
#else
    #include "mi/mdl_sdk.h"
#endif

#include "mdl_python.h"

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
%}

#if ADD_STANDALONE_SUPPORT
%{
extern mi::neuraylib::INeuray* load_and_get_ineuray(const char*);
extern bool load_plugin(mi::neuraylib::INeuray*, const char*);
extern bool unload();
%}
#endif

%pythoncode {
    from enum import Enum
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
    %nodefaultctor CLASS_TYPE;
    %nodefaultdtor CLASS_TYPE;

    // handle retain and release a bit different:
    // - we hide the internal ref countering from the python interface
    // - we add a new release function, that just like the __exit__,
    //   invalidates the python proxy using `drop` (which decrements the ref count internally)
    %rename(_retain) CLASS_TYPE::retain;
    %rename(_release) CLASS_TYPE::release;
    %ignore SmartPtr<CLASS_TYPE>::retain;
    %ignore SmartPtr<CLASS_TYPE>::drop;

    // hide the get function, this is not supposed to be used from python
    %ignore SmartPtr<CLASS_TYPE>::get;

    // hide debugging support
    %ignore SmartPtr<CLASS_TYPE>::get_debug_str;
    %ignore SmartPtr<CLASS_TYPE>::assign_open_handle_typename;

    // we don't want people to create objects using the constructor
    %nodefaultctor SmartPtr<CLASS_TYPE>;
    %nodefaultdtor SmartPtr<CLASS_TYPE>;

    // we re-implement get_interface here
    %ignore CLASS_TYPE::get_interface;

    %ignore CLASS_TYPE::s_kind;

    %extend SmartPtr<CLASS_TYPE> {

        // This method is also renamed to IID() above
        static const mi::base::Uuid IID()
        {
            return CLASS_TYPE::IID();
        }

        // this creates a static function at the target interface class to convert an interface pointer
        // into an object of the own class, e.g.:
        // [python] cfg = pymdlsdk.IMdl_configuration.get_interface(iinterface)
        static SmartPtr<CLASS_TYPE>* _get_interface(mi::base::IInterface* iface) {
            auto ptr = new SmartPtr<CLASS_TYPE>(iface->get_interface<CLASS_TYPE>());
            ptr->assign_open_handle_typename("CLASS_TYPE");
            return ptr;
        }

        // Allow "foo.get_interface(Bar)" instead of "Bar.get_interface(foo)"
        // Note, this will return an invalid pointer if casting fails
        %pythoncode {
            def get_interface(self, type):
                typed_interface = type._get_interface(self)
                typed_interface.thisown = True
                return typed_interface
        }

        // TODO create a string representation 
        //const char* __repr__()
        //{
        //    return (*$self).get_debug_str();
        //}
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
        auto ptr = new SmartPtr<IINTERFACE_TYPE>($1);
        ptr->assign_open_handle_typename("IINTERFACE_TYPE");
        %set_output(SWIG_NewPointerObj(ptr, $descriptor(SmartPtr<IINTERFACE_TYPE> ARGS*), SWIG_POINTER_OWN));
    }
    // TODO: needs to be tested when used (ask Joerg about this)
    %typemap(out) IINTERFACE_TYPE ARGS *const& {
        assert(false && "Untested typemap out!");
        auto ptr = new SmartPtr<IINTERFACE_TYPE>($1);
        ptr->assign_open_handle_typename("IINTERFACE_TYPE");
        %set_output(SWIG_NewPointerObj(ptr, $descriptor(SmartPtr<IINTERFACE_TYPE> ARGS*), SWIG_POINTER_OWN));
    }

    // Convert SmartPtr<IInterface> to IInterface* when passed as arguments
    %typemap(in) IINTERFACE_TYPE ARGS * {
        void* arg_ptr = 0;
        int res = SWIG_ConvertPtr($input, (void**)&arg_ptr, $descriptor(SmartPtr<IINTERFACE_TYPE> ARGS*), 0);
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
                    $1 = base_sptr->get_iinterface_weak<IINTERFACE_TYPE>();
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
                auto base_wptr = base_sptr->get_iinterface_weak<IINTERFACE_TYPE>();
                $1 = base_wptr ? 1 : 0;
            }
        }
    }
%enddef

// Instantiate Handle template for Neuray IInterfaces
%define NEURAY_CREATE_HANDLE_TEMPLATE(ORG_NAMESPACE, IINTERFACE_TYPE)
    //#warning Instanciated: IINTERFACE_TYPE
    %template(IINTERFACE_TYPE) SmartPtr<ORG_NAMESPACE::IINTERFACE_TYPE>;
%enddef

// Utility macro to wrap return values into a python function call
// e.g. for conversions like enum value conversions:
// IINTERFACE_TYPE:  mi::neuraylib::IValue
// FUNCTION_NAME:    get_kind
// PYTHON_ENUM_TYPE: IValue.Kind
%define WRAP_RETURN_IN_FUNCTION(IINTERFACE_TYPE, FUNCTION_NAME, WRAP_FUNCTION)
    %rename(_ ## FUNCTION_NAME) IINTERFACE_TYPE::FUNCTION_NAME;
    %extend SmartPtr<IINTERFACE_TYPE> {
        %pythoncode{
            def FUNCTION_NAME(self, *args):
                return WRAP_FUNCTION(self._ ## FUNCTION_NAME ## (*args))
        }
    }
%enddef
// Adds a template-like casting helper as we have in the neuray headers for c++
%define WRAP_TEMPLATE_RETURN_IN_FUNCTION(IINTERFACE_TYPE, FUNCTION_NAME)
    %rename(_ ## FUNCTION_NAME) IINTERFACE_TYPE::FUNCTION_NAME;
    %extend SmartPtr<IINTERFACE_TYPE> {
        %pythoncode{
            def FUNCTION_NAME(self, *args):
                return  self._ ## FUNCTION_NAME ## (*args)

            def FUNCTION_NAME ## _as(self, type, *args):
                iinterface = self._ ## FUNCTION_NAME ## (*args)
                if iinterface.is_valid_interface():
                    typed_interface = iinterface.get_interface(type)
                    iinterface = None
                    return typed_interface
                else:
                    return iinterface
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

//TODO: Fix for MSVC?
typedef long long          Sint64;  ///< 64-bit   signed integer.
typedef unsigned long long Uint64;  ///< 64-bit unsigned integer.

typedef float              Float32; ///< 32-bit float.
typedef double             Float64; ///< 64-bit float.

// Define Size to be signed, otherwise Java would use BigInteger
//TODO: Add a typemap to map this to Sint64 for the wrapper, possibly adding a check for negative values
typedef Sint64             Size;

} // namespace

namespace mi {
namespace base {
struct Uuid
{
    Uint32 m_id1; ///< First  value.
    Uint32 m_id2; ///< Second value.
    Uint32 m_id3; ///< Third  value.
    Uint32 m_id4; ///< Fourth value.
};
} // namespace
} // namespace

namespace mi {
namespace math {
struct Color_struct
{
    /// Red color component
    Float32 r;
    /// Green color component
    Float32 g;
    /// Blue color component
    Float32 b;
    /// Alpha value, 0.0 is fully transparent and 1.0 is opaque; value can lie outside that range.
    Float32 a;
};
} // namespace
} // namespace

#define mi_static_assert(ignored)


// ----------------------------------------------------------------------------
// mi::base
// ----------------------------------------------------------------------------

// Do not run the dice script on the base interface
%ignore mi::base::IInterface;
NEURAY_INIT_INTERFACE(mi::base::IInterface);
%include "mi/base/iinterface.h"
%include "mi/base/interface_declare.h"

NEURAY_DEFINE_HANDLE_TYPEMAP(mi::base::IInterface)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::base, IInterface)

// ----------------------------------------------------------------------------
// mi
// ----------------------------------------------------------------------------


DICE_INTERFACE_MI(IArray);
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
DICE_INTERFACE_MI(IDifference);
DICE_INTERFACE_MI(IString);
DICE_INTERFACE_MI(IData);
DICE_INTERFACE_MI(IData_simple);
DICE_INTERFACE_MI(IData_collection);
DICE_INTERFACE_MI(IFloat32);
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

%include "mi/neuraylib/vector_typedefs.h"
%include "mi/neuraylib/idata.h"
%include "mi/neuraylib/istring.h"
%include "mi/neuraylib/iarray.h"
%include "mi/neuraylib/inumber.h"
%include "mi/neuraylib/icompound.h"
%include "mi/neuraylib/ivector.h"

NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IArray)
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
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IString)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IData)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IData_simple)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IString)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IData_collection)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::IFloat32)
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

NEURAY_CREATE_HANDLE_TEMPLATE(mi, IArray)
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
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IString)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IData)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IData_simple)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IData_collection)
NEURAY_CREATE_HANDLE_TEMPLATE(mi, IFloat32)
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

// ----------------------------------------------------------------------------
// mi::neuray
// ----------------------------------------------------------------------------
// one note on the order here, currently trying to keep the blocks and in each
// block we have alphabetical sorting

DICE_INTERFACE(IAttribute_set);
DICE_INTERFACE(IBsdf_measurement)
DICE_INTERFACE(ICanvas)
DICE_INTERFACE(ICanvas_base)
DICE_INTERFACE(ICompiled_material)
DICE_INTERFACE(IDatabase);
DICE_INTERFACE(IDeserialized_function_name);
DICE_INTERFACE(IDeserialized_module_name);
DICE_INTERFACE(IImage_api);
DICE_INTERFACE(IMdl_configuration);
DICE_INTERFACE(IMessage);
DICE_INTERFACE(IMdl_execution_context);
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
DICE_INTERFACE(IMaterial_definition)
DICE_INTERFACE(IMaterial_instance)
DICE_INTERFACE(IModule)
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

%extend mi::neuraylib::ITile {

        mi::math::Color_struct get_pixel(
            Uint32 x_offset,
            Uint32 y_offset
            ) const
        {
            mi::math::Color_struct color;
            $self->get_pixel(x_offset, y_offset, (mi::Float32*)(& color.r));
            return color;
        }

        void set_pixel(
            Uint32 x_offset,
            Uint32 y_offset,
            const mi::math::Color_struct* color)
        {
            $self->set_pixel(x_offset, y_offset, (mi::Float32*)(& color->r));
        }

 }

 %extend mi::IFloat32 {
        float get_value() const
        {
            float v;
            $self->get_value(v);
            return v;
        }
}

// special handling for: mi::neuraylib::INeuray
// ----------------------------------------------------------------------------
// Rewrite of special template functions to make life easier
// - note, this has to happen before the types are processed
// - of the original function is needed, just rename it with an underscore in front
// - then extent the wrapped class by a handwritten python or c++ function using the original name
%rename(_get_api_component) mi::neuraylib::INeuray::get_api_component;
%extend SmartPtr< mi::neuraylib::INeuray > {

    %pythoncode {
        class Status(Enum):
            PRE_STARTING = _pymdlsdk._INeuray_PRE_STARTING
            STARTING = _pymdlsdk._INeuray_STARTING
            STARTED = _pymdlsdk._INeuray_STARTED
            SHUTTINGDOWN = _pymdlsdk._INeuray_SHUTTINGDOWN
            SHUTDOWN = _pymdlsdk._INeuray_SHUTDOWN
            FAILURE = _pymdlsdk._INeuray_FAILURE

        def get_api_component(self, type):
            iinterface = self._get_api_component(type.IID())
            if iinterface.is_valid_interface():
                typed_interface = iinterface.get_interface(type)
                iinterface = None
                return typed_interface
            else:
                return iinterface
    }
}
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::INeuray, get_status, INeuray.Status)

// special handling for: mi::neuraylib::IExpression
// ----------------------------------------------------------------------------
// We manually define the enums in the correct proxy class
%extend SmartPtr<mi::neuraylib::IExpression> {
    %pythoncode{
        class Kind(Enum) :
            EK_CONSTANT = _pymdlsdk._IExpression_EK_CONSTANT
            EK_CALL = _pymdlsdk._IExpression_EK_CALL
            EK_PARAMETER = _pymdlsdk._IExpression_EK_PARAMETER
            EK_DIRECT_CALL = _pymdlsdk._IExpression_EK_DIRECT_CALL
            EK_TEMPORARY = _pymdlsdk._IExpression_EK_TEMPORARY
    }
}

// change all functions that return IType::Kind
// .. and this needs to be done for all overloads
// maybe we can use a type map for this as well, but I wasn't able to inject python code into them
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IExpression, get_kind, IExpression.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IExpression_constant, get_kind, IExpression.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IExpression_call, get_kind, IExpression.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IExpression_parameter, get_kind, IExpression.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IExpression_direct_call, get_kind, IExpression.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IExpression_temporary, get_kind, IExpression.Kind)

WRAP_TEMPLATE_RETURN_IN_FUNCTION(mi::neuraylib::IExpression, get_type)
WRAP_TEMPLATE_RETURN_IN_FUNCTION(mi::neuraylib::IExpression_constant, get_type)
WRAP_TEMPLATE_RETURN_IN_FUNCTION(mi::neuraylib::IExpression_call, get_type)
WRAP_TEMPLATE_RETURN_IN_FUNCTION(mi::neuraylib::IExpression_parameter, get_type)
WRAP_TEMPLATE_RETURN_IN_FUNCTION(mi::neuraylib::IExpression_direct_call, get_type)
WRAP_TEMPLATE_RETURN_IN_FUNCTION(mi::neuraylib::IExpression_temporary, get_type)

// special handling for: mi::neuraylib::IExpression_constant
// ----------------------------------------------------------------------------
WRAP_TEMPLATE_RETURN_IN_FUNCTION(mi::neuraylib::IExpression_constant, get_value)

// special handling for: mi::neuraylib::IExpression_list
// ----------------------------------------------------------------------------
WRAP_TEMPLATE_RETURN_IN_FUNCTION(mi::neuraylib::IExpression_list, get_expression)

// special handling for: mi::neuraylib::IExpression_factory
// Note, mind the const at the end of the function and the parameters and omit param namespaces
// Whenever the (output) return code is optional, we add a manual renamed function
// with a '_with_ret' suffix.
// ----------------------------------------------------------------------------
%rename(create_cast_with_ret) mi::neuraylib::IExpression_factory::create_cast(IExpression*, IType const*, char const*, bool, Sint32*) const;
%rename(create_direct_call_with_ret) mi::neuraylib::IExpression_factory::create_direct_call(char const*, IExpression_list*, Sint32*) const;
%ignore mi::neuraylib::IExpression_factory::create_constant(IValue*) const; // omit the const overload

// special handling for: mi::neuraylib::IMaterial_definition
// ----------------------------------------------------------------------------
%rename(create_material_instance_with_ret) mi::neuraylib::IMaterial_definition::create_material_instance(IExpression_list const*, Sint32*) const;

// special handling for: mi::neuraylib::IFunction_definition
// ----------------------------------------------------------------------------
%rename(create_function_call_with_ret) mi::neuraylib::IFunction_definition::create_function_call(IExpression_list const*, Sint32*) const;

// special handling for: mi::neuraylib::IMdl_factory
// ----------------------------------------------------------------------------
%rename(create_texture_with_ret) mi::neuraylib::IMdl_factory::create_texture(ITransaction*, char const*, IType_texture::Shape, Float32, char const*, bool, Sint32*);
%rename(create_light_profile_with_ret) mi::neuraylib::IMdl_factory::create_light_profile(ITransaction*, char const*, bool, Sint32*);
%rename(create_bsdf_measurement_with_ret) mi::neuraylib::IMdl_factory::create_bsdf_measurement(ITransaction*, char const*, bool, Sint32*);

// special handling for: mi::neuraylib::ICompiled_material
// ----------------------------------------------------------------------------
%rename(get_connected_function_db_name_with_ret) mi::neuraylib::ICompiled_material::get_connected_function_db_name(char const*, Size, Sint32*) const;

// special handling for: mi::neuraylib::IType
// ----------------------------------------------------------------------------
%extend SmartPtr<mi::neuraylib::IType> {
    %pythoncode{
        class Kind(Enum) :
            TK_ALIAS = _pymdlsdk._IType_TK_ALIAS
            TK_BOOL = _pymdlsdk._IType_TK_BOOL
            TK_INT = _pymdlsdk._IType_TK_INT
            TK_ENUM = _pymdlsdk._IType_TK_ENUM
            TK_FLOAT = _pymdlsdk._IType_TK_FLOAT
            TK_DOUBLE = _pymdlsdk._IType_TK_DOUBLE
            TK_STRING = _pymdlsdk._IType_TK_STRING
            TK_VECTOR = _pymdlsdk._IType_TK_VECTOR
            TK_MATRIX = _pymdlsdk._IType_TK_MATRIX
            TK_COLOR = _pymdlsdk._IType_TK_COLOR
            TK_ARRAY = _pymdlsdk._IType_TK_ARRAY
            TK_STRUCT = _pymdlsdk._IType_TK_STRUCT
            TK_TEXTURE = _pymdlsdk._IType_TK_TEXTURE
            TK_LIGHT_PROFILE = _pymdlsdk._IType_TK_LIGHT_PROFILE
            TK_BSDF_MEASUREMENT = _pymdlsdk._IType_TK_BSDF_MEASUREMENT
            TK_BSDF = _pymdlsdk._IType_TK_BSDF
            TK_HAIR_BSDF = _pymdlsdk._IType_TK_HAIR_BSDF
            TK_EDF = _pymdlsdk._IType_TK_EDF
            TK_VDF = _pymdlsdk._IType_TK_VDF
    }
}

%rename(get_value_code_with_ret) mi::neuraylib::IType_enumeration::get_value_code(Size, Sint32*) const;

WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IType, get_kind, IType.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IType_alias, get_kind, IType.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IType_atomic, get_kind, IType.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IType_bool, get_kind, IType.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IType_int, get_kind, IType.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IType_enumeration, get_kind, IType.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IType_float, get_kind, IType.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IType_double, get_kind, IType.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IType_string, get_kind, IType.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IType_compound, get_kind, IType.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IType_vector, get_kind, IType.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IType_matrix, get_kind, IType.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IType_color, get_kind, IType.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IType_array, get_kind, IType.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IType_structure, get_kind, IType.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IType_reference, get_kind, IType.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IType_resource, get_kind, IType.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IType_texture, get_kind, IType.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IType_light_profile, get_kind, IType.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IType_bsdf_measurement, get_kind, IType.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IType_df, get_kind, IType.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IType_bsdf, get_kind, IType.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IType_hair_bsdf, get_kind, IType.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IType_edf, get_kind, IType.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IType_vdf, get_kind, IType.Kind)


// special handling for: mi::neuraylib::IValue
// ----------------------------------------------------------------------------
%extend SmartPtr<mi::neuraylib::IValue> {
    %pythoncode{
        class Kind(Enum) :
            VK_BOOL = _pymdlsdk._IValue_VK_BOOL
            VK_INT = _pymdlsdk._IValue_VK_INT
            VK_ENUM = _pymdlsdk._IValue_VK_ENUM
            VK_FLOAT = _pymdlsdk._IValue_VK_FLOAT
            VK_DOUBLE = _pymdlsdk._IValue_VK_DOUBLE
            VK_STRING = _pymdlsdk._IValue_VK_STRING
            VK_VECTOR = _pymdlsdk._IValue_VK_VECTOR
            VK_MATRIX = _pymdlsdk._IValue_VK_MATRIX
            VK_COLOR = _pymdlsdk._IValue_VK_COLOR
            VK_ARRAY = _pymdlsdk._IValue_VK_ARRAY
            VK_STRUCT = _pymdlsdk._IValue_VK_STRUCT
            VK_INVALID_DF = _pymdlsdk._IValue_VK_INVALID_DF
            VK_TEXTURE = _pymdlsdk._IValue_VK_TEXTURE
            VK_LIGHT_PROFILE = _pymdlsdk._IValue_VK_LIGHT_PROFILE
            VK_BSDF_MEASUREMENT = _pymdlsdk._IValue_VK_BSDF_MEASUREMENT
    }
}

WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IValue, get_kind, IValue.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IValue_bool, get_kind, IValue.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IValue_int, get_kind, IValue.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IValue_enumeration, get_kind, IValue.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IValue_float, get_kind, IValue.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IValue_double, get_kind, IValue.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IValue_string, get_kind, IValue.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IValue_vector, get_kind, IValue.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IValue_matrix, get_kind, IValue.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IValue_color, get_kind, IValue.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IValue_array, get_kind, IValue.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IValue_structure, get_kind, IValue.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IValue_invalid_df, get_kind, IValue.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IValue_texture, get_kind, IValue.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IValue_light_profile, get_kind, IValue.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IValue_bsdf_measurement, get_kind, IValue.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IValue_atomic, get_kind, IValue.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IValue_string_localized, get_kind, IValue.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IValue_compound, get_kind, IValue.Kind)
WRAP_RETURN_IN_FUNCTION(mi::neuraylib::IValue_resource, get_kind, IValue.Kind)

WRAP_TEMPLATE_RETURN_IN_FUNCTION(mi::neuraylib::IValue_compound, get_value)
WRAP_TEMPLATE_RETURN_IN_FUNCTION(mi::neuraylib::IValue_color, get_value)
WRAP_TEMPLATE_RETURN_IN_FUNCTION(mi::neuraylib::IValue_vector, get_value)
WRAP_TEMPLATE_RETURN_IN_FUNCTION(mi::neuraylib::IValue_matrix, get_value)
WRAP_TEMPLATE_RETURN_IN_FUNCTION(mi::neuraylib::IValue_array, get_value)
WRAP_TEMPLATE_RETURN_IN_FUNCTION(mi::neuraylib::IValue_structure, get_value)
WRAP_TEMPLATE_RETURN_IN_FUNCTION(mi::neuraylib::IValue_structure, get_field)

// special handling for: mi::neuraylib::ITransaction
// ----------------------------------------------------------------------------
WRAP_TEMPLATE_RETURN_IN_FUNCTION(mi::neuraylib::ITransaction, access)
WRAP_TEMPLATE_RETURN_IN_FUNCTION(mi::neuraylib::ITransaction, edit)

// special handling for: mi::neuraylib::IImage
// ----------------------------------------------------------------------------
%ignore mi::neuraylib::IImage::set_from_canvas(ICanvas const*);
%ignore mi::neuraylib::IImage::set_from_canvas(ICanvas*);
%ignore mi::neuraylib::IImage::set_from_canvas(IArray const*);
%ignore mi::neuraylib::IImage::set_from_canvas(IArray*);

// ----------------------------------------------------------------------------

// from now on we handle mi::Sint32* as out parameter (this could be changed or later)
%apply mi::Sint32* OUTPUT{ int* };

// Note, the order of the includes is important here !
#ifdef IRAY_SDK
    %include "mi/neuraylib.h"
#else
    %include "mi/mdl_sdk.h"
#endif

%include "mi/neuraylib/version.h"
%include "mi/neuraylib/iattribute_set.h"
%include "mi/neuraylib/iscene_element.h"
%include "mi/neuraylib/idatabase.h"
%include "mi/neuraylib/itype.h"
%include "mi/neuraylib/ivalue.h"
%include "mi/neuraylib/iexpression.h"
%include "mi/neuraylib/ifunction_call.h"
%include "mi/neuraylib/ifunction_definition.h"
%include "mi/neuraylib/iimage.h"
%include "mi/neuraylib/imaterial_definition.h"
%include "mi/neuraylib/imaterial_instance.h"
%include "mi/neuraylib/iimage_api.h"
%include "mi/neuraylib/imdl_configuration.h"
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


NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IAttribute_set)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IBsdf_measurement)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::ICanvas)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::ICanvas_base)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::ICompiled_material)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IDatabase)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IDeserialized_function_name)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IDeserialized_module_name)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IImage_api)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IMdl_configuration)
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IMdl_execution_context)
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
NEURAY_DEFINE_HANDLE_TYPEMAP(mi::neuraylib::IMaterial_definition)
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


NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IAttribute_set)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IBsdf_measurement)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, ICanvas)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, ICanvas_base)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, ICompiled_material)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IDatabase)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IDeserialized_function_name)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IDeserialized_module_name)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IImage_api)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IMdl_configuration)
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IMdl_execution_context)
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
NEURAY_CREATE_HANDLE_TEMPLATE(mi::neuraylib, IMaterial_definition)
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

#if ADD_STANDALONE_SUPPORT
// external entry points for loading the library
extern mi::neuraylib::INeuray* load_and_get_ineuray(const char*);
extern bool load_plugin(mi::neuraylib::INeuray*, const char*);
extern int unload();
#endif

