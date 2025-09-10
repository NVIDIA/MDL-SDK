import inspect
import os
import unittest

# Needed by check_enumerators().
import _pymdlsdk

try:  # pragma: no cover
    # testing from within a package or CI
    from .setup import SDK, BindingModule
    from .unittest_base import UnittestBase
    pymdlsdk = BindingModule
except ImportError:  # pragma: no cover
    # local testing
    from setup import SDK
    from unittest_base import UnittestBase
    import pymdlsdk


# Test basic features of the binding logic it self
class MainBinding(UnittestBase):
    sdk: SDK = None
    tf: pymdlsdk.IType_factory = None

    @classmethod
    def setUpClass(self):
        print(f"Running tests in {__file__} in process with id: {os.getpid()}")
        self.sdk = SDK()
        self.sdk.load(addExampleSearchPath=False, loadImagePlugins=False)
        self.tf = self.sdk.mdlFactory.create_type_factory(self.sdk.transaction)

    @classmethod
    def tearDownClass(self):
        self.tf = None
        self.sdk.unload()
        self.sdk = None
        print(f"\nFinished tests in {__file__}\n")

    # Returned IInterfaces can be nullptr in C++. They are returned as invalid interfaces in python.
    # Meaning the returned object is NOT None.
    def test_with_nullptr(self):
        color: pymdlsdk.IType_color = self.tf.create_color()
        self.assertIsValidInterface(color)

        # acquiring an incompatible interface
        invalidInt: pymdlsdk.IType_int = color.get_interface(pymdlsdk.IType_int)
        self.assertIsNotNone(invalidInt)  # returned object exists
        self.assertFalse(invalidInt.is_valid_interface())  # but the interface is not valid

        # same for with-blocks, meaning you can always use a with block for returned IInterfaces
        with color.get_interface(pymdlsdk.IType_texture) as invalidTexture:
            self.assertIsNotNone(invalidTexture)  # returned object exists
            self.assertFalse(invalidTexture.is_valid_interface())  # but the interface is not valid

    # Accessing, i.e. calling, invalid IInterfaces should throw an Exception rather than crashing.
    def test_nullptr_access(self):
        invalidBool: pymdlsdk.IType_bool = self.tf.create_color().get_interface(pymdlsdk.IType_bool)
        self.assertIsNotNone(invalidBool)  # returned object exists
        self.assertFalse(invalidBool.is_valid_interface())  # but the interface is not valid
        try:
            _: pymdlsdk.IType.Kind = invalidBool.get_kind()
            self.assertTrue(False, "expected exception not fired")  # pragma: no cover
            # should not be reached
        except RuntimeError as ex:
            print(f"expected error: {ex}")
        except Exception as ex:  # pragma: no cover
            print(f"exception: {ex}")  # would catch if we do not catch `RuntimeError` before
            self.assertTrue(False, "expected exception of wrong type")  # should not be reached

    def test_manual_release(self):
        color: pymdlsdk.IType_color = self.tf.create_color()
        self.assertEqual(color.__iinterface_refs__(), 1)
        self.assertTrue(color.is_valid_interface())
        color.release()  # the manual release
        self.assertFalse(color.is_valid_interface())

    def test_attach_neuray_functions(self):
        handle: int = self.sdk.neuray.__iinterface_ptr_as_uint64__(retain=True)  # handle normally comes from a running application
        neuray: pymdlsdk.INeuray = pymdlsdk.attach_ineuray(handle)
        self.assertEqual(neuray.get_status(), pymdlsdk.INeuray.Status.STARTED)
        neuray = None

    def test_attach_transaction_functions(self):
        handle: int = self.sdk.transaction.__iinterface_ptr_as_uint64__(retain=True)  # handle normally comes from a running application
        trans: pymdlsdk.ITransaction = pymdlsdk.attach_itransaction(handle)
        self.assertIsValidInterface(trans)
        self.assertTrue(trans.is_open())
        trans = None

    def test_swig_repr(self):
        text: str = self.sdk.neuray.__repr__()
        self.assertNotNullOrEmpty(text)
        color: pymdlsdk.IType_color = self.tf.create_color()
        invalidFloat: pymdlsdk.IType_float = color.get_interface(pymdlsdk.IType_float)
        text: str = invalidFloat.__repr__()
        self.assertNotNullOrEmpty(text)

    def test_print_ref_counts(self):
        pymdlsdk._enable_print_ref_counts(True)
        with self.sdk.neuray.get_api_component(pymdlsdk.IMdl_configuration) as cfg:
            pass
        pymdlsdk._enable_print_ref_counts(False)

    def check_enumerators(
        self,
        enum_name: str,
        cpp_prefix: str,
        cpp_exclude_prefixes: list,
        cpp_exclude_suffixes: list,
        cpp_strip_prefix: str,
        python_name):
        """
        Checks that enumerators of an enum agree between the C++ API and the Python binding.

        Args:
            enum_name:            (short) name of the enum to be checked (for logging only)
            cpp_prefix:           prefix of enumerators from the C++ API to consider
            cpp_exclude_prefixes: list of prefixes of enumerators from the C++ API to ignore
            cpp_exclude_suffixes: list of suffixes of enumerators from the C++ API to ignore
            cpp_strip_prefix:     prefix of considered enumerators from the C++ API to be
                                  stripped for comparisons with the Python binding
            python_name:          identifier from the Python binding
        """

        # Get enumerators in the C++ API as extracted by SWIG from the public headers.
        cpp = []
        for name, value in inspect.getmembers(_pymdlsdk):
            if not name.startswith(cpp_prefix):
                continue
            exclude = False
            for prefix in cpp_exclude_prefixes:
                if name.startswith(prefix): # pragma: no cover
                    exclude = True
                    break
            if exclude: # pragma: no cover
                continue
            for suffix in cpp_exclude_suffixes:
                if name.endswith(suffix):
                    exclude = True
                    break
            if exclude:
                continue
            cpp.append((name.lstrip(cpp_strip_prefix), value))
        cpp = sorted(cpp, key=lambda pair: (pair[1], pair[0]))

        # Get enumerators in the Python binding as defined in
        # prod/bindings/mdl_python/swig_library/mi_neuraylib_enums.i.
        python = [(enum.name, enum.value) for enum in list(python_name)]
        python = sorted(python, key=lambda pair: (pair[1], pair[0]))

        # Fail if not equal.
        if python != cpp: # pragma: no cover
            print(f"\nError: Mismatch between enumerators of {enum_name} in the C++ API and the Python binding:")
            print(f"Enumerators in the C++ API missing in the Python binding: {set(cpp)-set(python)}")
            print(f"Enumerators in the Python binding missing in the C++ API: {set(python)-set(cpp)}")
            print("Did you forget to adapt the Python binding in mi_neuraylib_enums.i?\n")
        self.assertTrue(python == cpp)

    def test_enumerators(self):
        self.check_enumerators(
            "IFunction_definition::Semantics",
            "_IFunction_definition_DS_",
            ["_IFunction_definition_DS_INTRINSIC_NVIDIA_DF_"],
            ["_FIRST", "_LAST"],
            "_IFunction_definition_",
            pymdlsdk.IFunction_definition.Semantics)

# run all tests of this file
if __name__ == '__main__':
    unittest.main()  # pragma: no cover
