import unittest
import os
import inspect

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


class MainInspection(UnittestBase):
    sdk: SDK = None
    tf: pymdlsdk.IType_factory = None
    vf: pymdlsdk.IValue_factory = None

    @classmethod
    def setUpClass(self):
        print(f"Running tests in {__file__} in process with id: {os.getpid()}")
        self.sdk = SDK()
        self.sdk.load(addExampleSearchPath=False, loadImagePlugins=False)
        self.tf = self.sdk.mdlFactory.create_type_factory(self.sdk.transaction)
        self.vf = self.sdk.mdlFactory.create_value_factory(self.sdk.transaction)

    @classmethod
    def tearDownClass(self):
        self.vf = None
        self.tf = None
        self.sdk.unload()
        self.sdk = None
        print(f"\nFinished tests in {__file__}\n")

    def test_unique_type_iids(self):
        for className, classObj in inspect.getmembers(pymdlsdk):
            type_dict = {}
            if inspect.isclass(classObj):
                if className.startswith('I'):
                    hasIID: bool = False
                    for funcName, funcObj in inspect.getmembers(classObj):
                        if funcName == "IID":
                            iid: str = str(funcObj())
                            # print(f"class: {className} Type-IID: {iid}")
                            hasIID = True
                            self.assertFalse(iid in type_dict)
                            type_dict[iid] = classObj
                    self.assertTrue(hasIID, f"Type: {className} has no IDD")


# run all tests of this file
if __name__ == '__main__':
    unittest.main()  # pragma: no cover
