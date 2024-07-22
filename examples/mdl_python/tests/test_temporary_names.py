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


class MainModulesTempNamesMdl(UnittestBase):
    # We have this test suite just to enable temporary/let names. For other tests we want
    # the default behavior. Since this feature needs to be enabled before we start neuray, 
    # it makes sense to factor it out into a separate class.

    sdk: SDK = None

    @classmethod
    def setUpClass(self):
        print(f"Running tests in {__file__} in process with id: {os.getpid()}")
        self.sdk = SDK()
        self.sdk.load(addExampleSearchPath=True, loadImagePlugins=True, enableTemporaryNames=True)

    @classmethod
    def tearDownClass(self):
        self.sdk.unload()
        self.sdk = None
        print(f"\nFinished tests in {__file__}\n")

    def test_temporary_names(self):
        moduleQualifier: str = "::nvidia::sdk_examples::tutorials"
        moduleDbName: str = self.load_module(moduleQualifier)
        with self.sdk.transaction.access_as(pymdlsdk.IModule, moduleDbName) as mdlModule:
            for i in range(mdlModule.get_function_count()):
                fDbName: str = mdlModule.get_function(i)
                fDefinition: pymdlsdk.IFunction_definition = self.sdk.transaction.access_as(pymdlsdk.IFunction_definition, fDbName)
                for i in range(fDefinition.get_temporary_count()):
                    temp: pymdlsdk.IExpression = fDefinition.get_temporary(i)
                    self.assertIsValidInterface(temp)
                    if fDefinition.get_temporary_name(i) is not None:
                        self.assertNotEqual(fDefinition.get_temporary_name(i), "")
                    temp = None

# run all tests of this file
if __name__ == '__main__':
    unittest.main()  # pragma: no cover
