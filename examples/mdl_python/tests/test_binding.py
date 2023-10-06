import unittest
import pymdlsdk
from setup import SDK

# Test basic features of the binding logic it self

class Main(unittest.TestCase):
    sdk: SDK = None

    @classmethod
    def setUpClass(self):
        print(f"Running tests in {__file__}")
        self.sdk = SDK()
        self.sdk.load(addExampleSearchPath=False, loadImagePlugins=False)

    @classmethod
    def tearDownClass(self):
        self.sdk.unload()
        self.sdk = None
        print(f"\nFinished tests in {__file__}\n")

    # Returned IInterfaces can be nullptr in C++. They are returned as invalid interfaces in python.
    # Meaning the returned object is NOT None.
    def test_with_nullptr(self):
        config: pymdlsdk.IMdl_configuration = self.sdk.neuray.get_api_component(pymdlsdk.IMdl_configuration)
        er: pymdlsdk.IMdl_entity_resolver = config.get_entity_resolver()
        # file will not be resolved
        resolved_resource: pymdlsdk.IMdl_resolved_resource
        with er.resolve_resource(file_path="/DOES_NOT_EXIST.png", owner_file_path=None, owner_name=None, pos_line=0, pos_column=0) as resolved_resource:
            self.assertIsNotNone(resolved_resource)  # returned object exists
            self.assertFalse(resolved_resource.is_valid_interface())  # but the interface is not valid

    # Accessing, i.e. calling, invalid IInterfaces should throw an Exception rather than crashing. 
    def test_nullptr_access(self):
        config: pymdlsdk.IMdl_configuration = self.sdk.neuray.get_api_component(pymdlsdk.IMdl_configuration)
        er: pymdlsdk.IMdl_entity_resolver = config.get_entity_resolver()
        # file will not be resolved
        resolved_resource: pymdlsdk.IMdl_resolved_resource = er.resolve_resource(
            file_path="/DOES_NOT_EXIST.png", owner_file_path=None, owner_name=None, pos_line=0, pos_column=0)
        self.assertIsNotNone(resolved_resource)  # returned object exists
        self.assertFalse(resolved_resource.is_valid_interface())  # but the interface is not valid
        try:
            resolved_resource.get_count()
            self.assertTrue(False, "expected exception not fired")  # should not be reached
        except RuntimeError as ex:
            print(f"expected error: {ex}")
        except Exception as ex:  # would catch if we do not catch `RuntimeError` before
            print(f"exception: {ex}")
            self.assertTrue(False, "expected exception of wrong type")  # should not be reached

        resolved_resource_2: pymdlsdk.IMdl_resolved_resource
        with er.resolve_resource(file_path="/DOES_NOT_EXIST.png", owner_file_path=None, owner_name=None, pos_line=0, pos_column=0) as resolved_resource_2:
            self.assertIsNotNone(resolved_resource_2)  # returned object exists
            self.assertFalse(resolved_resource_2.is_valid_interface())  # but the interface is not valid


# run all tests of this file
if __name__ == '__main__':
    unittest.main()
