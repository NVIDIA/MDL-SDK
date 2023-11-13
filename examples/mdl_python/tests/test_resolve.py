import unittest
import os

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


class MainResolve(UnittestBase):
    sdk: SDK = None

    @classmethod
    def setUpClass(self):
        print(f"Running tests in {__file__} in process with id: {os.getpid()}")
        self.sdk = SDK()
        self.sdk.load(addExampleSearchPath=True, loadImagePlugins=False)

    @classmethod
    def tearDownClass(self):
        self.sdk.unload()
        self.sdk = None
        print(f"\nFinished tests in {__file__}\n")

    def test_load_module_success(self):
        self.assertNotEqual(self.load_module("::df"), "")  # note, don't load modules like this in an application!

    def test_load_module_failure(self):
        self.assertEqual(self.load_module("::NOT_EXISTING"), "")

    def UPCOMING_test_resolve_resource_absolute(self):  # pragma: no cover
        config: pymdlsdk.IMdl_configuration = self.sdk.neuray.get_api_component(pymdlsdk.IMdl_configuration)
        self.assertIsNotNone(config)
        er: pymdlsdk.IMdl_entity_resolver = config.get_entity_resolver()
        self.assertIsNotNone(er)
        resourcePath: str = "/nvidia/sdk_examples/resources/example_roughness.png"
        resolved_resource: pymdlsdk.IMdl_resolved_resource = er.resolve_resource(
            file_path=resourcePath, owner_file_path=None, owner_name=None, pos_line=0, pos_column=0, context=None)
        self.assertTrue(resolved_resource.is_valid_interface())
        self.assertEqual(resolved_resource.get_count(), 1)  # not animated
        filename_mask: str = resolved_resource.get_filename_mask().replace("\\", "/")
        self.assertTrue(filename_mask.endswith(resourcePath))
        self.assertTrue(len(filename_mask) > len(resourcePath))
        frame0: pymdlsdk.IMdl_resolved_resource_element = resolved_resource.get_element(0)
        self.assertEqual(frame0.get_count(), 1)  # not tiled
        self.assertEqual(frame0.get_mdl_file_path(0), resourcePath)  # same for not animated and not tiled
        filename_frame0_tile0: str = frame0.get_filename(0).replace("\\", "/")
        self.assertEqual(filename_frame0_tile0, filename_mask)  # same for not animated and not tiled

    def UPCOMING_resolve_module(self, mdlName) -> str:  # pragma: no cover
        config: pymdlsdk.IMdl_configuration = self.sdk.neuray.get_api_component(pymdlsdk.IMdl_configuration)
        self.assertIsNotNone(config)
        er: pymdlsdk.IMdl_entity_resolver = config.get_entity_resolver()
        self.assertIsNotNone(er)
        resolved_module: pymdlsdk.IMdl_resolved_module = er.resolve_module(mdlName, None, None, 0, 0, None)
        self.assertTrue(resolved_module.is_valid_interface())
        moduleFileName = resolved_module.get_filename()
        self.assertTrue(moduleFileName.replace("\\", "/").endswith(mdlName.replace("::", "/") + ".mdl"))
        return moduleFileName

    def UPCOMING_test_resolve_module(self):  # pragma: no cover
        self.assertTrue(len(self.resolve_module("::nvidia::sdk_examples::tutorials")) > 0)

    def UPCOMING_test_resolve_resource_relative(self):  # pragma: no cover
        config: pymdlsdk.IMdl_configuration = self.sdk.neuray.get_api_component(pymdlsdk.IMdl_configuration)
        self.assertIsNotNone(config)
        er: pymdlsdk.IMdl_entity_resolver = config.get_entity_resolver()
        self.assertIsNotNone(er)
        resourcePath: str = "resources/metal_cast_iron_roughness.png"
        ownerName: str = "::nvidia::sdk_examples::tutorials_distilling"
        ownerFilePath: str = self.resolve_module(ownerName)
        resolved_resource: pymdlsdk.IMdl_resolved_resource = er.resolve_resource(
            file_path=resourcePath, owner_file_path=ownerFilePath, owner_name=ownerName, pos_line=0, pos_column=0, context=None)
        self.assertTrue(resolved_resource.is_valid_interface())
        self.assertEqual(resolved_resource.get_count(), 1)  # not animated
        filename_mask: str = resolved_resource.get_filename_mask().replace("\\", "/")
        self.assertTrue(filename_mask.endswith(resourcePath))
        self.assertTrue(len(filename_mask) > len(resourcePath))
        frame0: pymdlsdk.IMdl_resolved_resource_element = resolved_resource.get_element(0)
        self.assertEqual(frame0.get_count(), 1)  # not tiled
        self.assertEqual(frame0.get_mdl_file_path(0), "/nvidia/sdk_examples/" + resourcePath)  # same for not animated and not tiled
        filename_frame0_tile0: str = frame0.get_filename(0).replace("\\", "/")
        self.assertEqual(filename_frame0_tile0, filename_mask)  # same for not animated and not tiled


# run all tests of this file
if __name__ == '__main__':
    unittest.main()  # pragma: no cover
