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


# Test basic features of the binding logic it self
class MainConfiguration(UnittestBase):
    sdk: SDK = None
    tf: pymdlsdk.IType_factory = None

    @classmethod
    def setUpClass(self):
        print(f"Running tests in {__file__} in process with id: {os.getpid()}")
        self.sdk = SDK()
        self.sdk.load(addExampleSearchPath=True, addExampleResourcePath=True)

    @classmethod
    def tearDownClass(self):
        self.tf = None
        self.sdk.unload()
        self.sdk = None
        print(f"\nFinished tests in {__file__}\n")

    # Returned IInterfaces can be nullptr in C++. They are returned as invalid interfaces in python.
    # Meaning the returned object is NOT None.
    def test_search_paths(self):
        cfg: pymdlsdk.IMdl_configuration = self.sdk.neuray.get_api_component(pymdlsdk.IMdl_configuration)

        # system wide installation for all users
        systemCount: int = cfg.get_mdl_system_paths_length()
        for i in range(systemCount):
            print(f"system path: {cfg.get_mdl_system_path(i)}")
        
        # user specific paths
        userCount: int = cfg.get_mdl_user_paths_length()
        for i in range(userCount):
            print(f"user path: {cfg.get_mdl_user_path(i)}")

        # actual search paths and resource paths set
        # Note, this only contains paths that exist on disk while the two above might not exists
        spCount: int = cfg.get_mdl_paths_length()
        for i in range(spCount):
            print(f"search path: {cfg.get_mdl_path(i).get_c_str()}")
        resCount: int = cfg.get_resource_paths_length()
        for i in range(resCount):
            print(f"resource path: {cfg.get_resource_path(i).get_c_str()}")
        
        # the last one should be the example search path because the `setup` added it last
        examplePath: str = cfg.get_resource_path(cfg.get_resource_paths_length()-1).get_c_str()
        self.assertNotNullOrEmpty(examplePath)
        # remove the example path
        cfg.remove_mdl_path(examplePath)
        self.assertEqual(cfg.get_mdl_paths_length(), spCount - 1)
        cfg.remove_resource_path(examplePath)
        self.assertEqual(cfg.get_resource_paths_length(), resCount - 1)

        # remove all paths
        cfg.clear_mdl_paths()
        self.assertZero(cfg.get_mdl_paths_length())
        cfg.clear_resource_paths()
        self.assertZero(cfg.get_resource_paths_length())

    def test_options(self):
        cfg: pymdlsdk.IMdl_configuration = self.sdk.neuray.get_api_component(pymdlsdk.IMdl_configuration)

        self.assertEqual(cfg.get_material_ior_frequency(), pymdlsdk.IType.Modifier.MK_VARYING)
        self.assertEqual(cfg.set_material_ior_frequency(pymdlsdk.IType.Modifier.MK_UNIFORM), -1)  # not allowed after starting neuray

        self.assertEqual(cfg.get_implicit_cast_enabled(), True)
        self.assertEqual(cfg.set_implicit_cast_enabled(False), -1)  # not allowed after starting neuray

        self.assertEqual(cfg.get_expose_names_of_let_expressions(), False)
        self.assertEqual(cfg.set_expose_names_of_let_expressions(True), -1)  # not allowed after starting neuray

        self.assertEqual(cfg.get_simple_glossy_bsdf_legacy_enabled(), False)
        self.assertEqual(cfg.set_simple_glossy_bsdf_legacy_enabled(True), -1)  # not allowed after starting neuray

    def test_entity_resolver(self):
        cfg: pymdlsdk.IMdl_configuration = self.sdk.neuray.get_api_component(pymdlsdk.IMdl_configuration)

        er: pymdlsdk.IMdl_entity_resolver = cfg.get_entity_resolver()
        self.assertIsValidInterface(er)
        cfg.set_entity_resolver(None)  # not possible to set a custom entity resolver implemented in python yet
                                       # setting uninstalls a potentially set ER 
    

    def test_context_options(self):
        context: pymdlsdk.IMdl_execution_context = self.sdk.mdlFactory.create_execution_context()

        name: str = "fold_all_bool_parameters"
        self.assertZero(context.set_option(name, True))
        self.assertEqual(context.get_option_type(name), "Boolean")
        self.assertEqual(context.get_option(name), True)
        retCode: pymdlsdk.ReturnCode = pymdlsdk.ReturnCode()
        self.assertEqual(context.get_option(name, retCode), True)
        self.assertEqual(retCode.value, 0)
 
        name = "meters_per_scene_unit"
        self.assertZero(context.set_option(name, 100.0))
        self.assertEqual(context.get_option_type(name), "Float32")
        self.assertAlmostEqual(context.get_option(name), 100.0)
        retCode = pymdlsdk.ReturnCode()
        self.assertAlmostEqual(context.get_option(name, retCode), 100.0)
        self.assertEqual(retCode.value, 0)

        name = "optimization_level"
        self.assertZero(context.set_option(name, 1))
        self.assertEqual(context.get_option_type(name), "Sint32")
        self.assertEqual(context.get_option(name), 1)
        retCode = pymdlsdk.ReturnCode()
        self.assertEqual(context.get_option(name, retCode), 1)
        self.assertEqual(retCode.value, 0)

        name = "internal_space"
        self.assertZero(context.set_option(name, "coordinate_world"))
        self.assertEqual(context.get_option_type(name), "String")
        self.assertEqual(context.get_option(name), "coordinate_world")
        retCode = pymdlsdk.ReturnCode()
        self.assertEqual(context.get_option(name, retCode), "coordinate_world")
        self.assertEqual(retCode.value, 0)

        name = "target_type"
        self.load_module("::base")  # load at least one module in order so see the predefined types
        tf: pymdlsdk.IType_factory = self.sdk.mdlFactory.create_type_factory(self.sdk.transaction)
        tMaterial: pymdlsdk.IType_structure = tf.get_predefined_struct(pymdlsdk.IType_structure.Predefined_id.SID_MATERIAL)
        self.assertIsValidInterface(tMaterial)
        self.assertZero(context.set_option(name, tMaterial))
        self.assertEqual(context.get_option_type(name), "IInterface")
        optionInterface: pymdlsdk.IInterface = context.get_option(name)
        self.assertIsValidInterface(optionInterface)
        optionTargetMaterial: pymdlsdk.IType_structure = optionInterface.get_interface(pymdlsdk.IType_structure)
        self.assertEqual(optionTargetMaterial.get_predefined_id(), pymdlsdk.IType_structure.Predefined_id.SID_MATERIAL)
        retCode = pymdlsdk.ReturnCode()
        optionInterface: pymdlsdk.IInterface = context.get_option(name, retCode)
        self.assertEqual(retCode.value, 0)
        self.assertIsValidInterface(optionInterface)
        optionTargetMaterial: pymdlsdk.IType_structure = optionInterface.get_interface(pymdlsdk.IType_structure)
        self.assertEqual(optionTargetMaterial.get_predefined_id(), pymdlsdk.IType_structure.Predefined_id.SID_MATERIAL)

        for i in range(context.get_option_count()):
            name: str = context.get_option_name(i)
            type: str = context.get_option_type(name)
            self.assertNotEqual(type, "None")
            print(f"option: '{name}' of type '{type}'")

# run all tests of this file
if __name__ == '__main__':
    unittest.main()  # pragma: no cover
