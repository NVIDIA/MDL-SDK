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

import warnings

class MainDatabase(UnittestBase):
    sdk: SDK = None

    @classmethod
    def setUpClass(self):
        print(f"Running tests in {__file__} in process with id: {os.getpid()}")
        self.sdk = SDK()
        self.sdk.load(addExampleSearchPath=False, loadImagePlugins=False)

    @classmethod
    def tearDownClass(self):
        self.sdk.unload()
        self.sdk = None
        print(f"\nFinished tests in {__file__}\n")

    def test_setupIsDone(self):
        self.assertIsNotNone(self.sdk)
        self.assertIsValidInterface(self.sdk.neuray)
        self.assertEqual(self.sdk.neuray.get_status(), pymdlsdk.INeuray.Status.STARTED)

    def assertArrayWithOneStringElement(self, array, name):
        self.assertEqual(array.get_length(), 1)
        element0: pymdlsdk.IString = array.get_element_as(pymdlsdk.IString, 0)
        self.assertIsValidInterface(element0)
        self.assertEqual(element0.get_c_str(), name)

    def test_IDatabase(self):
        database = self.sdk.neuray.get_api_component(pymdlsdk.IDatabase)
        self.assertIsValidInterface(database)
        global_scope = database.get_global_scope()
        self.assertIsValidInterface(global_scope)
        scope1_id = 0
        scope2_id = 0
        with database.create_scope(global_scope, 10) as scope1:
            self.assertIsValidInterface(scope1)
            scope1_id = scope1.get_id()
            with database.create_or_get_named_scope("my_scope", scope1, 20) as scope2:
                self.assertIsValidInterface(scope2)
                scope2_id = scope2.get_id()
        with database.get_scope(scope1_id) as scope1:
            self.assertIsValidInterface(scope1)
            self.assertEqual(scope1.get_id(), scope1_id)
        with database.get_named_scope("my_scope") as scope2:
            self.assertIsValidInterface(scope2)
            self.assertEqual(scope2.get_id(), scope2_id)
        database.remove_scope(scope2_id)
        database.remove_scope(scope1_id)
        database.garbage_collection()
        database.garbage_collection(pymdlsdk.IDatabase.Garbage_collection_priority.PRIORITY_LOW)

    def test_IScope(self):
        database = self.sdk.neuray.get_api_component(pymdlsdk.IDatabase)
        global_scope = database.get_global_scope()
        self.assertIsValidInterface(global_scope)
        self.assertEqual(global_scope.get_id(), '0')
        self.assertIsNone(global_scope.get_name())
        self.assertIsNotValidInterface(global_scope.get_parent())
        with database.create_scope(global_scope, 10) as scope1:
            self.assertIsValidInterface(scope1)
            self.assertNotEqual(scope1.get_id(), '0')
            self.assertIsNone(scope1.get_name())
            with scope1.get_parent() as scope1_parent:
                self.assertEqual(scope1_parent.get_id(), global_scope.get_id())
            self.assertEqual(scope1.get_privacy_level(), 10)
            with database.create_or_get_named_scope("my_scope", scope1, 20) as scope2:
                self.assertIsValidInterface(scope2)
                self.assertNotEqual(scope2.get_id(), '0')
                self.assertEqual(scope2.get_name(), "my_scope")
                with scope2.get_parent() as scope2_parent:
                    self.assertEqual(scope2_parent.get_id(), scope1.get_id())
                self.assertEqual(scope2.get_privacy_level(), 20)

    def test_ITransaction(self):
        database = self.sdk.neuray.get_api_component(pymdlsdk.IDatabase)
        global_scope = database.get_global_scope()
        # commit()/is_open()
        with global_scope.create_transaction() as transaction:
            self.assertIsValidInterface(transaction)
            self.assertTrue(transaction.is_open())
            self.assertEqual(transaction.commit(), 0)
            self.assertFalse(transaction.is_open())
        # abort()/is_open()
        with global_scope.create_transaction() as transaction:
            transaction.abort()
            self.assertFalse(transaction.is_open())
        # create()
        with global_scope.create_transaction() as transaction:
            with transaction.create("Image") as image:
                self.assertIsValidInterface(image)
            with transaction.create_as(pymdlsdk.IImage, "Image") as image:
                self.assertIsValidInterface(image)
            with transaction.create_as(pymdlsdk.ITexture, "Image") as image:
                self.assertIsNotValidInterface(image)
            transaction.commit()
        # store()
        with global_scope.create_transaction() as transaction:
            with transaction.create("Image") as image:
                self.assertEqual(transaction.store(image, "image1"), 0)
            with transaction.create("Image") as image:
                self.assertEqual(transaction.store(image, "image2", 255), 0)
            with transaction.create("Image") as image:
               self.assertEqual(transaction.store(
                   image, "image3", pymdlsdk.ITransaction.LOCAL_SCOPE), 0)
            transaction.commit()
        # access()
        with global_scope.create_transaction() as transaction:
            with transaction.access("image1") as image:
                self.assertIsValidInterface(image)
            with transaction.access_as(pymdlsdk.IImage, "image1") as image:
                self.assertIsValidInterface(image)
            with transaction.access_as(pymdlsdk.ITexture, "image1") as image:
                self.assertIsNotValidInterface(image)
            with transaction.access("non-existing") as image:
                self.assertIsNotValidInterface(image)
            with transaction.access_as(pymdlsdk.ITexture, "non-existing") as image:
                self.assertIsNotValidInterface(image)
            with transaction.access(None) as image:
                self.assertIsNotValidInterface(image)
            with transaction.access_as(pymdlsdk.ITexture, None) as image:
                self.assertIsNotValidInterface(image)
            transaction.commit()
        # edit()
        old_time_stamp = ""
        with global_scope.create_transaction() as transaction:
            # get timestamp before edit for later tests
            old_time_stamp = transaction.get_time_stamp()
            with transaction.edit("image1") as image:
                self.assertIsValidInterface(image)
            with transaction.edit_as(pymdlsdk.IImage, "image1") as image:
                self.assertIsValidInterface(image)
            with transaction.edit_as(pymdlsdk.ITexture, "image1") as image:
                self.assertIsNotValidInterface(image)
            with transaction.edit("non-existing") as image:
                self.assertIsNotValidInterface(image)
            with transaction.edit_as(pymdlsdk.ITexture, "non-existing") as image:
                self.assertIsNotValidInterface(image)
            with transaction.edit(None) as image:
                self.assertIsNotValidInterface(image)
            with transaction.edit_as(pymdlsdk.ITexture, None) as image:
                self.assertIsNotValidInterface(image)
            transaction.commit()
        # copy()
        with global_scope.create_transaction() as transaction:
            self.assertEqual(transaction.copy("image1", "image4"), 0)
            self.assertEqual(transaction.copy("image1", "image5", 255), 0)
            self.assertEqual(transaction.copy(
                "image1", "image6", pymdlsdk.ITransaction.LOCAL_SCOPE), 0)
            self.assertEqual(transaction.copy(None, "image4"), -2)
            self.assertEqual(transaction.copy("image1", None), -2)
            self.assertEqual(transaction.copy("non-existing", "image4"), -4)
            transaction.commit()
        # remove()
        with global_scope.create_transaction() as transaction:
            self.assertEqual(transaction.remove("image1"), 0)
            self.assertEqual(transaction.remove("image2", False), 0)
            self.assertEqual(transaction.remove("image4", True), 0)
            self.assertEqual(transaction.remove(None), -2)
            self.assertEqual(transaction.remove("non-existing"), -1)
            transaction.commit()
        # name_of()
        with global_scope.create_transaction() as transaction:
            with transaction.access("image5") as image:
                self.assertEqual(transaction.name_of(image), "image5")
            transaction.commit()
        # get_time_stamp() / has_changed_since_time_stamp()
        with global_scope.create_transaction() as transaction:
            current_time_stamp = transaction.get_time_stamp()
            self.assertNotNullOrEmpty(transaction.get_time_stamp("image5"))
            self.assertTrue(transaction.has_changed_since_time_stamp("image5", old_time_stamp))
            self.assertFalse(transaction.has_changed_since_time_stamp("image5", current_time_stamp))
            transaction.commit()
        # get_id()
        with global_scope.create_transaction() as transaction:
            self.assertNotNullOrEmpty(transaction.get_id())
            transaction.commit()
        # get_scope()
        with global_scope.create_transaction() as transaction:
            scope = transaction.get_scope()
            self.assertEqual(scope.get_id(), global_scope.get_id())
            transaction.commit()
        # list_elements()
        with global_scope.create_transaction() as transaction:
            # prepare type_names array
            type_name = self.sdk.transaction.create_as(pymdlsdk.IString, "String")
            self.assertIsValidInterface(type_name)
            type_name.set_c_str("Image")
            type_names = self.sdk.transaction.create_as(pymdlsdk.IArray, "String[1]")
            self.assertIsValidInterface(type_names)
            self.assertEqual(type_names.set_element(0, type_name), 0)
            # actual test
            result = transaction.list_elements("image5")
            self.assertArrayWithOneStringElement(result, "image5")
            result = transaction.list_elements("image5", "i.*")
            self.assertArrayWithOneStringElement(result, "image5")
            result = transaction.list_elements("image5", "i.*", type_names)
            self.assertArrayWithOneStringElement(result, "image5")
            result = transaction.list_elements("image5", "i.*", None)
            self.assertArrayWithOneStringElement(result, "image5")
            transaction.commit()
        # get_privacy_level()
        with global_scope.create_transaction() as transaction:
            self.assertEqual(transaction.get_privacy_level("image5"), 0)
            transaction.commit()

class MainNeuray(UnittestBase):
    sdk: SDK = None

    @classmethod
    def setUpClass(self):
        print(f"Running tests in {__file__} in process with id: {os.getpid()}")
        self.sdk = SDK()
        self.sdk.load(addExampleSearchPath=False, loadImagePlugins=False)

    @classmethod
    def tearDownClass(self):
        self.sdk.unload()
        self.sdk = None
        print(f"\nFinished tests in {__file__}\n")

    def test_setupIsDone(self):
        self.assertIsNotNone(self.sdk)
        self.assertIsValidInterface(self.sdk.neuray)
        self.assertEqual(self.sdk.neuray.get_status(), pymdlsdk.INeuray.Status.STARTED)

    def test_INeuray(self):
        # get_api_component()
        database = self.sdk.neuray.get_api_component(pymdlsdk.IDatabase)
        self.assertIsValidInterface(database)
        scope = self.sdk.neuray.get_api_component(pymdlsdk.IScope)
        self.assertIsNotValidInterface(scope)
        # get_interface_version()
        interface_version = self.sdk.neuray.get_interface_version()
        self.assertTrue(interface_version >= 50)
        # get_version()
        version = self.sdk.neuray.get_version()
        self.assertEqual(version[:7], "MDL SDK")

# This class does not use the shared load/start/shutdown/unload code in SDK to be able to test
# plugin loading (needs to happen before start()).
class MainPluginConfiguration(UnittestBase):
    sdk: SDK = None

    @classmethod
    def setUpClass(self):
        print(f"Running tests in {__file__} in process with id: {os.getpid()}")

    @classmethod
    def tearDownClass(self):
        self.neuray = None
        pymdlsdk._print_open_handle_statistic()
        if not pymdlsdk.unload():
            raise Exception('Failed to unload the SDK.')  # pragma: no cover
        print(f"\nFinished tests in {__file__}\n")

    def test_IPlugin_configuration(self):
        self.neuray = pymdlsdk.load_and_get_ineuray('')
        if not self.neuray.is_valid_interface():
            raise Exception('Failed to load the MDL SDK.')  # pragma: no cover

        plugin_configuration = self.neuray.get_api_component(pymdlsdk.IPlugin_configuration)
        self.assertIsValidInterface(plugin_configuration)
        mi_base_dll_file_ext = ".dll" if os.name == "nt" else ".so"
        self.assertEqual(
            plugin_configuration.load_plugin_library('nv_openimageio' + mi_base_dll_file_ext), 0)
        self.assertTrue(plugin_configuration.get_plugin_length() >= 10)

# run all tests of this file
if __name__ == '__main__':
    unittest.main()  # pragma: no cover
