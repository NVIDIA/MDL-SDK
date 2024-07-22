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


class MainIData(UnittestBase):
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
        self.assertIsValidInterface(self.sdk.transaction)

    def test_IString(self):
        i: pymdlsdk.IInterface = self.assertDeprecationWarning(lambda: self.sdk.transaction.create("String", 0, None))  # old signature
        self.assertIsValidInterface(i)  # still returns an IInterface
        i = self.sdk.transaction.create("String")                      # using 'create' regularly
        self.assertIsValidInterface(i)
        a: pymdlsdk.IString = i.get_interface(pymdlsdk.IString)        # needs casting
        self.assertIsValidInterface(a)
        value: str = a.get_c_str()
        self.assertEqual(value, "")
        value = "MDL"
        a.set_c_str(value)
        value2: str = a.get_c_str()
        self.assertEqual(value, value2)

    def test_ISize(self):
        a: pymdlsdk.ISize = self.assertDeprecationWarning(lambda: self.sdk.transaction.create_as(pymdlsdk.ISize, "Size", 0, None))
        self.assertIsValidInterface(a)
        a = self.sdk.transaction.create_as(pymdlsdk.ISize, "Size", [])  # create as with no constructor arguments
        self.assertIsValidInterface(a)
        a = self.sdk.transaction.create_as(pymdlsdk.ISize, "Size")      # using 'create' regularly
        self.assertIsValidInterface(a)
        value: int = a.get_value()
        self.assertEqual(value, 0)
        value = 12345
        a.set_value(value)
        self.assertEqual(value, a.get_value())

    def test_IFloat32(self):
        a: pymdlsdk.IFloat32 = self.sdk.transaction.create_as(pymdlsdk.IFloat32, "Float32")
        self.assertIsValidInterface(a)
        value: float = a.get_value()
        self.assertEqual(value, 0)
        value = -1.2
        a.set_value(value)
        self.assertAlmostEqual(value, a.get_value())  # floating point off by 0.0000000476837158

    def test_IUint8(self):
        a: pymdlsdk.IUint8 = self.sdk.transaction.create_as(pymdlsdk.IUint8, "Uint8")
        self.assertIsValidInterface(a)
        value: int = a.get_value()
        self.assertEqual(value, 0)
        value = 234
        a.set_value(value)
        self.assertEqual(value, a.get_value())

    def test_IBoolean(self):
        a: pymdlsdk.IBoolean = self.sdk.transaction.create_as(pymdlsdk.IBoolean, "Boolean")
        self.assertIsValidInterface(a)
        value: bool = a.get_value()
        self.assertEqual(value, False)
        value = True
        a.set_value(value)
        self.assertEqual(value, a.get_value())

    def test_IColor(self):
        a: pymdlsdk.IColor = self.sdk.transaction.create_as(pymdlsdk.IColor, "Color")
        self.assertIsValidInterface(a)
        value: pymdlsdk.Color_struct = a.get_value()
        self.assertEqual(value.r, 0)
        self.assertEqual(value.g, 0)
        self.assertEqual(value.b, 0)
        self.assertEqual(value.a, 0)
        value.r = 0.25
        value.g = 0.5
        value.b = 0.75
        value.a = 1.0
        a.set_value(value)
        value2 = a.get_value()
        self.assertEqual(value.r, value2.r)
        self.assertEqual(value.g, value2.g)
        self.assertEqual(value.b, value2.b)
        self.assertEqual(value.a, value2.a)

    def test_IColor3(self):
        a: pymdlsdk.IColor3 = self.sdk.transaction.create_as(pymdlsdk.IColor3, "Color3")
        self.assertIsValidInterface(a)
        value: pymdlsdk.Color_struct = a.get_value()
        self.assertEqual(value.r, 0)
        self.assertEqual(value.g, 0)
        self.assertEqual(value.b, 0)
        self.assertEqual(value.a, 1)
        value.r = 0.25
        value.g = 0.5
        value.b = 0.75
        a.set_value(value)
        value2 = a.get_value()
        self.assertEqual(value.r, value2.r)
        self.assertEqual(value.g, value2.g)
        self.assertEqual(value.b, value2.b)

    def test_IFloat32_3(self):
        a: pymdlsdk.IFloat32_3 = self.sdk.transaction.create_as(pymdlsdk.IFloat32_3, "Float32<3>")
        self.assertIsValidInterface(a)
        value: pymdlsdk.Float32_3_struct = a.get_value()
        self.assertEqual(value.x, 0)
        self.assertEqual(value.y, 0)
        self.assertEqual(value.z, 0)
        value.x = 1.2
        value.y = 3.4
        value.z = -5.6
        a.set_value(value)
        value2 = a.get_value()
        self.assertEqual(value.x, value2.x)
        self.assertEqual(value.y, value2.y)
        self.assertEqual(value.z, value2.z)

    def test_IFloat64_2(self):
        a: pymdlsdk.IFloat64_2 = self.sdk.transaction.create_as(pymdlsdk.IFloat64_2, "Float64<2>")
        self.assertIsValidInterface(a)
        value: pymdlsdk.Float64_2_struct = a.get_value()
        self.assertEqual(value.x, 0)
        self.assertEqual(value.y, 0)
        value.x = 1.2
        value.y = 3.4
        a.set_value(value)
        value2 = a.get_value()
        self.assertEqual(value.x, value2.x)
        self.assertEqual(value.y, value2.y)

    def test_IBoolean_4(self):
        a: pymdlsdk.IBoolean_4 = self.sdk.transaction.create_as(pymdlsdk.IBoolean_4, "Boolean<4>")
        self.assertIsValidInterface(a)
        value: pymdlsdk.Boolean_4_struct = a.get_value()
        self.assertFalse(value.x)
        self.assertFalse(value.y)
        self.assertFalse(value.z)
        self.assertFalse(value.w)
        value.x = True
        value.y = False
        value.z = True
        value.w = False
        a.set_value(value)
        value2 = a.get_value()
        self.assertEqual(value.x, value2.x)
        self.assertEqual(value.y, value2.y)
        self.assertEqual(value.z, value2.z)
        self.assertEqual(value.w, value2.w)

    def test_ISint32_2_3(self):
        a: pymdlsdk.ISint32_2_3 = self.sdk.transaction.create_as(pymdlsdk.ISint32_2_3, "Sint32<2,3>")
        self.assertIsValidInterface(a)
        value = a.get_value()
        self.assertEqual(value.xx, 0)
        self.assertEqual(value.xy, 0)
        self.assertEqual(value.xz, 0)
        self.assertEqual(value.yx, 0)
        self.assertEqual(value.yy, 0)
        self.assertEqual(value.yz, 0)
        value.xx = -11
        value.xy = 12
        value.xz = -13
        value.yx = 14
        value.yy = -15
        value.yz = 16
        a.set_value(value)
        value2 = a.get_value()
        self.assertEqual(value.xx, value2.xx)
        self.assertEqual(value.xy, value2.xy)
        self.assertEqual(value.xz, value2.xz)
        self.assertEqual(value.yx, value2.yx)
        self.assertEqual(value.yy, value2.yy)
        self.assertEqual(value.yz, value2.yz)

    def test_IFloat32_3_3(self):
        a: pymdlsdk.IFloat32_3_3 = self.sdk.transaction.create_as(pymdlsdk.IFloat32_3_3, "Float32<3,3>")
        self.assertIsValidInterface(a)
        value = a.get_value()
        self.assertEqual(value.xx, 0)
        self.assertEqual(value.xy, 0)
        self.assertEqual(value.xz, 0)
        self.assertEqual(value.yx, 0)
        self.assertEqual(value.yy, 0)
        self.assertEqual(value.yz, 0)
        self.assertEqual(value.zx, 0)
        self.assertEqual(value.zy, 0)
        self.assertEqual(value.zz, 0)
        value.xx = -11.0
        value.xy = 12.0
        value.xz = -13.0
        value.yx = 14.0
        value.yy = -15.0
        value.yz = 16.0
        value.zx = -17.0
        value.zy = 18.0
        value.zz = -19.0
        a.set_value(value)
        value2 = a.get_value()
        self.assertEqual(value.xx, value2.xx)
        self.assertEqual(value.xy, value2.xy)
        self.assertEqual(value.xz, value2.xz)
        self.assertEqual(value.yx, value2.yx)
        self.assertEqual(value.yy, value2.yy)
        self.assertEqual(value.yz, value2.yz)
        self.assertEqual(value.zx, value2.zx)
        self.assertEqual(value.zy, value2.zy)
        self.assertEqual(value.zz, value2.zz)

    def run_IData_collection_tests(self, collectionClassType, collection: pymdlsdk.IData_collection):
        self.assertIsValidInterface(collection)
        base: pymdlsdk.IData_collection
        cast_back: pymdlsdk.IData_collection
        with collection.get_interface(pymdlsdk.IData_collection) as base, \
             base.get_interface(collectionClassType) as cast_back:
            self.assertIsValidInterface(base)
            self.assertIsValidInterface(cast_back)
            if collectionClassType != pymdlsdk.IData_collection:  # because its actually some specialization
                self.assertEqual(collection.get_iid(), collectionClassType.IID())
                self.assertTrue(collection.compare_iid(base.get_iid()))
            self.assertEqual(collection.get_iid(), cast_back.get_iid())
        self.assertTrue(collection.compare_iid(pymdlsdk.IData_collection.IID()))
        length: int = collection.get_length()
        if collection.get_type_name() == "Float32[4]":
            self.assertEqual(length, 4)
            keyToTest: str = '1'
            indexToTest: int = 1
            elementType = pymdlsdk.IFloat32
            testValue: pymdlsdk.IFloat32 = self.sdk.transaction.create_as(pymdlsdk.IFloat32, "Float32")
            testValue.set_value(3.1415)
        elif collection.get_type_name() == "Map<Sint32>":
            keyToTest: str = 'forty two'
            for i in range(collection.get_length()):
                if collection.get_key(i) == keyToTest:
                    indexToTest: int = i
                    break
            elementType = pymdlsdk.ISint32
            testValue: pymdlsdk.ISint32 = self.sdk.transaction.create_as(pymdlsdk.ISint32, "Sint32")
            testValue.set_value(24)
        elif collection.get_type_name() == "Uvtile":
            keyToTest = "u"
            for i in range(collection.get_length()):
                if collection.get_key(i) == keyToTest:
                    indexToTest: int = i
                    break
            elementType = pymdlsdk.ISint32
            testValue: pymdlsdk.ISint32 = self.sdk.transaction.create_as(pymdlsdk.ISint32, "Sint32")
            testValue.set_value(13)
        elif collection.get_type_name() == "String[]":
            keyToTest: str = '1'
            indexToTest: int = 1
            elementType = pymdlsdk.IString
            testValue: pymdlsdk.IString = self.sdk.transaction.create_as(pymdlsdk.IString, "String")
            testValue.set_c_str('blub')
        else:
            self.assertTrue(False)  # pragma: no cover

        self.assertTrue(collection.has_key(keyToTest))
        self.assertEqual(collection.get_key(indexToTest), keyToTest)
        valueA = collection.get_value(keyToTest).get_interface(elementType)
        valueB = collection.get_value(indexToTest).get_interface(elementType)
        if elementType == pymdlsdk.IString:
            self.assertEqual(valueA.get_c_str(), valueB.get_c_str())
        else:
            self.assertEqual(valueA.get_value(), valueB.get_value())
        self.assertEqual(0, collection.set_value(keyToTest, testValue))
        self.assertEqual(0, collection.set_value(indexToTest, testValue))
        valueC = collection.get_value_as(elementType, indexToTest)
        valueD = collection.get_value_as(elementType, keyToTest)
        if elementType == pymdlsdk.IString:
            self.assertEqual(testValue.get_c_str(), valueC.get_c_str())
            self.assertEqual(testValue.get_c_str(), valueD.get_c_str())
        else:
            self.assertEqual(testValue.get_value(), valueC.get_value())
            self.assertEqual(testValue.get_value(), valueD.get_value())
        valueAPtr: int = valueA.__iinterface_ptr_as_uint64__()
        valueBPtr: int = valueB.__iinterface_ptr_as_uint64__()
        valueCPtr: int = valueC.__iinterface_ptr_as_uint64__()
        valueDPtr: int = valueD.__iinterface_ptr_as_uint64__()
        testValuePtr: int = testValue.__iinterface_ptr_as_uint64__()
        self.assertEqual(valueAPtr, valueBPtr)
        self.assertEqual(valueCPtr, valueDPtr)
        self.assertNotEqual(valueAPtr, valueCPtr)
        self.assertEqual(valueCPtr, testValuePtr)
        wrongType: pymdlsdk.IMaterial_instance = collection.get_value_as(pymdlsdk.IMaterial_instance, keyToTest)  # wrong type
        self.assertIsNotNone(wrongType)
        self.assertFalse(wrongType.is_valid_interface())
        wrongKeyOrIndex = collection.get_value_as(elementType, "SOMETHING_NOT_EXISTING")
        self.assertIsNotNone(wrongKeyOrIndex)
        self.assertFalse(wrongKeyOrIndex.is_valid_interface())
        with self.assertRaises(IndexError):
            wrongKey = collection.get_value(3.14)
        with self.assertRaises(IndexError):
            collection.set_value(3.14, valueA)

    def test_IArray(self):
        array: pymdlsdk.IArray = self.sdk.transaction.create_as(pymdlsdk.IArray, "Float32[4]")
        collection: pymdlsdk.IData_collection = array.get_interface(pymdlsdk.IData_collection)
        self.run_IData_collection_tests(pymdlsdk.IArray, array)
        self.run_IData_collection_tests(pymdlsdk.IData_collection, collection)  # also test base class functions

        data: pymdlsdk.IData = array.get_interface(pymdlsdk.IData)
        self.assertIsValidInterface(data)
        self.assertEqual(data.get_type_name(), "Float32[4]")

        self.assertFalse(array.empty())
        element0: pymdlsdk.IFloat32 = array.get_element_as(pymdlsdk.IFloat32, 0)
        element1: pymdlsdk.IFloat32 = array.get_element(1).get_interface(pymdlsdk.IFloat32)
        element2: pymdlsdk.IFloat32 = array.get_element(2).get_interface(pymdlsdk.IFloat32)
        element3: pymdlsdk.IFloat32 = array.get_element(3).get_interface(pymdlsdk.IFloat32)
        self.assertTrue(element0.is_valid_interface(), "Invalid IInterface")
        self.assertTrue(element1.is_valid_interface(), "Invalid IInterface")
        self.assertTrue(element2.is_valid_interface(), "Invalid IInterface")
        self.assertTrue(element3.is_valid_interface(), "Invalid IInterface")
        self.assertEqual(element0.get_value(), 0)
        self.assertNotEqual(element1.get_value(), 0)  # something around 3.1415
        self.assertEqual(element2.get_value(), 0)
        self.assertEqual(element3.get_value(), 0)
        element0.set_value(-1.0)
        element1.set_value(2.0)
        element2.set_value(-3.0)
        element3.set_value(-4.0)
        array.set_element(0, element0)
        array.set_element(1, element1)
        array.set_element(2, element2)
        array.set_element(3, element3)
        self.assertEqual(element0.get_value(), array.get_element(0).get_interface(pymdlsdk.IFloat32).get_value())
        self.assertEqual(element1.get_value(), array.get_element_as(pymdlsdk.IFloat32, 1).get_value())
        self.assertEqual(element2.get_value(), array.get_element_as(pymdlsdk.IFloat32, 2).get_value())
        self.assertEqual(element3.get_value(), array.get_element_as(pymdlsdk.IFloat32, 3).get_value())
        wrong_element_type: pymdlsdk.ISint32 = array.get_element_as(pymdlsdk.ISint32, 3)  # wrong type
        self.assertIsNotNone(wrong_element_type)
        self.assertFalse(wrong_element_type.is_valid_interface())
        wrong_index: pymdlsdk.IFloat32 = array.get_element_as(pymdlsdk.IFloat32, 4)
        self.assertIsNotNone(wrong_index)
        self.assertFalse(wrong_index.is_valid_interface())

    def test_IMap(self):
        map: pymdlsdk.IMap = self.sdk.transaction.create_as(pymdlsdk.IMap, "Map<Sint32>")
        self.assertTrue(map.empty())
        self.assertEqual(0, map.insert('forty two', self.sdk.transaction.create_as(pymdlsdk.ISint32, "Sint32")))
        map.get_value_as(pymdlsdk.ISint32, 'forty two').set_value(42)
        self.assertEqual(0, map.insert('forty three', self.sdk.transaction.create_as(pymdlsdk.ISint32, "Sint32")))
        map.get_value_as(pymdlsdk.ISint32, 'forty three').set_value(43)
        self.assertEqual(0, map.insert('', self.sdk.transaction.create_as(pymdlsdk.ISint32, "Sint32")))
        self.assertFalse(map.empty())
        self.run_IData_collection_tests(pymdlsdk.IMap, map)
        collection: pymdlsdk.IData_collection = map.get_interface(pymdlsdk.IData_collection)
        self.run_IData_collection_tests(pymdlsdk.IData_collection, collection)  # also test base class functions

        self.assertEqual(0, map.erase('forty three'))
        self.assertNotEqual(0, map.erase('forty four'))
        self.assertEqual(2, map.get_length())
        self.assertNotEqual(0, map.insert('', self.sdk.transaction.create_as(pymdlsdk.ISint32, "Sint32")))
        self.assertEqual(2, map.get_length())
        map.clear()
        self.assertEqual(0, map.get_length())
        self.assertTrue(map.empty())

    def test_IStructure(self):
        uvtile: pymdlsdk.IStructure = self.sdk.transaction.create_as(pymdlsdk.IStructure, "Uvtile")
        self.run_IData_collection_tests(pymdlsdk.IStructure, uvtile)
        collection: pymdlsdk.IData_collection = uvtile.get_interface(pymdlsdk.IData_collection)
        self.run_IData_collection_tests(pymdlsdk.IData_collection, collection)  # also test base class functions

        decl: pymdlsdk.IStructure_decl = uvtile.get_structure_decl()
        self.assertIsValidInterface(decl)
        self.assertEqual(decl.get_structure_type_name(), 'Uvtile')
        self.assertEqual(decl.get_length(), 4)
        self.assertEqual("frame", decl.get_member_name(2))
        self.assertEqual(decl.get_member_type_name(2), "Size")
        self.assertEqual(decl.get_member_type_name("frame"), "Size")
        self.assertZero(0, decl.add_member("Float", "foo"))
        self.assertZero(0, decl.remove_member("foo"))

    def test_IVoid(self):
        ivoid: pymdlsdk.IVoid = self.sdk.transaction.create_as(pymdlsdk.IVoid, "Void")
        self.assertIsValidInterface(ivoid)
        self.assertEqual(ivoid.get_type_name(), "Void")
        wrong_interface: pymdlsdk.IString = self.sdk.transaction.create_as(pymdlsdk.IString, "Void")
        self.assertIsNotNone(wrong_interface)
        self.assertFalse(wrong_interface.is_valid_interface())
        wrong_name: pymdlsdk.IVoid = self.sdk.transaction.create_as(pymdlsdk.IVoid, "FOO123")
        self.assertIsNotNone(wrong_name)
        self.assertFalse(wrong_name.is_valid_interface())

    def test_IDynamic_array(self):
        array: pymdlsdk.IDynamic_array = self.sdk.transaction.create_as(pymdlsdk.IDynamic_array, "String[]")
        self.assertIsValidInterface(array)
        element0: pymdlsdk.IString = self.sdk.transaction.create_as(pymdlsdk.IString, "String")
        element1: pymdlsdk.IString = self.sdk.transaction.create_as(pymdlsdk.IString, "String")
        element0.set_c_str("foo")
        element1.set_c_str("bar")
        self.assertTrue(array.empty())
        self.assertIsNotValidInterface(array.front_as(pymdlsdk.IString))  # empty
        self.assertIsNotValidInterface(array.back_as(pymdlsdk.IString))  # empty
        self.assertIsNotValidInterface(array.get_element_as(pymdlsdk.IString, 0))  # empty
        array.set_length(1)
        self.assertFalse(array.empty())
        array.set_element(0, element0)
        array.push_back(element1)
        self.assertEqual(array.get_length(), 2)
        self.run_IData_collection_tests(pymdlsdk.IDynamic_array, array)
        newValueOfElement1: str = 'blub'
        self.assertEqual(element0.get_c_str(), array.front().get_interface(pymdlsdk.IString).get_c_str())
        self.assertEqual(element0.get_c_str(), array.front_as(pymdlsdk.IString).get_c_str())
        self.assertEqual(element0.get_c_str(), array.front_as(pymdlsdk.IString).get_c_str())
        self.assertIsNotValidInterface(array.front_as(pymdlsdk.IFloat32))  # wrong interface type

        self.assertEqual(newValueOfElement1, array.back().get_interface(pymdlsdk.IString).get_c_str())
        self.assertEqual(newValueOfElement1, array.back_as(pymdlsdk.IString).get_c_str())
        self.assertEqual(element0.get_c_str(), array.get_element(0).get_interface(pymdlsdk.IString).get_c_str())
        self.assertEqual(newValueOfElement1, array.get_element_as(pymdlsdk.IString, 1).get_c_str())
        self.assertIsNotValidInterface(array.get_element_as(pymdlsdk.IBoolean, 1))  # wrong interface type

        self.assertZero(array.insert(1, element1))
        self.assertEqual(array.get_length(), 3)
        self.assertEqual(element0.get_c_str(), array.get_element_as(pymdlsdk.IString, 0).get_c_str())
        self.assertEqual(element1.get_c_str(), array.get_element_as(pymdlsdk.IString, 1).get_c_str())
        self.assertEqual(newValueOfElement1, array.get_element_as(pymdlsdk.IString, 2).get_c_str())
        self.assertZero(array.erase(0))
        self.assertEqual(array.get_length(), 2)
        self.assertEqual(element1.get_c_str(), array.get_element_as(pymdlsdk.IString, 0).get_c_str())
        self.assertEqual(newValueOfElement1, array.get_element_as(pymdlsdk.IString, 1).get_c_str())
        self.assertZero(array.pop_back())
        self.assertEqual(array.get_length(), 1)
        self.assertEqual(element1.get_c_str(), array.get_element_as(pymdlsdk.IString, 0).get_c_str())
        array.clear()
        self.assertEqual(array.get_length(), 0)

    def test_constructors(self):
        self.assertFalse(pymdlsdk.Boolean_2_struct().y)
        self.assertFalse(pymdlsdk.Boolean_3_struct().z)
        self.assertFalse(pymdlsdk.Boolean_4_struct().w)
        self.assertFalse(pymdlsdk.Boolean_2_2_struct().yy)
        self.assertFalse(pymdlsdk.Boolean_2_3_struct().yz)
        self.assertFalse(pymdlsdk.Boolean_2_4_struct().yw)
        self.assertFalse(pymdlsdk.Boolean_3_2_struct().zy)
        self.assertFalse(pymdlsdk.Boolean_3_3_struct().zz)
        self.assertFalse(pymdlsdk.Boolean_3_4_struct().zw)
        self.assertFalse(pymdlsdk.Boolean_4_2_struct().wy)
        self.assertFalse(pymdlsdk.Boolean_4_3_struct().wz)
        self.assertFalse(pymdlsdk.Boolean_4_4_struct().ww)

        self.assertEqual(pymdlsdk.Sint32_2_struct().y, 0)
        self.assertEqual(pymdlsdk.Sint32_3_struct().z, 0)
        self.assertEqual(pymdlsdk.Sint32_4_struct().w, 0)
        self.assertEqual(pymdlsdk.Sint32_2_2_struct().yy, 0)
        self.assertEqual(pymdlsdk.Sint32_2_3_struct().yz, 0)
        self.assertEqual(pymdlsdk.Sint32_2_4_struct().yw, 0)
        self.assertEqual(pymdlsdk.Sint32_3_2_struct().zy, 0)
        self.assertEqual(pymdlsdk.Sint32_3_3_struct().zz, 0)
        self.assertEqual(pymdlsdk.Sint32_3_4_struct().zw, 0)
        self.assertEqual(pymdlsdk.Sint32_4_2_struct().wy, 0)
        self.assertEqual(pymdlsdk.Sint32_4_3_struct().wz, 0)
        self.assertEqual(pymdlsdk.Sint32_4_4_struct().ww, 0)

        self.assertEqual(pymdlsdk.Uint32_2_struct().y, 0)
        self.assertEqual(pymdlsdk.Uint32_3_struct().z, 0)
        self.assertEqual(pymdlsdk.Uint32_4_struct().w, 0)
        self.assertEqual(pymdlsdk.Uint32_2_2_struct().yy, 0)
        self.assertEqual(pymdlsdk.Uint32_2_3_struct().yz, 0)
        self.assertEqual(pymdlsdk.Uint32_2_4_struct().yw, 0)
        self.assertEqual(pymdlsdk.Uint32_3_2_struct().zy, 0)
        self.assertEqual(pymdlsdk.Uint32_3_3_struct().zz, 0)
        self.assertEqual(pymdlsdk.Uint32_3_4_struct().zw, 0)
        self.assertEqual(pymdlsdk.Uint32_4_2_struct().wy, 0)
        self.assertEqual(pymdlsdk.Uint32_4_3_struct().wz, 0)
        self.assertEqual(pymdlsdk.Uint32_4_4_struct().ww, 0)

        self.assertEqual(pymdlsdk.Float32_2_struct().y, 0)
        self.assertEqual(pymdlsdk.Float32_3_struct().z, 0)
        self.assertEqual(pymdlsdk.Float32_4_struct().w, 0)
        self.assertEqual(pymdlsdk.Float32_2_2_struct().yy, 0)
        self.assertEqual(pymdlsdk.Float32_2_3_struct().yz, 0)
        self.assertEqual(pymdlsdk.Float32_2_4_struct().yw, 0)
        self.assertEqual(pymdlsdk.Float32_3_2_struct().zy, 0)
        self.assertEqual(pymdlsdk.Float32_3_3_struct().zz, 0)
        self.assertEqual(pymdlsdk.Float32_3_4_struct().zw, 0)
        self.assertEqual(pymdlsdk.Float32_4_2_struct().wy, 0)
        self.assertEqual(pymdlsdk.Float32_4_3_struct().wz, 0)
        self.assertEqual(pymdlsdk.Float32_4_4_struct().ww, 0)

        self.assertEqual(pymdlsdk.Float64_2_struct().y, 0)
        self.assertEqual(pymdlsdk.Float64_3_struct().z, 0)
        self.assertEqual(pymdlsdk.Float64_4_struct().w, 0)
        self.assertEqual(pymdlsdk.Float64_2_2_struct().yy, 0)
        self.assertEqual(pymdlsdk.Float64_2_3_struct().yz, 0)
        self.assertEqual(pymdlsdk.Float64_2_4_struct().yw, 0)
        self.assertEqual(pymdlsdk.Float64_3_2_struct().zy, 0)
        self.assertEqual(pymdlsdk.Float64_3_3_struct().zz, 0)
        self.assertEqual(pymdlsdk.Float64_3_4_struct().zw, 0)
        self.assertEqual(pymdlsdk.Float64_4_2_struct().wy, 0)
        self.assertEqual(pymdlsdk.Float64_4_3_struct().wz, 0)
        self.assertEqual(pymdlsdk.Float64_4_4_struct().ww, 0)

        uuid: pymdlsdk.Uuid = pymdlsdk.Uuid()
        self.assertEqual(uuid.m_id1, 0)
        self.assertEqual(uuid.m_id2, 0)
        self.assertEqual(uuid.m_id3, 0)
        self.assertEqual(uuid.m_id4, 0)

        color: pymdlsdk.Color_struct = pymdlsdk.Color_struct()
        self.assertEqual(color.r, 0)
        self.assertEqual(color.g, 0)
        self.assertEqual(color.b, 0)
        self.assertEqual(color.a, 0)

    def test_iinterface_functions(self):
        namesAtomics: dict = {
            "Boolean": pymdlsdk.IBoolean,
            "Float32": pymdlsdk.IFloat32,
            "Float64": pymdlsdk.IFloat64,
            "Uint8": pymdlsdk.IUint8,
            "Uint16": pymdlsdk.IUint16,
            "Uint32": pymdlsdk.IUint32,
            "Uint64": pymdlsdk.IUint64,
            "Sint8": pymdlsdk.ISint8,
            "Sint16": pymdlsdk.ISint16,
            "Sint32": pymdlsdk.ISint32,
            "Sint64": pymdlsdk.ISint64,
            "Size": pymdlsdk.ISize,
            "Difference": pymdlsdk.IDifference,
            "String": pymdlsdk.IString,
        }
        namesVectors: dict = {
            "Boolean<2>": pymdlsdk.IBoolean_2,
            "Sint32<2>": pymdlsdk.ISint32_2,
            "Uint32<2>": pymdlsdk.IUint32_2,
            "Float32<2>": pymdlsdk.IFloat32_2,
            "Float64<2>": pymdlsdk.IFloat64_2,
            "Boolean<3>": pymdlsdk.IBoolean_3,
            "Sint32<3>": pymdlsdk.ISint32_3,
            "Uint32<3>": pymdlsdk.IUint32_3,
            "Float32<3>": pymdlsdk.IFloat32_3,
            "Float64<3>": pymdlsdk.IFloat64_3,
            "Boolean<4>": pymdlsdk.IBoolean_4,
            "Sint32<4>": pymdlsdk.ISint32_4,
            "Uint32<4>": pymdlsdk.IUint32_4,
            "Float32<4>": pymdlsdk.IFloat32_4,
            "Float64<4>": pymdlsdk.IFloat64_4,
            "Color": pymdlsdk.IColor,
            "Color3": pymdlsdk.IColor3,
        }
        namesMatrices: dict = {
            "Boolean<2,2>": pymdlsdk.IBoolean_2_2,
            "Sint32<2,2>": pymdlsdk.ISint32_2_2,
            "Uint32<2,2>": pymdlsdk.IUint32_2_2,
            "Float32<2,2>": pymdlsdk.IFloat32_2_2,
            "Float64<2,2>": pymdlsdk.IFloat64_2_2,
            "Boolean<3,2>": pymdlsdk.IBoolean_3_2,
            "Sint32<3,2>": pymdlsdk.ISint32_3_2,
            "Uint32<3,2>": pymdlsdk.IUint32_3_2,
            "Float32<3,2>": pymdlsdk.IFloat32_3_2,
            "Float64<3,2>": pymdlsdk.IFloat64_3_2,
            "Boolean<4,2>": pymdlsdk.IBoolean_4_2,
            "Sint32<4,2>": pymdlsdk.ISint32_4_2,
            "Uint32<4,2>": pymdlsdk.IUint32_4_2,
            "Float32<4,2>": pymdlsdk.IFloat32_4_2,
            "Float64<4,2>": pymdlsdk.IFloat64_4_2,

            "Boolean<2,3>": pymdlsdk.IBoolean_2_3,
            "Sint32<2,3>": pymdlsdk.ISint32_2_3,
            "Uint32<2,3>": pymdlsdk.IUint32_2_3,
            "Float32<2,3>": pymdlsdk.IFloat32_2_3,
            "Float64<2,3>": pymdlsdk.IFloat64_2_3,
            "Boolean<3,3>": pymdlsdk.IBoolean_3_3,
            "Sint32<3,3>": pymdlsdk.ISint32_3_3,
            "Uint32<3,3>": pymdlsdk.IUint32_3_3,
            "Float32<3,3>": pymdlsdk.IFloat32_3_3,
            "Float64<3,3>": pymdlsdk.IFloat64_3_3,
            "Boolean<4,3>": pymdlsdk.IBoolean_4_3,
            "Sint32<4,3>": pymdlsdk.ISint32_4_3,
            "Uint32<4,3>": pymdlsdk.IUint32_4_3,
            "Float32<4,3>": pymdlsdk.IFloat32_4_3,
            "Float64<4,3>": pymdlsdk.IFloat64_4_3,

            "Boolean<2,4>": pymdlsdk.IBoolean_2_4,
            "Sint32<2,4>": pymdlsdk.ISint32_2_4,
            "Uint32<2,4>": pymdlsdk.IUint32_2_4,
            "Float32<2,4>": pymdlsdk.IFloat32_2_4,
            "Float64<2,4>": pymdlsdk.IFloat64_2_4,
            "Boolean<3,4>": pymdlsdk.IBoolean_3_4,
            "Sint32<3,4>": pymdlsdk.ISint32_3_4,
            "Uint32<3,4>": pymdlsdk.IUint32_3_4,
            "Float32<3,4>": pymdlsdk.IFloat32_3_4,
            "Float64<3,4>": pymdlsdk.IFloat64_3_4,
            "Boolean<4,4>": pymdlsdk.IBoolean_4_4,
            "Sint32<4,4>": pymdlsdk.ISint32_4_4,
            "Uint32<4,4>": pymdlsdk.IUint32_4_4,
            "Float32<4,4>": pymdlsdk.IFloat32_4_4,
            "Float64<4,4>": pymdlsdk.IFloat64_4_4,
        }
        namesAll: dict = {}
        namesAll.update(namesAtomics)
        namesAll.update(namesVectors)
        namesAll.update(namesMatrices)
        namesAll["Uvtile"] = pymdlsdk.IStructure
        namesAll["__Float32"] = pymdlsdk.INumber  # __ to have a unique key
        namesAll["__Float32<2>"] = pymdlsdk.ICompound
        namesAll["__Uint32<2,3>"] = pymdlsdk.ICompound

        for name, classType in namesAll.items():
            if name.startswith('__'):
                name = name[2:]
            iinterface = self.sdk.transaction.create_as(classType, name)
            self.assertIsValidInterface(iinterface)
            self.assertEqual(iinterface.get_type_name(), name)
            if classType == pymdlsdk.INumber:  # we use a specialized type here
                inumber: pymdlsdk.INumber = iinterface.get_interface(pymdlsdk.INumber)
                self.assertNotEqual(inumber.get_iid(), classType.IID())
                dataSimple: pymdlsdk.IData_simple = inumber.get_interface(pymdlsdk.IData_simple)
                self.assertIsValidInterface(dataSimple)
                self.assertEqual(dataSimple.get_type_name(), "Float32")
            elif classType == pymdlsdk.ICompound:  # we use a specialized type here
                icompound: pymdlsdk.ICompound = iinterface.get_interface(pymdlsdk.ICompound)
                self.assertNotEqual(icompound.get_iid(), classType.IID())
            else:
                self.assertEqual(iinterface.get_iid(), classType.IID())
            self.assertTrue(iinterface.compare_iid(classType.IID()))
            with self.sdk.transaction.create_as(classType, name) as iinterface2:
                self.assertIsValidInterface(iinterface2)
            iinterface3 = iinterface.get_interface(pymdlsdk.IInterface)
            self.assertIsValidInterface(iinterface3)
            if name == "String":
                iinterface.set_c_str("foo")
                self.assertEqual(iinterface.get_c_str(), "foo")
            elif name == "Uvtile":
                u: pymdlsdk.ISint32 = self.sdk.transaction.create_as(pymdlsdk.ISint32, "Sint32")
                u.set_value(1)
                self.assertEqual(u.get_value(), 1)
                v: pymdlsdk.ISint32 = self.sdk.transaction.create_as(pymdlsdk.ISint32, "Sint32")
                v.set_value(2)
                self.assertEqual(v.get_value(), 2)
                iinterface.set_value("u", u)
                iinterface.set_value("v", v)
                self.assertTrue(iinterface.has_key("u"))  # note has_key is not the deprecated python function here (NOT W601)
                self.assertTrue(iinterface.has_key("v"))
                self.assertEqual(iinterface.get_value("u").get_interface(pymdlsdk.ISint32).get_value(), 1)
                self.assertEqual(iinterface.get_value_as(pymdlsdk.ISint32, "v").get_value(), 2)
            elif classType == pymdlsdk.INumber or classType == pymdlsdk.ICompound:
                pass  # removed all get/set value functions, users need to cast to the specialization
            else:
                value = iinterface.get_value()
                self.assertIsNotNone(value)
                iinterface.set_value(value)
            if name in namesVectors.keys():
                r: int = iinterface.get_number_of_rows()
                self.assertEqual(1, iinterface.get_number_of_columns())
                self.assertEqual(r, iinterface.get_length())
                typename = iinterface.get_element_type_name()
                key: str = iinterface.get_key(0)
                self.assertTrue(iinterface.has_key(key))
                if name == "Color" or name == "Color3":
                    self.assertEqual(typename, "Float32")
                    if classType != pymdlsdk.ICompound:
                        data = iinterface.get_value()
                        data.r = True if typename == "Boolean" else 42
                        iinterface.set_value(data)
                        data2 = iinterface.get_value()
                        self.assertEqual(data.r, data2.r)
                else:
                    self.assertEqual(typename, name[:-3])
                    if classType != pymdlsdk.ICompound:
                        data = iinterface.get_value()
                        data.x = True if typename == "Boolean" else 42
                        iinterface.set_value(data)
                        data2 = iinterface.get_value()
                        self.assertEqual(data.x, data2.x)
            elif name in namesMatrices.keys():
                r: int = iinterface.get_number_of_rows()
                c: int = iinterface.get_number_of_columns()
                self.assertEqual(r * c, iinterface.get_length())
                typename = iinterface.get_element_type_name()
                self.assertEqual(typename, name[:-5])
                key: str = iinterface.get_key(0)
                self.assertTrue(iinterface.has_key(key))
                if classType != pymdlsdk.ICompound:
                    data = iinterface.get_value()
                    data.xx = True if typename == "Boolean" else 42
                    iinterface.set_value(data)
                    data2 = iinterface.get_value()
                    self.assertEqual(data.xx, data2.xx)


# run all tests of this file
if __name__ == '__main__':
    unittest.main()  # pragma: no cover
