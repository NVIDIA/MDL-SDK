import unittest
import pymdlsdk
from setup import SDK

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

    def test_setupIsDone(self):
        self.assertIsNotNone(self.sdk)
        self.assertIsNotNone(self.sdk.neuray)
        self.assertIsNotNone(self.sdk.transaction)

    def test_IString(self):
        i: pymdlsdk.IInterface = self.sdk.transaction.create("String", 0, None) # using 'create' at least once
        self.assertTrue(i.is_valid_interface(), "Invalid IInterface")           # returns an IInterface
        a: pymdlsdk.IString = i.get_interface(pymdlsdk.IString)                 # needs casting
        self.assertTrue(a.is_valid_interface(), "Invalid IInterface")
        value = a.get_c_str()
        self.assertEqual(value, "")
        value = "MDL"
        a.set_c_str(value)
        value2 = a.get_c_str()
        self.assertEqual(value, value2) 

    def test_ISize(self):
        a: pymdlsdk.ISize = self.sdk.transaction.create_as(pymdlsdk.ISize, "Size", 0, None)
        self.assertTrue(a.is_valid_interface(), "Invalid IInterface")
        value = a.get_value()
        self.assertEqual(value, 0)
        value = 12345
        a.set_value(value)
        value2 = a.get_value()
        self.assertEqual(value, value2)

    def test_IFloat32(self):
        a: pymdlsdk.IFloat32 = self.sdk.transaction.create_as(pymdlsdk.IFloat32, "Float32")
        self.assertTrue(a.is_valid_interface(), "Invalid IInterface")
        value = a.get_value()
        self.assertEqual(value, 0)
        value = -1.2
        a.set_value(value)
        value2 = a.get_value()
        self.assertAlmostEqual(value, value2) # floating point off by 0.0000000476837158

    def test_IUint8(self):
        a: pymdlsdk.IUint8 = self.sdk.transaction.create_as(pymdlsdk.IUint8, "Uint8")
        self.assertTrue(a.is_valid_interface(), "Invalid IInterface")
        value = a.get_value()
        self.assertEqual(value, 0)
        value = 234
        a.set_value(value)
        value2 = a.get_value()
        self.assertEqual(value, value2)

    def test_IColor(self):
        a: pymdlsdk.IColor = self.sdk.transaction.create_as(pymdlsdk.IColor, "Color")
        self.assertTrue(a.is_valid_interface(), "Invalid IInterface")
        value = a.get_value()
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
        self.assertTrue(a.is_valid_interface(), "Invalid IInterface")
        value = a.get_value()
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
        self.assertTrue(a.is_valid_interface(), "Invalid IInterface")
        value = a.get_value()
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
        self.assertTrue(a.is_valid_interface(), "Invalid IInterface")
        value = a.get_value()
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
        self.assertTrue(a.is_valid_interface(), "Invalid IInterface")
        value = a.get_value()
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
        self.assertTrue(a.is_valid_interface(), "Invalid IInterface")
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
        self.assertTrue(a.is_valid_interface(), "Invalid IInterface")
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

    def test_IArray(self):
        array: pymdlsdk.IArray = self.sdk.transaction.create_as(pymdlsdk.IArray, "Float32[4]")
        self.assertTrue(array.is_valid_interface(), "Invalid IInterface")
        self.assertEqual(array.get_length(), 4)
        element0: pymdlsdk.IFloat32 = array.get_element_as(pymdlsdk.IFloat32, 0)
        element1: pymdlsdk.IFloat32 = array.get_element(1).get_interface(pymdlsdk.IFloat32)
        element2: pymdlsdk.IFloat32 = array.get_element(2).get_interface(pymdlsdk.IFloat32)
        element3: pymdlsdk.IFloat32 = array.get_element(3).get_interface(pymdlsdk.IFloat32)
        self.assertTrue(element0.is_valid_interface(), "Invalid IInterface")
        self.assertTrue(element1.is_valid_interface(), "Invalid IInterface")
        self.assertTrue(element2.is_valid_interface(), "Invalid IInterface")
        self.assertTrue(element3.is_valid_interface(), "Invalid IInterface")
        self.assertEqual(element0.get_value(), 0)
        self.assertEqual(element1.get_value(), 0)
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

    def test_IDynamic_array(self):
        array: pymdlsdk.IDynamic_array = self.sdk.transaction.create_as(pymdlsdk.IDynamic_array, "String[]")
        self.assertTrue(array.is_valid_interface(), "Invalid IInterface")
        element0: pymdlsdk.IString = self.sdk.transaction.create_as(pymdlsdk.IString, "String")
        element1: pymdlsdk.IString = self.sdk.transaction.create_as(pymdlsdk.IString, "String")
        element0.set_c_str("foo")
        element1.set_c_str("bar")
        array.push_back(element0)
        array.push_back(element1)
        self.assertEqual(array.get_length(), 2)
        self.assertEqual(element0.get_c_str(), array.front().get_interface(pymdlsdk.IString).get_c_str())
        self.assertEqual(element0.get_c_str(), array.front_as(pymdlsdk.IString).get_c_str())
        self.assertEqual(element1.get_c_str(), array.back().get_interface(pymdlsdk.IString).get_c_str())
        self.assertEqual(element1.get_c_str(), array.back_as(pymdlsdk.IString).get_c_str())
        self.assertEqual(element0.get_c_str(), array.get_element(0).get_interface(pymdlsdk.IString).get_c_str())
        self.assertEqual(element1.get_c_str(), array.get_element_as(pymdlsdk.IString, 1).get_c_str())
        array.clear()
        self.assertEqual(array.get_length(), 0)
        
    def test_IStructure(self):
        tile: pymdlsdk.IStructure = self.sdk.transaction.create_as(pymdlsdk.IStructure, "Uvtile")
        self.assertTrue(tile.is_valid_interface(), "Invalid IInterface")
        u: pymdlsdk.ISint32 = self.sdk.transaction.create_as(pymdlsdk.ISint32, "Sint32")
        u.set_value(1)
        self.assertEqual(u.get_value(), 1)
        v: pymdlsdk.ISint32 = self.sdk.transaction.create_as(pymdlsdk.ISint32, "Sint32")
        v.set_value(2)
        self.assertEqual(v.get_value(), 2)
        tile.set_value("u", u)
        tile.set_value("v", v)
        self.assertTrue(tile.has_key("u"))
        self.assertTrue(tile.has_key("v"))
        self.assertFalse(tile.has_key("not_a_member"))
        self.assertEqual(u.get_value(), tile.get_value("u").get_interface(pymdlsdk.ISint32).get_value())
        self.assertEqual(v.get_value(), tile.get_value_as(pymdlsdk.ISint32, "v").get_value())
        
# run all tests of this file
if __name__ == '__main__':
    unittest.main()