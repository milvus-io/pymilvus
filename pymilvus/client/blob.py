import struct

# reference: https://docs.python.org/3/library/struct.html#struct.pack

def boolToBytes(b):
    return struct.pack("?", b)

def int8ToBytes(i):
    return struct.pack("b", i)

def int16ToBytes(i):
    return struct.pack("h", i)

def int32ToBytes(i):
    return struct.pack("i", i)

def int64ToBytes(i):
    return struct.pack("q", i)

def floatToBytes(f):
    return struct.pack("f", f)

def doubleToBytes(d):
    return struct.pack("d", d)

def stringToBytes(s):
    return bytes(s, encoding='utf8')

def vectorBinaryToBytes(v):
    return bytes(v)

def vectorFloatToBytes(v):
    bs = bytes()
    for f in v:
        bs += floatToBytes(f)
    return bs

def bytesToInt64(v):
    return struct.unpack("q", v)[0]
