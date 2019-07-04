class Status(object):
    """
    :attribute code: int (optional) default as ok

    :attribute message: str (optional) current status message
    """
    SUCCESS = 0
    UNEXPECTED_ERROR = 1
    CONNECT_FAILED = 2
    PERMISSION_DENIED = 3
    TABLE_NOT_EXISTS = 4
    ILLEGAL_ARGUMENT = 5
    ILLEGAL_RANGE = 6
    ILLEGAL_DIMENSION = 7
    ILLEGAL_INDEX_TYPE = 8
    ILLEGAL_TABLE_NAME = 9
    ILLEGAL_TOPK = 10
    ILLEGAL_ROWRECORD = 11
    ILLEGAL_VECTOR_ID = 12
    ILLEGAL_SEARCH_RESULT = 13
    FILE_NOT_FOUND = 14
    META_FAILED = 15
    CACHE_FAILED = 16
    CANNOT_CREATE_FOLDER = 17
    CANNOT_CREATE_FILE = 18
    CANNOT_DELETE_FOLDER = 19
    CANNOT_DELETE_FILE = 20

    def __init__(self, code=SUCCESS, message=None):
        self.code = code
        self.message = message

    def __repr__(self):
        L = ['%s=%r' % (key, value)
             for key, value in self.__dict__.items()]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(L))

    def __eq__(self, other):
        """Make Status comparable with self by code"""
        if isinstance(other, int):
            return self.code == other
        else:
            return isinstance(other, self.__class__) and self.code == other.code

    def __ne__(self, other):
        return not (self == other)

    def OK(self):
        return self.code == Status.SUCCESS

