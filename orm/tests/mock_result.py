class MockMutationResult:
    def __init__(self):
        self._primary_keys = []
        self._insert_cnt = 0
        self._delete_cnt = 0
        self._upsert_cnt = 0
        self._timestamp = 0

    @property
    def primary_keys(self):
        return self._primary_keys

    @property
    def insert_count(self):
        return self._insert_cnt

    @property
    def delete_count(self):
        return self._delete_cnt

    @property
    def upsert_count(self):
        return self._upsert_cnt

    @property
    def timestamp(self):
        return self._timestamp
