import threading


class Singleton(type):
    _ins = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._ins:
            with cls._lock:
                if cls not in cls._ins:
                    ins = super().__call__(*args, **kwargs)
                    cls._ins[cls] = ins
        return cls._ins[cls]
