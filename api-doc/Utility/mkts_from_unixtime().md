# mkts_from_unixtime()

This method generates a hybrid timestamp based on Unix Epoch time, timedelta and incremental time interval.

## Invocation

```python
mkts_from_unixtime(epoch, milliseconds=0.0, delta=None)
```

## Parameters

| Parameter       | Description                                    | Type               | Required |
| --------------- | ---------------------------------------------- | ------------------ | -------- |
| `epoch`         | Unix Epoch time                                | Integer            | True     |
| `milliseconds`  | Incremental time interval. Unit: milliseconds  | Float              | False    |
| `delta`         | A duration indicates the difference between two pieces of date, time, or datetime instances to microsecond resolution                                                         | datetime.timedelta | False    |

## Return

A new hybrid timestamp.
