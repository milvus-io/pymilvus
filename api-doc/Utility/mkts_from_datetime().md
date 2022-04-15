# mkts_from_datetime()

This method generates a hybrid timestamp based on datetime, timedelta and incremental time interval.

## Invocation

```python
mkts_from_datetime(d_time, milliseconds=0.0, delta=None)
```

## Parameters

| Parameter       | Description                                    | Type               | Required |
| --------------- | ---------------------------------------------- | ------------------ | -------- |
| `d_time`        | Datetime                                       | datetime.datetime  | True     |
| `milliseconds`  | Incremental time interval. Unit: milliseconds  | Float              | False    |
| `delta`         | A duration indicates the difference between two pieces of date, time, or datetime instances to microsecond resolution                                                         | datetime.timedelta | False    |

## Return

A new hybrid timestamp.
