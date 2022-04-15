# mkts_from_hybridts()

This method generates a hybrid timestamp based on an existing hybrid timestamp, timedelta and incremental time interval.

## Invocation

```python
mkts_from_hybridts(hybridts, milliseconds=0.0, delta=None)
```

## Parameters

| Parameter       | Description                                    | Type               | Required |
| --------------- | ---------------------------------------------- | ------------------ | -------- |
| `hybridts`      | The original hybrid timestamp                  | Non-negative integer range from 0 to 18446744073709551615                                                                    | True     |
| `milliseconds`  | Incremental time interval. Unit: milliseconds  | Float              | False    |
| `delta`         | A duration indicates the difference between two pieces of date, time, or datetime instances to microsecond resolution                                                         | datetime.timedelta | False    |

## Return

A new hybrid timestamp.
