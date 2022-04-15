# hybridts_to_unixtime()

This method converts a hybrid timestamp to datetime according to timezone.

## Invocation

```python
hybridts_to_datetime(hybridts, tz=None)
```

## Parameters

| Parameter       | Description                                    | Type               | Required |
| --------------- | ---------------------------------------------- | ------------------ | -------- |
| `hybridts`      | Hybrid timestamp                               | Integer            | True     |
| `tz`            | Timezone defined by a fixed offset from UTC. If argument `tz` is set `None` or not specified, the hybrid timestamp is converted to the local date and time of the platform.              | datetime.timezone  | True     |

## Return

The datetime object.

## Raises

`Exception`: error if `tz` is not of type datetime.timezone.