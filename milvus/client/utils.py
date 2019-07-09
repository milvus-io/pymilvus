import datetime
from .Exceptions import ParamError


def is_legal_array(array):
    if not array or not isinstance(array, list):
        return False
    return True


def int_or_str(item):
    if isinstance(item, int):
        return str(item)
    else:
        return item


def is_correct_date_str(param):
    try:
        datetime.datetime.strptime(param, '%Y-%m-%d')
    except ValueError:
        raise ParamError('Incorrect data format, should be YYYY-MM-DD')
    return True


def legal_dimension(dim):
    if not isinstance(dim, int) or dim <= 0 or dim > 16384:
        return False
    return True


def parser_range_date(date):
    if isinstance(date, datetime.date):
        return date.strftime('%Y-%m-%d')
    elif isinstance(date, str):
        if not is_correct_date_str(date):
            raise ParamError('Date string should be YY-MM-DD format!')
        else:
            return date
    else:
        raise ParamError(
            'Date should be YY-MM-DD format string or datetime.date, '
            'or datetime.datetime object')


def is_legal_date_range(start, end):
    start_date = datetime.datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(end, "%Y-%m-%d")
    if (end_date - start_date).days < 0:
        return False
    else:
        return True


