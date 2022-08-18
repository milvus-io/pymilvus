from .connections import connections

"""
Role, can be granted privileges which are allowed to execute some objects' apis.
"""
class Role:
    """ Constructs a role by name
    :param name: role name.
    :type  name: str
    """
    def __init__(self, name: str, using="default", **kwargs):
        self._name = name
        self._using = using
        self._kwargs = kwargs

    def _get_connection(self):
        return connections._fetch_handler(self._using)

    @property
    def name(self):
        return self._name

    """ Create a role
    It will success if the role isn't existed, otherwise fail.
    """
    def create(self):
        return self._get_connection().create_role(self._name)

    """ Drop a role
    It will success if the role is existed, otherwise fail.
    """
    def drop(self):
        return self._get_connection().drop_role(self._name)

    """ Add user to role
    The user will get permissions that the role are allowed to perform operations.
    :param username: user name.
    :type  username: str
    """
    def add_user(self, username: str):
        return self._get_connection().add_user_to_role(username, self._name)

    """ Remove user from role
    The user will remove permissions that the role are allowed to perform operations.
    :param username: user name.
    :type  username: str
    """
    def remove_user(self, username: str):
        return self._get_connection().remove_user_from_role(username, self._name)

    """ Get all users who are added to the role.
    :return a RoleInfo object which contains a RoleItem group
        According to the RoleItem, you can get a list of usernames.
        RoleInfo groups:
        - UserItem: <role_name:admin>, <users:('root',)>
    """
    def get_users(self):
        roles = self._get_connection().select_one_role(self._name, True)
        if len(roles.groups) == 0:
            return []
        return roles.groups[0].users

    """ Check whether the role is existed.
    :return a bool value
        It will be True if the role is existed, otherwise False.
    """
    def is_exist(self):
        roles = self._get_connection().select_one_role(self._name, False)
        return len(roles.groups) != 0

    """ Grant a privilege for the role
    :param object: object type.
    :type  object: str
    :param object_name: identifies a specific object name.
    :type  object_name: str
    :param privilege: privilege name.
    :type  privilege: str
    """
    def grant(self, object: str, object_name: str, privilege: str):
        return self._get_connection().grant_privilege(self._name, object, object_name, privilege)

    """ Revoke a privilege for the role
    :param object: object type.
    :type  object: str
    :param object_name: identifies a specific object name.
    :type  object_name: str
    :param privilege: privilege name.
    :type  privilege: str
    """
    def revoke(self, object: str, object_name: str, privilege: str):
        return self._get_connection().revoke_privilege(self._name, object, object_name, privilege)

    """ List a grant info for the role and the specific object
    :param object: object type.
    :type  object: str
    :param object_name: identifies a specific object name.
    :type  object_name: str
    :return a GrantInfo object
        GrantInfo groups:
        - GrantItem: <object:Collection>, <object_name:foocol2>, <role_name:general31>, <grantor_name:root>, <privilege:Load>
    :rtype GrantInfo
    """
    def list_grant(self, object: str, object_name: str):
        return self._get_connection().select_grant_for_role_and_object(self._name, object, object_name)

    """ List a grant info for the role
        :return a GrantInfo object
            GrantInfo groups:
            - GrantItem: <object:Collection>, <object_name:foocol2>, <role_name:general31>, <grantor_name:root>, <privilege:Load>
        :rtype GrantInfo
        """
    def list_grants(self):
        return self._get_connection().select_grant_for_one_role(self._name)
