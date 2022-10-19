from .connections import connections

class Role:
    """ Role, can be granted privileges which are allowed to execute some objects' apis. """

    def __init__(self, name: str, using="default", **kwargs):
        """ Constructs a role by name
            :param name: role name.
            :type  name: str
        """
        self._name = name
        self._using = using
        self._kwargs = kwargs

    def _get_connection(self):
        return connections._fetch_handler(self._using)

    @property
    def name(self):
        return self._name

    def create(self):
        """ Create a role
            It will success if the role isn't existed, otherwise fail.
        """
        return self._get_connection().create_role(self._name)

    def drop(self):
        """ Drop a role
            It will success if the role is existed, otherwise fail.
        """
        return self._get_connection().drop_role(self._name)

    def add_user(self, username: str):
        """ Add user to role
            The user will get permissions that the role are allowed to perform operations.
            :param username: user name.
            :type  username: str
        """
        return self._get_connection().add_user_to_role(username, self._name)

    def remove_user(self, username: str):
        """ Remove user from role
            The user will remove permissions that the role are allowed to perform operations.
            :param username: user name.
            :type  username: str
        """
        return self._get_connection().remove_user_from_role(username, self._name)

    def get_users(self):
        """ Get all users who are added to the role.
            :return a RoleInfo object which contains a RoleItem group
                According to the RoleItem, you can get a list of usernames.

            RoleInfo groups:
            - UserItem: <role_name:admin>, <users:('root',)>
        """
        roles = self._get_connection().select_one_role(self._name, True)
        if len(roles.groups) == 0:
            return []
        return roles.groups[0].users

    def is_exist(self):
        """ Check whether the role is existed.
            :return a bool value
                It will be True if the role is existed, otherwise False.
        """
        roles = self._get_connection().select_one_role(self._name, False)
        return len(roles.groups) != 0

    def grant(self, object: str, object_name: str, privilege: str):
        """ Grant a privilege for the role
            :param object: object type.
            :type  object: str
            :param object_name: identifies a specific object name.
            :type  object_name: str
            :param privilege: privilege name.
            :type  privilege: str
        """
        return self._get_connection().grant_privilege(self._name, object, object_name, privilege)

    def revoke(self, object: str, object_name: str, privilege: str):
        """ Revoke a privilege for the role
            :param object: object type.
            :type  object: str
            :param object_name: identifies a specific object name.
            :type  object_name: str
            :param privilege: privilege name.
            :type  privilege: str
        """
        return self._get_connection().revoke_privilege(self._name, object, object_name, privilege)

    def list_grant(self, object: str, object_name: str):
        """ List a grant info for the role and the specific object
            :param object: object type.
            :type  object: str
            :param object_name: identifies a specific object name.
            :type  object_name: str
            :return a GrantInfo object
            :rtype GrantInfo

            GrantInfo groups:
            - GrantItem: <object:Collection>, <object_name:foo>, <role_name:x>, <grantor_name:root>, <privilege:Load>
        """
        return self._get_connection().select_grant_for_role_and_object(self._name, object, object_name)

    def list_grants(self):
        """ List a grant info for the role
            :return a GrantInfo object
            :rtype GrantInfo

            GrantInfo groups:
            - GrantItem: <object:Collection>, <object_name:foo>, <role_name:x>, <grantor_name:root>, <privilege:Load>
        """
        return self._get_connection().select_grant_for_one_role(self._name)
