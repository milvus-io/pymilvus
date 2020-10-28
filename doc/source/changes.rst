============
Chang logs
============


** V0.3.0 **
  * Incompatibly upgrade APIs for supporting hybrid functionality. The passing parameters of the following APIs has been changed:
    - create_collection()
    - insert()
    - create_index()
    - drop_index()
    - search()
    - search_in_segment()
    - get_entity_by_id()

  * Add passing parameter `threshold` in API compact()
  * Raise exception if the status of returned values from server is not OK for whole APIs.
  * Remove parameter `status` in return values of APIs except compact() and delete_entity_by_id() where status is reserved experimentally.
  * Change the passing parameters of get_config() and set_config()