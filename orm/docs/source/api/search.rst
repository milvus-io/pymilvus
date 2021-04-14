.. _search:

Search
=========
.. currentmodule:: milvus_orm

Search .
SearchResult.
Hits.
Hit.

Constructor 
-----------
.. autosummary::
   :toctree: api/
   :template: autosummaryclass.rst

   Search


Search Methods
---------------------
.. autosummary::
   :toctree: api/

    Search.execute

SearchResult Methods
---------------------
.. autosummary::
   :toctree: api/

    SearchResult.__iter__
    SearchResult.__len__    
    SearchResult.done

Hits Methods
---------------------
.. autosummary::
   :toctree: api/

    Hits.__iter__
    Hits.__len__

Hits Attributes
---------------------
.. autosummary::
   :toctree: api/
   
    Hits.ids
    Hits.distances

Hit Attributes
---------------------
.. autosummary::
   :toctree: api/

    Hit.id
    Hit.score    
    Hit.distance