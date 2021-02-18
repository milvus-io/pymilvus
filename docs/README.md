# Documentation Structure 

All the documentation files store under folder `source`. There are two default file:
  - **conf.py**: configuration file about documentation.
  - **index.rst**: corresponding to the index page of documentation website. The rest other page
                   need to be displaced here. For example, there is a page named `param`, and under 
                   folder `source` a file named `param.rst` exists, `param` need to be added in index.rst
                   under `.. toctree::`.

                   
# Documentation Update
Except index.rst, each other rst file correspond to a page on documentation website. Add a new rst file if 
you want to add a new page on website or modify rst files to update existing page contents.


# Browser documentation locally

## Setup environment
    ```shell
    $ pip install -r requirements.txt
    ```

## Build documentation locally
    ```shell
    $ cd docs
    $ make html
    ```
The documentation could be generated under directory build/html.

## Browser documentation locally
To preview it, you can open index.html in your browser.

Or run a web server in directory `build/html`:
    ```
    $ python -m http.server
    ```
Then open your browser to `http://localhost:8000`.

# Submit documentation change
After documentation changed, please use Git to store you modification and push to your remote repository,
then pull a new request to repository `milvus-io/pymilvus`.
 
